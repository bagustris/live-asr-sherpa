#!/usr/bin/env bash
# Evaluate all Japanese ASR models on adlib-devterm and JVNV, then update the leaderboard.
#
# Usage: benchmark/run_ja_benchmark.sh [OPTIONS]
#
# Options:
#   --force                  Re-run even if output JSON already exists
#   --precision PREC         ONNX precision variant: int8 | fp16 | fp32
#                            Only applied to models that have multiple variants
#                            (reazonspeech-ja, reazonspeech-ja-en, reazonspeech-ja-en-mls-5k,
#                             sense_voice). Models with a single variant ignore this flag.
#   --jvnv-dir DIR           Path to extracted JVNV v1 corpus (default: /data/jvnv_v1)
#   --dataset DATASET        Which datasets to run: adlib | jvnv | all  (default: all)
#   --model MODEL_TYPE       Run only this model type; can be repeated
#                            Valid values: parakeet-ctc-ja, reazonspeech-ja,
#                              reazonspeech-ja-en, reazonspeech-ja-en-mls-5k,
#                              whisper, sense_voice, cohere_transcribe
#   -h, --help               Show this help and exit

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="$REPO_ROOT/benchmark"
MODELS_DIR="$REPO_ROOT/models"

# Prefer the project venv so the script works without `source .venv/bin/activate`
PYTHON="${REPO_ROOT}/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python3

# ── Defaults ──────────────────────────────────────────────────────────────────
FORCE=0
PRECISION=""
JVNV_DIR="/data/jvnv_v1"
RUN_ADLIB=1
RUN_JVNV=1
SELECTED_MODELS=()

# ── Model registry ────────────────────────────────────────────────────────────
# Fields: model_type | model_dir_basename | has_precision_variants
# has_precision_variants=yes → --precision applies; no → always uses the single
# available variant (declared int8-only in the sherpa-onnx release).
ALL_MODELS=(
  "parakeet-ctc-ja|parakeet-ctc-ja-int8|no"
  "reazonspeech-ja|reazonspeech-ja|yes"
  "reazonspeech-ja-en|reazonspeech-ja-en|yes"
  "reazonspeech-ja-en-mls-5k|reazonspeech-ja-en-mls-5k|yes"
  "whisper|sherpa-onnx-whisper-large-v3|no"
  "sense_voice|sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17|yes"
  "cohere_transcribe|cohere-transcribe-14-lang-int8|no"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
die()  { echo "Error: $*" >&2; exit 1; }
info() { echo "  $*"; }

usage() {
  # Print the header comment block (lines 2–N until the first non-comment line)
  awk 'FNR==1{next} /^#/{print substr($0,3); next} {exit 0}' "$0"
  exit 0
}

# Download model if not already present by calling sherox's _validate_model.
ensure_model() {
  local model_type="$1" model_dir="$2"
  "$PYTHON" - <<PYEOF
import sys; sys.path.insert(0, "$REPO_ROOT")
from sherox.asr import _validate_model
_validate_model("$model_dir", "$model_type")
PYEOF
}

# Build a temp directory containing symlinks to only the requested precision
# variant. Prints the temp dir path; prints nothing if no matching files exist.
make_prec_dir() {
  local src="$1" prec="$2"
  local tmpdir
  tmpdir=$(mktemp -d)

  # Vocabulary file(s)
  for f in "$src"/*tokens.txt; do
    [[ -e "$f" ]] && ln -sf "$f" "$tmpdir/"
  done

  # ONNX files filtered by precision
  if [[ "$prec" == "fp32" ]]; then
    for f in "$src"/*.onnx; do
      local n; n=$(basename "$f")
      [[ "$n" == *.int8.onnx || "$n" == *.fp16.onnx ]] && continue
      [[ -e "$f" ]] && ln -sf "$f" "$tmpdir/"
    done
  else
    for f in "$src"/*."$prec".onnx; do
      [[ -e "$f" ]] && ln -sf "$f" "$tmpdir/"
    done
  fi

  local count
  count=$(find "$tmpdir" -name "*.onnx" 2>/dev/null | wc -l)
  if [[ "$count" -eq 0 ]]; then
    rm -rf "$tmpdir"
    return
  fi

  printf "%s" "$tmpdir"
}

# Overwrite model_dir and add precision field in a result JSON so the leaderboard
# can group by the real model directory regardless of which temp dir was used.
patch_json() {
  PATCH_FILE="$1" PATCH_MODEL_DIR="$2" PATCH_PRECISION="$3" "$PYTHON" - <<'PYEOF'
import json, os
path = os.environ["PATCH_FILE"]
with open(path) as f:
    d = json.load(f)
d["model_dir"]  = os.environ["PATCH_MODEL_DIR"]
d["precision"]  = os.environ["PATCH_PRECISION"]
with open(path, "w") as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
PYEOF
}

# Return 0 if a result JSON already exists for this model+dataset combination.
# Checks the canonical filename first (fast path), then scans all results*.json
# files by content so older results saved under different names are recognised.
result_exists() {
  local canonical_file="$1" model_dir_base="$2" ds_field="$3"
  [[ -f "$canonical_file" ]] && return 0
  MODEL_DIR_BASE="$model_dir_base" DS_FIELD="$ds_field" \
  BENCH_DIR="$BENCH_DIR" REPO_ROOT="$REPO_ROOT" "$PYTHON" - <<'PYEOF'
import json, os, sys
from pathlib import Path
bench = Path(os.environ["BENCH_DIR"])
root  = Path(os.environ["REPO_ROOT"])
want_dir = os.environ["MODEL_DIR_BASE"]
want_ds  = os.environ["DS_FIELD"]
for p in [*bench.glob("results*.json"), *root.glob("results*.json")]:
    try:
        d = json.loads(p.read_text())
        if Path(d.get("model_dir", "")).name == want_dir and d.get("dataset", "") == want_ds:
            sys.exit(0)
    except Exception:
        pass
sys.exit(1)
PYEOF
}

# Global cleanup on exit/interrupt
PREC_TMPDIR=""
cleanup() { if [[ -n "$PREC_TMPDIR" ]]; then rm -rf "$PREC_TMPDIR"; fi; }
trap cleanup EXIT INT TERM

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --precision)
      PRECISION="${2:?--precision requires an argument}"
      [[ "$PRECISION" =~ ^(int8|fp16|fp32)$ ]] \
        || die "--precision must be int8, fp16, or fp32"
      shift 2
      ;;
    --jvnv-dir)
      JVNV_DIR="${2:?--jvnv-dir requires a path}"
      shift 2
      ;;
    --dataset)
      case "${2:-}" in
        adlib) RUN_ADLIB=1; RUN_JVNV=0 ;;
        jvnv)  RUN_ADLIB=0; RUN_JVNV=1 ;;
        all)   RUN_ADLIB=1; RUN_JVNV=1 ;;
        *) die "--dataset must be adlib, jvnv, or all" ;;
      esac
      shift 2
      ;;
    --model)
      SELECTED_MODELS+=("${2:?--model requires a value}")
      shift 2
      ;;
    -h|--help) usage ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# ── Main loop ─────────────────────────────────────────────────────────────────
cd "$REPO_ROOT"

RAN=0 SKIPPED=0

for entry in "${ALL_MODELS[@]}"; do
  IFS='|' read -r model_type model_dir_base has_variants <<< "$entry"

  # --model filter
  if [[ ${#SELECTED_MODELS[@]} -gt 0 ]]; then
    match=0
    for sel in "${SELECTED_MODELS[@]}"; do
      [[ "$sel" == "$model_type" ]] && match=1 && break
    done
    [[ $match -eq 0 ]] && continue
  fi

  full_model_dir="$MODELS_DIR/$model_dir_base"

  # Precision suffix (only for models that expose multiple variants)
  if [[ "$has_variants" == "yes" && -n "$PRECISION" ]]; then
    prec="$PRECISION"
    prec_suffix="_${PRECISION}"
  else
    prec=""
    prec_suffix=""
  fi

  adlib_out="$BENCH_DIR/results_${model_dir_base}${prec_suffix}_adlib.json"
  jvnv_out="$BENCH_DIR/results_${model_dir_base}${prec_suffix}_jvnv.json"

  # Skip check — canonical filename first, then JSON content scan
  need_adlib=$RUN_ADLIB
  need_jvnv=$RUN_JVNV
  if [[ $FORCE -eq 0 ]]; then
    result_exists "$adlib_out" "$model_dir_base" "holotherapper/adlib-devterm" && need_adlib=0
    result_exists "$jvnv_out"  "$model_dir_base" "JVNV"                        && need_jvnv=0
  fi
  if [[ $need_adlib -eq 0 && $need_jvnv -eq 0 ]]; then
    echo "skip  $model_type${prec_suffix:+ ($PRECISION)} — results exist (--force to rerun)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  echo ""
  echo "══ ${model_type}${prec_suffix:+ ($PRECISION)} ══"

  # Auto-download
  info "ensuring model is present…"
  ensure_model "$model_type" "$full_model_dir"

  # Precision-filtered dir
  PREC_TMPDIR=""
  if [[ -n "$prec" ]]; then
    PREC_TMPDIR=$(make_prec_dir "$full_model_dir" "$prec")
    if [[ -z "$PREC_TMPDIR" ]]; then
      echo "  WARNING: no $prec files in $model_dir_base — skipping"
      continue
    fi
    effective_dir="$PREC_TMPDIR"
  else
    effective_dir="$full_model_dir"
  fi

  # adlib-devterm
  if [[ $need_adlib -eq 1 ]]; then
    info "running adlib-devterm…"
    "$PYTHON" benchmark/benchmark_ja.py --offline \
      --model-type "$model_type" \
      --model-dir  "$effective_dir" \
      --language   ja \
      --output     "$adlib_out"
    [[ -n "$prec" ]] && patch_json "$adlib_out" "$full_model_dir" "$prec"
    RAN=$((RAN + 1))
  else
    info "adlib-devterm: skipping (result exists)"
  fi

  # JVNV
  if [[ $need_jvnv -eq 1 ]]; then
    if [[ ! -d "$JVNV_DIR" ]]; then
      info "WARNING: JVNV dir '$JVNV_DIR' not found — skipping (pass --jvnv-dir)"
    else
      info "running JVNV…"
      "$PYTHON" benchmark/benchmark_jvnv.py --offline \
        --model-type "$model_type" \
        --model-dir  "$effective_dir" \
        --jvnv-dir   "$JVNV_DIR" \
        --language   ja \
        --output     "$jvnv_out"
      [[ -n "$prec" ]] && patch_json "$jvnv_out" "$full_model_dir" "$prec"
      RAN=$((RAN + 1))
    fi
  else
    info "JVNV: skipping (result exists)"
  fi

  # Clean up precision temp dir
  if [[ -n "$PREC_TMPDIR" ]]; then
    rm -rf "$PREC_TMPDIR"
    PREC_TMPDIR=""
  fi

done

echo ""
echo "Done — ran $RAN benchmark(s), skipped $SKIPPED model(s)."
echo "Updating leaderboard…"
"$PYTHON" benchmark/update_asr_leaderboard_jp.py
