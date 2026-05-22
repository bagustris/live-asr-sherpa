#!/usr/bin/env bash
# Run the Japanese ASR diarization benchmark across all available models
# and regenerate ASR_DIARIZATION_LEADERBOARD_JP.md at the end.
#
# Usage:
#   bash benchmark/run_diarization_benchmark.sh [options]
#
# Options:
#   --dataset {callhome|sakura|both}   Default: both
#   --max-convs N                      Limit conversations per run (smoke test)
#   --threads N                        CPU threads per model  (default: 4)
#   --num-speakers N                   Known speakers (-1 = auto, default: -1)
#   --models MODEL[,MODEL,...]         Comma-separated list to run a subset
#                                      (default: all models below)
#   --output-dir DIR                   Where to write JSON results
#                                      (default: benchmark/)
#   --skip-existing                    Skip a model if its result JSON already exists
#   -h, --help                         Show this help
#
# Examples:
#   # Full benchmark, all models
#   bash benchmark/run_diarization_benchmark.sh
#
#   # Smoke test: 2 conversations, callhome only
#   bash benchmark/run_diarization_benchmark.sh --dataset callhome --max-convs 2
#
#   # Single model
#   bash benchmark/run_diarization_benchmark.sh --models reazonspeech-ja
#
#   # Skip already-evaluated models
#   bash benchmark/run_diarization_benchmark.sh --skip-existing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Model registry  (name → model-type flag for benchmark_diarization.py)
# ---------------------------------------------------------------------------
# Format: "model_name|model_type"
# model_name    : used for the output filename and display
# model_type    : value passed to --model-type
ALL_MODELS=(
    "parakeet-ctc-ja|parakeet-ctc-ja"
    "reazonspeech-ja|reazonspeech-ja"
    "reazonspeech-ja-en|reazonspeech-ja-en"
    "reazonspeech-ja-en-mls-5k|reazonspeech-ja-en-mls-5k"
    "whisper-large-v3|whisper"
    "sense-voice|sense_voice"
    "cohere-transcribe|cohere_transcribe"
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATASET="both"
MAX_CONVS=""
THREADS=4
NUM_SPEAKERS=-1
SELECTED_MODELS=""
OUTPUT_DIR="$SCRIPT_DIR"
SKIP_EXISTING=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET="$2";         shift 2 ;;
        --max-convs)     MAX_CONVS="$2";        shift 2 ;;
        --threads)       THREADS="$2";          shift 2 ;;
        --num-speakers)  NUM_SPEAKERS="$2";     shift 2 ;;
        --models)        SELECTED_MODELS="$2";  shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";       shift 2 ;;
        --skip-existing) SKIP_EXISTING=true;    shift   ;;
        -h|--help)
            sed -n '/^# Usage/,/^[^#]/p' "$0" | head -n -1 | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Filter model list if --models was given
# ---------------------------------------------------------------------------
if [[ -n "$SELECTED_MODELS" ]]; then
    IFS=',' read -ra WANTED <<< "$SELECTED_MODELS"
    FILTERED=()
    for entry in "${ALL_MODELS[@]}"; do
        name="${entry%%|*}"
        for want in "${WANTED[@]}"; do
            if [[ "$name" == "$want" ]]; then
                FILTERED+=("$entry")
                break
            fi
        done
    done
    if [[ ${#FILTERED[@]} -eq 0 ]]; then
        echo "Error: none of the requested models matched." >&2
        echo "Available: $(IFS=', '; echo "${ALL_MODELS[*]%%|*}")" >&2
        exit 1
    fi
    ALL_MODELS=("${FILTERED[@]}")
fi

# ---------------------------------------------------------------------------
# Build benchmark command base
# ---------------------------------------------------------------------------
BENCH_CMD=(
    python "$SCRIPT_DIR/benchmark_diarization.py"
    --dataset "$DATASET"
    --offline
    --threads "$THREADS"
    --num-speakers "$NUM_SPEAKERS"
    --language ja
)
[[ -n "$MAX_CONVS" ]] && BENCH_CMD+=(--max-convs "$MAX_CONVS")

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Run each model
# ---------------------------------------------------------------------------
RESULTS=()
FAILED=()
TOTAL=${#ALL_MODELS[@]}

echo "=========================================="
echo " Japanese ASR Diarization Benchmark"
echo " Dataset   : $DATASET"
echo " Models    : $TOTAL"
echo " Threads   : $THREADS"
echo " Output    : $OUTPUT_DIR"
[[ -n "$MAX_CONVS" ]] && echo " Max convs : $MAX_CONVS"
echo "=========================================="
echo ""

for entry in "${ALL_MODELS[@]}"; do
    MODEL_NAME="${entry%%|*}"
    MODEL_TYPE="${entry##*|}"
    OUTPUT_FILE="$OUTPUT_DIR/results_diarization_${MODEL_NAME}.json"

    echo "------------------------------------------"
    echo " Model : $MODEL_NAME  (type: $MODEL_TYPE)"
    echo "------------------------------------------"

    if $SKIP_EXISTING && [[ -f "$OUTPUT_FILE" ]]; then
        echo " Skipping — result already exists: $OUTPUT_FILE"
        RESULTS+=("$OUTPUT_FILE")
        continue
    fi

    set +e
    "${BENCH_CMD[@]}" \
        --model-type "$MODEL_TYPE" \
        --output "$OUTPUT_FILE"
    EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo ""
        echo " WARNING: $MODEL_NAME failed (exit $EXIT_CODE) — skipping."
        FAILED+=("$MODEL_NAME")
    else
        RESULTS+=("$OUTPUT_FILE")
        echo ""
        echo " Saved: $OUTPUT_FILE"
    fi
    echo ""
done

# ---------------------------------------------------------------------------
# Update leaderboard
# ---------------------------------------------------------------------------
echo "=========================================="
echo " Updating ASR_DIARIZATION_LEADERBOARD_JP.md"
echo "=========================================="

if [[ ${#RESULTS[@]} -eq 0 ]]; then
    echo "No successful results — skipping leaderboard update."
    exit 1
fi

python "$SCRIPT_DIR/update_asr_diarization_leaderboard_jp.py"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo " Done"
echo "=========================================="
echo " Evaluated : ${#RESULTS[@]} / $TOTAL model(s)"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo " Failed    : ${FAILED[*]}"
fi
echo " Leaderboard: $SCRIPT_DIR/ASR_DIARIZATION_LEADERBOARD_JP.md"
echo "=========================================="
