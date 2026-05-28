#!/usr/bin/env python3
"""Regenerate benchmark/ASR_LEADERBOARD_JP.md from results JSON files.

Scans `benchmark/` and the repo root for `results*.json`, groups by model
directory, and writes an HF Open ASR Leaderboard-style summary table.
JSON files are the source of truth; the markdown is fully derived.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = REPO_ROOT / "benchmark"
OUTPUT = BENCHMARK_DIR / "ASR_LEADERBOARD_JP.md"

DATASETS = {
    "holotherapper/adlib-devterm": "ADLIB-DEVTERM",
    "JVNV": "JVNV",
    "japanese-asr/ja_asr.jsut_basic5000": "JSUT5000",
    "JVS": "JVS",
}
DATASET_ORDER = ["ADLIB-DEVTERM", "JVNV", "JSUT5000", "JVS"]

MODEL_LABELS = {
    "parakeet-ctc-ja-int8": "Parakeet CTC JA (int8)",
    "sherpa-onnx-whisper-large-v3": "Whisper Large-V3",
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17": "SenseVoice",
    "reazonspeech-ja": "ReazonSpeech JA",
    "reazonspeech-ja-en": "ReazonSpeech JA-EN",
    "reazonspeech-ja-en-mls-5k": "ReazonSpeech JA-EN-MLS-5k",
    "cohere-transcribe-14-lang-int8": "Cohere Transcribe (int8)",
}


def discover() -> list[Path]:
    files = list(BENCHMARK_DIR.glob("results*.json"))
    files.extend(REPO_ROOT.glob("results*.json"))
    return files


def load() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in discover():
        with path.open() as f:
            data = json.load(f)
        ds_label = DATASETS.get(data.get("dataset", ""))
        if not ds_label:
            continue
        model_dir = Path(data.get("model_dir", "?")).name
        precision = data.get("precision") or ""   # "" / "auto" → no suffix
        if precision == "auto":
            precision = ""
        row_key = f"{model_dir}|{precision}"
        base_label = MODEL_LABELS.get(model_dir, model_dir)
        label = f"{base_label} ({precision})" if precision else base_label
        agg = data.get("aggregate", {})
        entry = rows.setdefault(row_key, {
            "label": label,
            "datasets": {},
        })
        entry["datasets"][ds_label] = {
            "cer": agg.get("cer"),
            "ker": agg.get("ker"),
            "rtf": agg.get("mean_rtf"),
            "composite": agg.get("composite_score"),
            "source": path.relative_to(REPO_ROOT).as_posix(),
        }
    return rows


def avg(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def fmt_pct(x):
    return f"{x * 100:.2f}%" if x is not None else "—"


def fmt_rtf(x):
    return f"{x:.4f}" if x is not None else "—"


def fmt_score(x):
    return f"{x:.4f}" if x is not None else "—"


def render(rows: dict[str, dict]) -> str:
    summary = []
    for entry in rows.values():
        per_ds = entry["datasets"]
        kers = [per_ds[d]["ker"] for d in DATASET_ORDER if d in per_ds and per_ds[d].get("ker") is not None]
        cers = [per_ds[d]["cer"] for d in DATASET_ORDER if d in per_ds]
        rtfs = [per_ds[d]["rtf"] for d in DATASET_ORDER if d in per_ds]
        scores = [per_ds[d]["composite"] for d in DATASET_ORDER if d in per_ds]
        summary.append({
            "label": entry["label"],
            "avg_ker": avg(kers),
            "avg_cer": avg(cers),
            "avg_rtf": avg(rtfs),
            "avg_score": avg(scores),
            "per_ds": per_ds,
        })
    summary.sort(key=lambda r: (r["avg_ker"] is None, r["avg_ker"] or 0))

    sources = sorted({
        d["source"]
        for entry in rows.values()
        for d in entry["datasets"].values()
    })

    out = []
    out.append("# Japanese ASR Leaderboard")
    out.append("")
    out.append(
        "HF Open ASR Leaderboard–style ranking of Japanese ASR models on "
        "[adlib-devterm](https://huggingface.co/datasets/holotherapper/adlib-devterm), "
        "[JVNV v1](https://ss-takashi.sakura.ne.jp/corpus/jvnv/), "
        "[JSUT Basic5000](https://huggingface.co/datasets/japanese-asr/ja_asr.jsut_basic5000), and "
        "[JVS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus). "
        "Lower is better for every column."
    )
    out.append("")
    out.append(
        "Numbers come from `benchmark/benchmark_ja.py`, `benchmark/benchmark_jvnv.py`, "
        "`benchmark/benchmark_jsut.py`, and `benchmark/benchmark_jvs.py` "
        "(offline pipeline, 4 threads, `--language ja`). Regenerate this file with "
        "`python benchmark/update_asr_leaderboard_jp.py`."
    )
    out.append("")
    # Build header with all dataset columns
    ds_ker_headers = " | ".join(f"{d} KER" for d in DATASET_ORDER)
    out.append(f"| Model | Avg KER | Avg CER | Avg RTF | Avg Composite | {ds_ker_headers} |")
    separator = "|-------|--------:|--------:|--------:|--------------:|" + \
                "".join("----------:|" for _ in DATASET_ORDER)
    out.append(separator)
    for r in summary:
        ds_ker_cols = " | ".join(
            fmt_pct(r["per_ds"].get(d, {}).get("ker")) for d in DATASET_ORDER
        )
        out.append(
            f"| {r['label']} | {fmt_pct(r['avg_ker'])} | {fmt_pct(r['avg_cer'])} "
            f"| {fmt_rtf(r['avg_rtf'])} | "
            f"{fmt_score(r['avg_score'])} | {ds_ker_cols} |"
        )
    out.append("")
    out.append("## Metric definitions")
    out.append("")
    out.append("- **KER** — Kana Error Rate (text converted to hiragana/kana before edit distance)")
    out.append("- **CER** — character error rate (NFKC + lowercase + punctuation-stripped)")
    out.append("- **RTF** — real-time factor (processing time / audio duration)")
    out.append("- **Composite** — `(KER + RTF) / 2`")
    out.append("")
    out.append("KER is the primary metric: it handles kanji/kana equivalence "
               "(e.g., 東京 ≡ とうきょう) by converting text to hiragana via pyopenjtalk G2P "
               "before comparison, making it a more phonetically accurate measure of ASR quality.")
    out.append("")
    out.append("Full details in [`benchmark/README.md`](README.md).")
    out.append("")
    out.append("## Reproduce")
    out.append("")
    out.append("```bash")
    out.append("MODEL=parakeet-ctc-ja   # or reazonspeech-ja, whisper, sense_voice, ...")
    out.append("")
    out.append("python benchmark/benchmark_ja.py --offline --model-type \"$MODEL\" \\")
    out.append("  --output \"benchmark/results_${MODEL}_adlib.json\"")
    out.append("")
    out.append("python benchmark/benchmark_jvnv.py --offline --jvnv-dir /data/jvnv_v1 \\")
    out.append("  --model-type \"$MODEL\" --output \"benchmark/results_${MODEL}_jvnv.json\"")
    out.append("")
    out.append("python benchmark/benchmark_jsut.py --offline --model-type \"$MODEL\" \\")
    out.append("  --output \"benchmark/results_${MODEL}_jsut.json\"")
    out.append("")
    out.append("python benchmark/benchmark_jvs.py --offline --jvs-dir /data/jvs_ver1 \\")
    out.append("  --model-type \"$MODEL\" --output \"benchmark/results_${MODEL}_jvs.json\"")
    out.append("")
    out.append("python benchmark/update_asr_leaderboard_jp.py")
    out.append("```")
    out.append("")
    out.append("## Sources")
    out.append("")
    for s in sources:
        out.append(f"- `{s}`")
    return "\n".join(out) + "\n"


def main() -> None:
    rows = load()
    if not rows:
        raise SystemExit(
            "No results found. Run a benchmark with --output, then re-run this script."
        )
    OUTPUT.write_text(render(rows))
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(rows)} model(s))")


if __name__ == "__main__":
    main()
