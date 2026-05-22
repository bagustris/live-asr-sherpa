#!/usr/bin/env python3
"""Regenerate benchmark/ASR_DIARIZATION_LEADERBOARD_JP.md from results JSON files.

Scans `benchmark/` and the repo root for `results_diarization*.json`, groups by
model directory, and writes a summary table with diarization metrics.
JSON files are the source of truth; the markdown is fully derived.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = REPO_ROOT / "benchmark"
OUTPUT = BENCHMARK_DIR / "ASR_DIARIZATION_LEADERBOARD_JP.md"

DATASETS = {
    "callhome-jpn": "CALLHOME-JPN",
    "sakura": "SAKURA",
    "mixed": None,  # multi-dataset runs — skip direct dataset attribution
}
DATASET_ORDER = ["CALLHOME-JPN", "SAKURA"]

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
    files = list(BENCHMARK_DIR.glob("results_diarization*.json"))
    files.extend(REPO_ROOT.glob("results_diarization*.json"))
    return files


def load() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in discover():
        with path.open() as f:
            data = json.load(f)

        # Determine dataset label
        raw_ds = data.get("dataset", "")
        ds_label = DATASETS.get(raw_ds)

        model_dir = Path(data.get("model_dir", "?")).name
        agg = data.get("aggregate", {})

        entry = rows.setdefault(model_dir, {
            "label": MODEL_LABELS.get(model_dir, model_dir),
            "datasets": {},
        })

        if ds_label and ds_label in DATASET_ORDER:
            entry["datasets"][ds_label] = {
                "der": agg.get("der"),
                "cer": agg.get("cer"),
                "cpcer": agg.get("cpcer"),
                "rtf": agg.get("mean_rtf"),
                "latency_ms": agg.get("mean_latency_ms"),
                "source": path.relative_to(REPO_ROOT).as_posix(),
            }
        elif raw_ds == "mixed":
            # Mixed-dataset run: use per-conversation data to split by dataset
            for conv in data.get("conversations", []):
                conv_ds = DATASETS.get(conv.get("dataset", ""), "")
                if not conv_ds or conv_ds not in DATASET_ORDER:
                    continue
                # Accumulate per-dataset conversation-level aggregation later
                entry["datasets"].setdefault(conv_ds, {
                    "_convs": [],
                    "source": path.relative_to(REPO_ROOT).as_posix(),
                })
                entry["datasets"][conv_ds]["_convs"].append(conv)

    # Finalise mixed-dataset aggregation
    for entry in rows.values():
        for ds_label, ds_data in entry["datasets"].items():
            convs = ds_data.pop("_convs", None)
            if convs is None:
                continue
            if not convs:
                entry["datasets"][ds_label] = {}
                continue
            total_audio = sum(c.get("audio_duration_s", 0) for c in convs)
            total_proc = sum(c.get("processing_time_s", 0) for c in convs)
            entry["datasets"][ds_label] = {
                "der": mean(c["der"] for c in convs),
                "cer": mean(c["cer"] for c in convs),
                "cpcer": mean(c["cpcer"] for c in convs),
                "rtf": total_proc / total_audio if total_audio > 0 else None,
                "latency_ms": total_proc / len(convs) * 1000 if convs else None,
                "source": ds_data["source"],
            }

    return rows


def avg(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def fmt_pct(x):
    return f"{x * 100:.2f}%" if x is not None else "—"


def fmt_rtf(x):
    return f"{x:.4f}" if x is not None else "—"


def fmt_lat(x):
    return f"{x:.0f}" if x is not None else "—"


def render(rows: dict[str, dict]) -> str:
    summary = []
    for entry in rows.values():
        per_ds = entry["datasets"]
        ders = [per_ds[d]["der"] for d in DATASET_ORDER if d in per_ds and per_ds[d].get("der") is not None]
        cers = [per_ds[d]["cer"] for d in DATASET_ORDER if d in per_ds and per_ds[d].get("cer") is not None]
        cpcers = [per_ds[d]["cpcer"] for d in DATASET_ORDER if d in per_ds and per_ds[d].get("cpcer") is not None]
        rtfs = [per_ds[d]["rtf"] for d in DATASET_ORDER if d in per_ds and per_ds[d].get("rtf") is not None]
        lats = [per_ds[d]["latency_ms"] for d in DATASET_ORDER if d in per_ds and per_ds[d].get("latency_ms") is not None]
        summary.append({
            "label": entry["label"],
            "avg_der": avg(ders),
            "avg_cer": avg(cers),
            "avg_cpcer": avg(cpcers),
            "avg_rtf": avg(rtfs),
            "avg_lat": avg(lats),
            "per_ds": per_ds,
        })
    summary.sort(key=lambda r: (r["avg_cpcer"] is None, r["avg_cpcer"] or 0))

    sources = sorted({
        ds_data["source"]
        for entry in rows.values()
        for ds_data in entry["datasets"].values()
        if ds_data.get("source")
    })

    out = []
    out.append("# Japanese ASR Diarization Leaderboard")
    out.append("")
    out.append(
        "Ranking of Japanese ASR models (with speaker diarization) on "
        "[Callhome Japan](https://huggingface.co/datasets/talkbank/callhome) and "
        "[Sakura](https://huggingface.co/datasets/talkbank/sakura). "
        "Lower is better for every column."
    )
    out.append("")
    out.append(
        "Numbers come from `benchmark/benchmark_diarization.py` "
        "(offline pipeline, `--diarization`, 4 threads, `--language ja`). "
        "Regenerate with `python benchmark/update_asr_diarization_leaderboard_jp.py`."
    )
    out.append("")
    out.append(
        "| Model | Avg RTF | Avg Latency (ms) | Avg DER | Avg CER | Avg cpCER"
        " | CALLHOME-JPN DER | SAKURA DER |"
    )
    out.append(
        "|-------|--------:|-----------------:|--------:|--------:|---------:"
        "|-----------------:|-----------:|"
    )
    for r in summary:
        der_ch = r["per_ds"].get("CALLHOME-JPN", {}).get("der")
        der_sk = r["per_ds"].get("SAKURA", {}).get("der")
        out.append(
            f"| {r['label']}"
            f" | {fmt_rtf(r['avg_rtf'])}"
            f" | {fmt_lat(r['avg_lat'])}"
            f" | {fmt_pct(r['avg_der'])}"
            f" | {fmt_pct(r['avg_cer'])}"
            f" | {fmt_pct(r['avg_cpcer'])}"
            f" | {fmt_pct(der_ch)}"
            f" | {fmt_pct(der_sk)} |"
        )
    out.append("")
    out.append("## Metric definitions")
    out.append("")
    out.append("- **RTF** — real-time factor (processing time / audio duration)")
    out.append("- **Latency** — mean per-conversation wall-clock time (ms)")
    out.append("- **DER** — diarization error rate = (missed + false alarm + speaker confusion) / total reference speech")
    out.append("  - Collar: ±250 ms around reference segment boundaries")
    out.append("  - Optimal speaker mapping via Hungarian algorithm")
    out.append("- **CER** — character error rate on chronologically concatenated transcript")
    out.append("  - Normalization: NFKC + lowercase + strip Japanese punctuation + strip whitespace")
    out.append("- **cpCER** — concatenated minimum-permutation CER")
    out.append("  - Groups text per speaker, finds optimal speaker-label mapping, computes CER")
    out.append("")
    out.append("Full details in [`benchmark/README.md`](README.md).")
    out.append("")
    out.append("## Reproduce")
    out.append("")
    out.append("```bash")
    out.append("MODEL=parakeet-ctc-ja   # or reazonspeech-ja, whisper, sense_voice, ...")
    out.append("")
    out.append("python benchmark/benchmark_diarization.py --dataset both --offline \\")
    out.append("  --model-type \"$MODEL\" \\")
    out.append("  --output \"benchmark/results_diarization_${MODEL}.json\"")
    out.append("")
    out.append("python benchmark/update_asr_diarization_leaderboard_jp.py")
    out.append("```")
    out.append("")
    if sources:
        out.append("## Sources")
        out.append("")
        for s in sources:
            out.append(f"- `{s}`")
    return "\n".join(out) + "\n"


def main() -> None:
    rows = load()
    if not rows:
        raise SystemExit(
            "No diarization results found. "
            "Run benchmark_diarization.py with --output, then re-run this script."
        )
    OUTPUT.write_text(render(rows))
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(rows)} model(s))")


if __name__ == "__main__":
    main()
