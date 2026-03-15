#!/usr/bin/env python3
"""
live-asr-sherpa benchmark — evaluate WER, RTF, latency, and composite score
for any model supported by live-asr-sherpa on LibriSpeech audio.

Metrics reported (per utterance and aggregate):
    WER             Word Error Rate — transcription accuracy (lower is better)
    RTF             Real-Time Factor — processing speed (lower is better)
    Latency (ms)    Wall-clock time to produce each transcription (lower is better)
    Composite Score (WER + mean_RTF) / 2 — single ranking metric (lower is better)

Input — one of:
    --data-dir PATH   LibriSpeech split directory (auto-builds manifest)
    --manifest FILE   Pre-built TSV/CSV: audio_path <sep> reference_text

Model arguments are shared with src/main.py:
    --model-dir, --model-type, --offline, --threads, --language,
    --sample-rate, --chunk-size

Usage examples:
    # Benchmark default offline model (Parakeet TDT) on LibriSpeech dev-clean-2
    python benchmark.py --data-dir /data/LibriSpeech/dev-clean-2 --offline

    # Quick smoke test: first 20 utterances
    python benchmark.py --data-dir /data/LibriSpeech/dev-clean-2 \\
        --offline --max-utts 20 --verbose

    # Benchmark online Zipformer streaming (simulated)
    python benchmark.py --data-dir /data/LibriSpeech/dev-clean-2

    # Benchmark Whisper offline, save results
    python benchmark.py --data-dir /data/LibriSpeech/dev-clean-2 \\
        --offline --model-type whisper --model-dir ../models/whisper-small \\
        --output results_whisper.json

    # Benchmark SenseVoice with Chinese audio
    python benchmark.py --data-dir /data/AISHELL/test \\
        --offline --model-type sense_voice --language zh \\
        --model-dir ../models/sense_voice

    # Use a pre-built manifest
    python benchmark.py --manifest data/test-clean.tsv --offline
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

# Ensure benchmark/ is on sys.path so we can import metrics from here
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

# Also ensure src/ is on sys.path to import asr_engine, config
_SRC_DIR = _BENCH_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from metrics import AggregateMetrics, UtteranceResult  # noqa: E402

# Default paths
DEFAULT_DATA_DIR = "/data/LibriSpeech/dev-clean-2"
_PROJECT_DIR = _BENCH_DIR.parent
DEFAULT_ONLINE_MODEL_DIR = str(_PROJECT_DIR / "models" / "zipformer-en-2023")
DEFAULT_OFFLINE_MODEL_DIR = str(_PROJECT_DIR / "models" / "parakeet-tdt-0.6b-v2")


# ---------------------------------------------------------------------------
# Manifest generation from LibriSpeech directory
# ---------------------------------------------------------------------------

def manifest_from_librispeech(data_dir: str) -> List[Tuple[str, str]]:
    """Walk a LibriSpeech split directory; return (audio_path, reference) pairs.

    Expected structure:
        <data_dir>/<speaker>/<chapter>/<utt_id>.flac
        <data_dir>/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt

    References are lowercased to match normalised ASR output.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    records: List[Tuple[str, str]] = []
    for trans_file in sorted(data_path.rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        with open(trans_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) < 2:
                    continue
                utt_id, text = parts
                for ext in (".flac", ".wav"):
                    audio_path = chapter_dir / f"{utt_id}{ext}"
                    if audio_path.exists():
                        records.append((str(audio_path), text.lower()))
                        break

    if not records:
        print(f"Error: no audio files found under {data_dir}", file=sys.stderr)
        sys.exit(1)

    return records


# ---------------------------------------------------------------------------
# Manifest loading from TSV/CSV file
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> List[Tuple[str, str]]:
    """Parse a TSV or CSV manifest into (audio_path, reference_text) pairs."""
    records: List[Tuple[str, str]] = []
    manifest_dir = Path(path).parent

    with open(path, newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 2:
                continue
            audio_path = row[0].strip()
            reference = row[1].strip()
            if not Path(audio_path).is_absolute():
                audio_path = str(manifest_dir / audio_path)
            records.append((audio_path, reference))

    if not records:
        print(f"Error: no valid records in manifest: {path}", file=sys.stderr)
        sys.exit(1)
    return records


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, float]:
    """Load an audio file; convert to mono float32 and resample if needed."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        try:
            import resampy
            audio = resampy.resample(audio, sr, target_sr)
        except ImportError:
            raise RuntimeError(
                f"Cannot resample {sr}→{target_sr} Hz for {path}: "
                "resampy not installed. Install with: pip install resampy"
            )

    duration = len(audio) / target_sr
    return audio, duration


# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------

def transcribe_offline(recognizer, audio: np.ndarray, sample_rate: int) -> str:
    """Transcribe a complete audio segment using an OfflineRecognizer."""
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def transcribe_online(
    recognizer,
    audio: np.ndarray,
    sample_rate: int,
    chunk_size: float = 0.1,
) -> str:
    """Simulate streaming by feeding audio in chunks to an OnlineRecognizer.

    Collects all endpoint-finalised text segments and joins them with spaces.
    A short tail of silence is added at the end to flush any buffered audio.
    """
    stream = recognizer.create_stream()
    chunk_samples = max(1, int(chunk_size * sample_rate))
    texts: List[str] = []

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]
        stream.accept_waveform(sample_rate, chunk)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        if recognizer.is_endpoint(stream):
            text = recognizer.get_result(stream).strip()
            if text:
                texts.append(text)
            recognizer.reset(stream)

    # Flush any remaining audio with a short silence tail
    tail = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail)
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    text = recognizer.get_result(stream).strip()
    if text:
        texts.append(text)

    return " ".join(texts)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    recognizer,
    records: List[Tuple[str, str]],
    offline: bool = True,
    sample_rate: int = 16000,
    chunk_size: float = 0.1,
    verbose: bool = False,
) -> Tuple[List[UtteranceResult], AggregateMetrics]:
    """Run all utterances through the recognizer; collect WER/RTF/latency."""
    results: List[UtteranceResult] = []

    for i, (audio_path, reference) in enumerate(records, 1):
        try:
            audio, duration = load_audio(audio_path, target_sr=sample_rate)
        except Exception as exc:
            print(f"  [{i:4d}] SKIP {Path(audio_path).name}: {exc}", flush=True)
            continue

        t_start = time.monotonic()
        try:
            if offline:
                hypothesis = transcribe_offline(recognizer, audio, sample_rate)
            else:
                hypothesis = transcribe_online(
                    recognizer, audio, sample_rate, chunk_size
                )
        except Exception as exc:
            print(f"  [{i:4d}] ERROR {Path(audio_path).name}: {exc}", flush=True)
            hypothesis = ""
        proc_time = time.monotonic() - t_start

        utt = UtteranceResult(
            audio_path=audio_path,
            reference=reference,
            hypothesis=hypothesis,
            audio_duration=duration,
            processing_time=proc_time,
        ).compute()

        results.append(utt)

        if verbose:
            print(
                f"  [{i:4d}/{len(records)}] "
                f"RTF={utt.rtf:.3f}  WER={utt.wer * 100:6.1f}%  "
                f"Lat={utt.latency_ms:.0f}ms"
            )
            print(f"    REF: {reference[:80]}")
            print(f"    HYP: {utt.hypothesis[:80]}")
        else:
            marker = "✓" if utt.wer == 0.0 else "✗"
            print(
                f"  [{i:4d}/{len(records)}] {marker}  "
                f"RTF={utt.rtf:.3f}  WER={utt.wer * 100:5.1f}%  "
                f"Lat={utt.latency_ms:.0f}ms",
                flush=True,
            )

    agg = AggregateMetrics.from_results(results)
    print(f"\n  ── Aggregate ──")
    print(f"  Utterances      : {agg.n_utterances}")
    print(f"  WER             : {agg.wer_pct:.2f}%")
    print(f"  Mean RTF        : {agg.mean_rtf:.4f}")
    print(f"  Mean Latency    : {agg.mean_latency_ms:.1f} ms")
    print(f"  Composite Score : {agg.composite_score:.4f}  (lower is better)")
    print(f"  Audio total     : {agg.total_audio_duration:.1f}s")
    print(f"  Proc total      : {agg.total_processing_time:.1f}s")

    return results, agg


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(model_name: str, agg: AggregateMetrics) -> None:
    """Print a formatted summary table for a single model run."""
    print("\n" + "=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    print(f"  Model            : {model_name}")
    print(f"  Utterances       : {agg.n_utterances}")
    print(f"  WER (%)          : {agg.wer_pct:.2f}")
    print(f"  Mean RTF         : {agg.mean_rtf:.4f}")
    print(f"  Mean Latency(ms) : {agg.mean_latency_ms:.1f}")
    print(f"  Composite Score  : {agg.composite_score:.4f}  (WER + mean_RTF) / 2")
    print("=" * 62)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark WER, RTF, latency, and composite score for "
            "live-asr-sherpa models on LibriSpeech (or custom) audio."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input source (mutually exclusive)
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        metavar="PATH",
        help="LibriSpeech split directory (auto-generates manifest)",
    )
    src.add_argument(
        "--manifest",
        metavar="FILE",
        help="Pre-built TSV/CSV manifest: audio_path <sep> reference_text",
    )

    # ── Model arguments (shared with src/main.py) ────────────────────────────
    p.add_argument(
        "--model-dir",
        default=None,
        metavar="PATH",
        help=(
            "Path to the model directory. "
            "Default: models/zipformer-en-2023 (online) or "
            "models/parakeet-tdt-0.6b-v2 (--offline)."
        ),
    )
    p.add_argument(
        "--model-type",
        default="",
        metavar="TYPE",
        help=(
            "Model architecture hint passed to sherpa-onnx. Leave blank for auto-detect. "
            "Online: transducer, zipformer, zipformer2, conformer, lstm, paraformer, "
            "ctc, wenet_ctc, zipformer2_ctc. "
            "Offline: transducer, nemo_transducer, paraformer, whisper, ctc, nemo_ctc, "
            "sense_voice, moonshine, fire_red_asr."
        ),
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Use the offline (non-streaming) pipeline. Required for Whisper, NeMo, "
            "SenseVoice, Moonshine, etc."
        ),
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz",
    )
    p.add_argument(
        "--chunk-size",
        type=float,
        default=0.1,
        help="Simulated streaming chunk size in seconds (online mode only)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU thread count for the ONNX runtime",
    )
    p.add_argument(
        "--language",
        default="en",
        metavar="LANG",
        help="Language code for Whisper and SenseVoice models (e.g. en, zh, ja)",
    )

    # ── Benchmark-specific arguments ─────────────────────────────────────────
    p.add_argument(
        "--max-utts",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N utterances (useful for quick smoke tests)",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-utterance REF/HYP lines",
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        help="Save full results to a JSON file",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Deferred import so the sys.path manipulation above takes effect first
    from config import Config
    from asr_engine import build_recognizer, build_offline_recognizer

    # Resolve model directory
    if args.model_dir is None:
        model_dir = DEFAULT_OFFLINE_MODEL_DIR if args.offline else DEFAULT_ONLINE_MODEL_DIR
    else:
        model_dir = args.model_dir

    # Load utterance records
    if args.manifest:
        records = load_manifest(args.manifest)
        print(f"Loaded {len(records)} utterances from {args.manifest}")
    else:
        records = manifest_from_librispeech(args.data_dir)
        print(f"Loaded {len(records)} utterances from {args.data_dir}")

    if args.max_utts is not None:
        records = records[: args.max_utts]
        print(f"Limited to first {len(records)} utterances (--max-utts)")

    # Build configuration
    cfg = Config(
        model_dir=model_dir,
        model_type=args.model_type,
        offline=args.offline,
        num_threads=args.threads,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        language=args.language,
    )

    # Load recognizer
    mode_str = "offline" if args.offline else "online (streaming)"
    print(f"\nLoading {mode_str} model: {Path(model_dir).name}")
    print(f"  model_type : '{cfg.model_type or '(auto-detect)'}'")
    print(f"  threads    : {cfg.num_threads}")
    t0 = time.monotonic()
    if args.offline:
        recognizer = build_offline_recognizer(cfg)
    else:
        recognizer = build_recognizer(cfg)
    load_time = time.monotonic() - t0
    print(f"  Loaded in {load_time:.1f}s\n")

    # Run benchmark
    results, agg = run_benchmark(
        recognizer,
        records,
        offline=args.offline,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        verbose=args.verbose,
    )

    # Print summary
    print_summary(Path(model_dir).name, agg)

    # Optionally save to JSON
    if args.output:
        output_data = {
            "model_dir": model_dir,
            "model_type": cfg.model_type,
            "offline": args.offline,
            "threads": cfg.num_threads,
            "aggregate": {
                "wer_pct": agg.wer_pct,
                "mean_rtf": agg.mean_rtf,
                "mean_latency_ms": agg.mean_latency_ms,
                "composite_score": agg.composite_score,
                "total_audio_duration_s": agg.total_audio_duration,
                "total_processing_time_s": agg.total_processing_time,
                "n_utterances": agg.n_utterances,
            },
            "utterances": [
                {
                    "audio_path": r.audio_path,
                    "reference": r.reference,
                    "hypothesis": r.hypothesis,
                    "audio_duration_s": r.audio_duration,
                    "processing_time_s": r.processing_time,
                    "latency_ms": r.latency_ms,
                    "edit_distance": r.edit_distance,
                    "wer": r.wer,
                    "rtf": r.rtf,
                }
                for r in results
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
