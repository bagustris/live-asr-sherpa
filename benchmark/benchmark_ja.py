#!/usr/bin/env python3
"""
Japanese ASR benchmark on the holotherapper/adlib-devterm HuggingFace dataset.

Evaluation follows the adlib protocol (https://github.com/holotherapper/adlib):
  - CER uses minimal normalization: NFC + newline removal only
    (fullwidth chars, punctuation, and case are kept as evaluation targets)
  - Flexible term replacement: before CER, alternative surface forms
    (e.g. "コンポーネント" → "component") are substituted so they do not
    incur a CER penalty
  - Term Accuracy measures whether technical IT terms appear in the output,
    with boundary-aware matching and flexible katakana/English equivalents
  - Composite Score = 0.4×(1−CER) + 0.6×TermAcc  (higher is better)
  - 95% Bootstrap confidence intervals (B=10,000, seed=42, ratio-based)

Additional metrics (not in adlib):
  RTF      Real-Time Factor
  Latency  Wall-clock time per utterance

The adlib test-case file (terms per utterance) is auto-fetched from GitHub
and cached at benchmark/devterm_test_cases.jsonl on first run.

Usage examples:
    # Benchmark default Japanese model (parakeet-ctc-ja)
    python benchmark_ja.py --offline

    # Smoke test: first 10 utterances, verbose
    python benchmark_ja.py --offline --max-utts 10 --verbose

    # Filter by category
    python benchmark_ja.py --offline --category backend

    # Save full results to JSON
    python benchmark_ja.py --offline --output benchmark/results_ja.json

    # ReazonSpeech model
    python benchmark_ja.py --offline --model-type reazonspeech-ja
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

_PROJECT_DIR = _BENCH_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from benchmark_utils import (  # noqa: E402
    _adlib_normalize,
    _bootstrap_ci,
    _compute_cer,
    _compute_term_accuracy,
    _content_normalize,
    _replace_flexible_terms,
    print_group_breakdown,
    transcribe_offline,
    transcribe_online,
)

DEFAULT_DATASET = "holotherapper/adlib-devterm"
DEFAULT_SPLIT = "test"
DEFAULT_ONLINE_MODEL_DIR = str(_PROJECT_DIR / "models" / "zipformer-ja")
DEFAULT_OFFLINE_MODEL_DIR = str(_PROJECT_DIR / "models" / "parakeet-ctc-ja-int8")
DEFAULT_OFFLINE_MODEL_TYPE = "parakeet-ctc-ja"

_MODEL_TYPE_ALIASES = {
    "parakeet-ctc-ja": "nemo_ctc",
    "reazonspeech-ja": "",
    "reazonspeech-ja-en": "",
    "reazonspeech-ja-en-mls-5k": "",
}

# adlib test-cases URL and local cache path
_ADLIB_CASES_URL = (
    "https://raw.githubusercontent.com/holotherapper/adlib/main/"
    "domains/devterm/dataset/test_cases.jsonl"
)
_ADLIB_CASES_CACHE = _BENCH_DIR / "devterm_test_cases.jsonl"


# ---------------------------------------------------------------------------
# Adlib test-case loading (terms per utterance)
# ---------------------------------------------------------------------------

def load_adlib_cases(cache_path: Path = _ADLIB_CASES_CACHE) -> Dict[str, Dict]:
    """Load adlib devterm test cases, fetching from GitHub if not cached.

    Returns a dict of {id: case_dict} where case_dict has 'reference' and 'terms'.
    """
    if not cache_path.exists():
        print(f"Fetching adlib test cases from GitHub → {cache_path.name} …", flush=True)
        try:
            urllib.request.urlretrieve(_ADLIB_CASES_URL, cache_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download adlib test cases: {exc}\n"
                f"  URL: {_ADLIB_CASES_URL}\n"
                f"  Save manually to: {cache_path}"
            ) from exc
        print(f"  Saved to {cache_path}", flush=True)

    cases: Dict[str, Dict] = {}
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cases[row["id"]] = row
    return cases


# ---------------------------------------------------------------------------
# HuggingFace dataset loading
# ---------------------------------------------------------------------------

def load_hf_samples(
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    category: Optional[str] = None,
    speaker: Optional[str] = None,
    max_utts: Optional[int] = None,
    target_sr: int = 16000,
) -> List[Dict]:
    """Stream the HuggingFace dataset and return decoded audio samples."""
    try:
        import soundfile as sf  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("pip install soundfile") from exc
    try:
        from datasets import Audio, load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("pip install datasets") from exc

    print(f"Loading dataset {dataset_name!r} split={split!r} …", flush=True)
    ds = load_dataset(dataset_name, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    samples: List[Dict] = []
    for sample in ds:
        if category and sample["category"] != category:
            continue
        if speaker and sample["speaker_id"] != speaker:
            continue

        audio_bytes = sample["audio"]["bytes"]
        with io.BytesIO(audio_bytes) as buf:
            audio, sr = sf.read(buf, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            try:
                import soxr  # noqa: PLC0415
                audio = soxr.resample(audio, sr, target_sr, quality="HQ").astype(np.float32)
            except ImportError:
                from math import gcd  # noqa: PLC0415
                from scipy.signal import resample_poly  # noqa: PLC0415
                g = gcd(target_sr, sr)
                audio = resample_poly(
                    audio, target_sr // g, sr // g,
                    window=("kaiser", 14.0), padtype="line",
                ).astype(np.float32)

        samples.append({
            "id": sample["id"],
            "audio": audio,
            "audio_duration": len(audio) / target_sr,
            "transcription": sample["transcription"],
            "category": sample["category"],
            "speaker_id": sample["speaker_id"],
        })
        if max_utts and len(samples) >= max_utts:
            break

    return samples


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    recognizer,
    samples: List[Dict],
    adlib_cases: Dict[str, Dict],
    offline: bool = True,
    sample_rate: int = 16000,
    chunk_size: float = 0.1,
    verbose: bool = False,
    term_acc: bool = False,
) -> Tuple[List[Dict], Dict]:
    results: List[Dict] = []

    total_edit_dist = 0
    total_ref_chars = 0
    total_correct_terms = 0
    total_terms = 0
    total_exact_correct = 0
    total_exact = 0
    total_flex_correct = 0
    total_flex = 0
    total_audio = 0.0
    total_proc = 0.0

    cer_pairs: List[Tuple[int, int]] = []
    term_pairs: List[Tuple[int, int]] = []

    for i, s in enumerate(samples, 1):
        audio = s["audio"]
        utt_id = s["id"]
        case = adlib_cases.get(utt_id, {})
        reference = case.get("reference", s["transcription"])
        terms = case.get("terms", [])

        t_start = time.monotonic()
        try:
            if offline:
                hypothesis = transcribe_offline(recognizer, audio, sample_rate)
            else:
                hypothesis = transcribe_online(recognizer, audio, sample_rate, chunk_size)
        except Exception as exc:
            print(f"  [{i:4d}] ERROR {utt_id}: {exc}", flush=True)
            hypothesis = ""
        proc_time = time.monotonic() - t_start

        duration = s["audio_duration"]
        rtf = proc_time / duration if duration > 0 else float("inf")

        cer, edit_dist, ref_len = _compute_cer(reference, hypothesis, terms)

        total_edit_dist += edit_dist
        total_ref_chars += ref_len
        total_audio += duration
        total_proc += proc_time
        cer_pairs.append((edit_dist, ref_len))

        result = {
            "id": utt_id,
            "category": s["category"],
            "speaker_id": s["speaker_id"],
            "reference": reference,
            "hypothesis": hypothesis,
            "audio_duration_s": duration,
            "processing_time_s": proc_time,
            "rtf": rtf,
            "latency_ms": proc_time * 1000.0,
            "char_edit_distance": edit_dist,
            "ref_chars": ref_len,
            "cer": cer,
        }

        if term_acc:
            term_result = _compute_term_accuracy(hypothesis, terms)
            total_correct_terms += term_result["correct"]
            total_terms += term_result["total"]
            total_exact_correct += term_result["exact_correct"]
            total_exact += term_result["exact_total"]
            total_flex_correct += term_result["flexible_correct"]
            total_flex += term_result["flexible_total"]
            term_pairs.append((term_result["correct"], term_result["total"]))
            result["term_accuracy"] = term_result["term_accuracy"]
            result["exact_term_accuracy"] = term_result["exact_term_accuracy"]
            result["flexible_term_accuracy"] = term_result["flexible_term_accuracy"]
            result["correct_terms"] = term_result["correct"]
            result["total_terms"] = term_result["total"]
            result["term_details"] = term_result["details"]

        results.append(result)

        marker = "✓" if cer == 0.0 else "✗"
        line = (
            f"  [{i:4d}/{len(samples)}] {marker}  RTF={rtf:.3f}  "
            f"CER={cer * 100:5.1f}%"
        )
        if term_acc:
            line += f"  TermAcc={result['term_accuracy'] * 100:5.1f}%"
        line += f"  Lat={proc_time * 1000:.0f}ms"
        line += f"  [{s['category']}  {s['speaker_id']}]" if verbose else f"  [{s['category']}]"
        print(line, flush=True)
        if verbose:
            print(f"    REF: {reference[:100]}")
            print(f"    HYP: {hypothesis[:100]}")

    micro_cer = min(total_edit_dist / total_ref_chars, 1.0) if total_ref_chars > 0 else 0.0
    mean_rtf = total_proc / total_audio if total_audio > 0 else float("inf")
    mean_lat = total_proc / len(results) * 1000.0 if results else float("inf")
    composite = (micro_cer + mean_rtf) / 2.0
    cer_ci = _bootstrap_ci(cer_pairs)

    agg: Dict = {
        "n_utterances": len(results),
        "cer": micro_cer,
        "cer_ci_95": list(cer_ci),
        "composite_score": composite,
        "mean_rtf": mean_rtf,
        "mean_latency_ms": mean_lat,
        "total_audio_s": total_audio,
        "total_proc_s": total_proc,
    }

    if term_acc:
        overall_term_acc = total_correct_terms / total_terms if total_terms > 0 else 1.0
        exact_term_acc = total_exact_correct / total_exact if total_exact > 0 else 1.0
        flex_term_acc = total_flex_correct / total_flex if total_flex > 0 else 1.0
        adlib_composite = 0.4 * (1 - micro_cer) + 0.6 * overall_term_acc
        term_ci = _bootstrap_ci(term_pairs)
        agg.update({
            "term_accuracy": overall_term_acc,
            "term_accuracy_ci_95": list(term_ci),
            "exact_term_accuracy": exact_term_acc,
            "flexible_term_accuracy": flex_term_acc,
            "adlib_composite_score": adlib_composite,
            "total_terms": total_terms,
        })

    print(f"\n  ── Aggregate ──")
    print(f"  Utterances      : {len(results)}")
    print(f"  CER             : {micro_cer * 100:.2f}%"
          f"  (95% CI: {cer_ci[0] * 100:.1f}% – {cer_ci[1] * 100:.1f}%)")
    if term_acc:
        print(f"  Term Accuracy   : {agg['term_accuracy'] * 100:.2f}%"
              f"  (95% CI: {agg['term_accuracy_ci_95'][0] * 100:.1f}% – {agg['term_accuracy_ci_95'][1] * 100:.1f}%)")
        print(f"    Exact         : {agg['exact_term_accuracy'] * 100:.2f}%")
        print(f"    Flexible      : {agg['flexible_term_accuracy'] * 100:.2f}%")
    print(f"  Mean RTF        : {mean_rtf:.4f}")
    print(f"  Mean Latency    : {mean_lat:.1f} ms")
    print(f"  Composite Score : {composite:.4f}  (CER + mean_RTF) / 2  (lower is better)")
    if term_acc:
        print(f"  Adlib Score     : {agg['adlib_composite_score']:.4f}"
              f"  0.4×(1−CER) + 0.6×TermAcc  (higher is better)")
    print(f"  Audio total     : {total_audio:.1f}s")
    print(f"  Proc total      : {total_proc:.1f}s")

    return results, agg


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(model_name: str, results: List[Dict], agg: Dict) -> None:
    has_terms = "term_accuracy" in agg
    print("\n" + "=" * 68)
    print("  SUMMARY — Japanese ASR Benchmark (adlib-devterm)")
    print("=" * 68)
    print(f"  Model            : {model_name}")
    print(f"  Dataset          : {DEFAULT_DATASET}")
    print(f"  Utterances       : {agg['n_utterances']}")
    print(f"  CER              : {agg['cer'] * 100:.2f}%"
          f"  (95% CI: {agg['cer_ci_95'][0] * 100:.1f}% – {agg['cer_ci_95'][1] * 100:.1f}%)")
    if has_terms:
        print(f"  Term Accuracy    : {agg['term_accuracy'] * 100:.2f}%"
              f"  (95% CI: {agg['term_accuracy_ci_95'][0] * 100:.1f}% – {agg['term_accuracy_ci_95'][1] * 100:.1f}%)")
        print(f"    Exact          : {agg['exact_term_accuracy'] * 100:.2f}%")
        print(f"    Flexible       : {agg['flexible_term_accuracy'] * 100:.2f}%")
    print(f"  Mean RTF         : {agg['mean_rtf']:.4f}")
    print(f"  Mean Latency(ms) : {agg['mean_latency_ms']:.1f}")
    print(f"  Composite Score  : {agg['composite_score']:.4f}  (CER + mean_RTF) / 2  (lower is better)")
    if has_terms:
        print(f"  Adlib Score      : {agg['adlib_composite_score']:.4f}  0.4×(1−CER) + 0.6×TermAcc  (higher is better)")
    print("=" * 68)
    breakdown_title = "CER & TermAcc by Category" if has_terms else "CER by Category"
    print_group_breakdown(results, "category", breakdown_title)
    breakdown_title = "CER & TermAcc by Speaker" if has_terms else "CER by Speaker"
    print_group_breakdown(results, "speaker_id", breakdown_title)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark Japanese ASR on holotherapper/adlib-devterm using the "
            "adlib evaluation protocol (CER + Term Accuracy + Composite Score)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET, metavar="REPO")
    p.add_argument("--split", default=DEFAULT_SPLIT, metavar="SPLIT")
    p.add_argument(
        "--category", default=None, metavar="CAT",
        choices=["backend", "cli", "concept", "frontend", "infra", "mixed"],
    )
    p.add_argument(
        "--speaker", default=None, metavar="SPK",
        choices=["spk-01", "spk-02", "spk-03"],
    )
    p.add_argument(
        "--adlib-cases", default=str(_ADLIB_CASES_CACHE), metavar="FILE",
        help="Path to adlib devterm test_cases.jsonl (auto-downloaded if absent)",
    )
    p.add_argument("--model-dir", default=None, metavar="PATH")
    p.add_argument("--model-type", default=DEFAULT_OFFLINE_MODEL_TYPE, metavar="TYPE")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--chunk-size", type=float, default=0.1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--language", default="ja", metavar="LANG")
    p.add_argument("--max-utts", type=int, default=None, metavar="N")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--output", metavar="FILE")
    p.add_argument(
        "--term-acc", action="store_true", default=False,
        help="Compute and report Term Accuracy and Adlib composite score (slower)",
    )
    return p


def _validate_args(args: argparse.Namespace) -> None:
    if args.sample_rate <= 0:
        print(f"Error: --sample-rate must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.threads <= 0:
        print(f"Error: --threads must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.max_utts is not None and args.max_utts <= 0:
        print(f"Error: --max-utts must be > 0", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args)

    from sherox.asr_engine import build_offline_recognizer, build_recognizer
    from sherox.config import Config

    model_dir = args.model_dir or (
        DEFAULT_OFFLINE_MODEL_DIR if args.offline else DEFAULT_ONLINE_MODEL_DIR
    )
    engine_model_type = _MODEL_TYPE_ALIASES.get(args.model_type, args.model_type)

    cfg = Config(
        model_dir=model_dir,
        model_type=engine_model_type,
        offline=args.offline,
        num_threads=args.threads,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        language=args.language,
    )

    mode_str = "offline" if args.offline else "online (streaming)"
    print(f"\nLoading {mode_str} model: {Path(model_dir).name}")
    print(f"  model_type : '{args.model_type or '(auto-detect)'}'")
    print(f"  language   : {cfg.language}")
    print(f"  threads    : {cfg.num_threads}")
    t0 = time.monotonic()
    recognizer = build_offline_recognizer(cfg) if args.offline else build_recognizer(cfg)
    print(f"  Loaded in {time.monotonic() - t0:.1f}s\n")

    adlib_cases = load_adlib_cases(Path(args.adlib_cases))
    print(f"  {len(adlib_cases)} adlib test cases loaded")

    samples = load_hf_samples(
        dataset_name=args.dataset,
        split=args.split,
        category=args.category,
        speaker=args.speaker,
        max_utts=args.max_utts,
        target_sr=args.sample_rate,
    )
    print(f"Loaded {len(samples)} utterances\n")

    results, agg = run_benchmark(
        recognizer, samples, adlib_cases,
        offline=args.offline,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        verbose=args.verbose,
        term_acc=args.term_acc,
    )

    print_summary(Path(model_dir).name, results, agg)

    if args.output:
        output_data = {
            "dataset": args.dataset,
            "split": args.split,
            "model_dir": model_dir,
            "model_type": args.model_type,
            "offline": args.offline,
            "language": args.language,
            "threads": cfg.num_threads,
            "filters": {"category": args.category, "speaker": args.speaker},
            "aggregate": agg,
            "utterances": [
                {k: v for k, v in r.items() if k != "term_details"}
                for r in results
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
