#!/usr/bin/env python3
"""
Japanese ASR benchmark on the JVS corpus (local dataset).

JVS (Japanese Versatile Speech) is a large-scale text-reading corpus with
100 speakers (50 male, 50 female) × 100 utterances (JSUT-style sentences),
plus whispered speech (100 utts) and falsetto speech (10 utts) per speaker.
  - Total: ~30 hours of audio
  - Sample rate: 24 kHz mono WAV → resampled to 16 kHz for ASR
  - Corpus page: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus

Directory structure assumed:
    <jvs-dir>/
      jvs001/
        parallel100/
          wav24kHz16bit/
            VOICEACTRESS100_001.wav
            ...
          transcripts_utf8.txt   (space-delimited: id  transcript)
        nonpara30/
          wav24kHz16bit/
            ...
          transcripts_utf8.txt
        whisper10/
          wav24kHz16bit/
            ...
          transcripts_utf8.txt
        falsetto10/
          ...
      jvs002/
        ...

Primary metric: KER (Kana Error Rate) — phonetically robust, handles kanji vs
kana equivalence.  CER is also reported for comparison.

Composite Score = (KER + mean_RTF) / 2  (lower is better)

Usage examples:
    # Full benchmark with default model
    python benchmark_jvs.py --jvs-dir /data/jvs_ver1 --offline

    # Smoke test: first 20 utterances, verbose
    python benchmark_jvs.py --jvs-dir /data/jvs_ver1 --offline --max-utts 20 --verbose

    # Filter by speaker or subset
    python benchmark_jvs.py --jvs-dir /data/jvs_ver1 --offline --speaker jvs001
    python benchmark_jvs.py --jvs-dir /data/jvs_ver1 --offline --subset parallel100

    # Only female or male speakers
    python benchmark_jvs.py --jvs-dir /data/jvs_ver1 --offline --gender female

    # Save full results to JSON
    python benchmark_jvs.py --jvs-dir /data/jvs_ver1 --offline --output results_jvs.json

    # Different model
    python benchmark_jvs.py --jvs-dir /data/jvs_ver1 --offline --model-type reazonspeech-ja
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    _bootstrap_ci,
    _compute_cer,
    _compute_ker,
    print_group_breakdown,
    transcribe_offline,
    transcribe_online,
)

DEFAULT_JVS_DIR = "/data/jvs_ver1"
DEFAULT_OFFLINE_MODEL_DIR = str(_PROJECT_DIR / "models" / "parakeet-ctc-ja-int8")
DEFAULT_OFFLINE_MODEL_TYPE = "parakeet-ctc-ja"
DEFAULT_ONLINE_MODEL_DIR = str(_PROJECT_DIR / "models" / "zipformer-ja")

# Subsets shipped with JVS
JVS_SUBSETS = ["parallel100", "nonpara30", "whisper10", "falsetto10"]

# Speakers jvs001 … jvs100; jvs001–050 = female, jvs051–100 = male
# (see https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
_FEMALE_SPEAKERS = {f"jvs{i:03d}" for i in range(1, 51)}
_MALE_SPEAKERS = {f"jvs{i:03d}" for i in range(51, 101)}

_MODEL_TYPE_ALIASES = {
    "parakeet-ctc-ja": "nemo_ctc",
    "reazonspeech-ja": "",
    "reazonspeech-ja-en": "",
    "reazonspeech-ja-en-mls-5k": "",
}


# ---------------------------------------------------------------------------
# JVS data loading
# ---------------------------------------------------------------------------

def _load_transcripts(transcript_path: Path) -> Dict[str, str]:
    """Parse JVS transcripts_utf8.txt → {utterance_id: transcript}.

    Supports both colon-delimited (``VOICEACTRESS100_001:text``) and
    whitespace-delimited (``VOICEACTRESS100_001 text goes here``) formats.
    """
    trans: Dict[str, str] = {}
    if not transcript_path.exists():
        return trans
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try colon-delimited first (official JVS format)
            if ":" in line:
                utt_id, _, text = line.partition(":")
                utt_id = utt_id.strip()
                text = text.strip()
                if utt_id and text:
                    trans[utt_id] = text
                    continue
            # Fallback to whitespace-delimited
            parts = line.split(None, 1)
            if len(parts) == 2:
                utt_id, text = parts
                trans[utt_id] = text
    return trans


def load_jvs_samples(
    jvs_dir: Path,
    speakers: Optional[List[str]] = None,
    subsets: Optional[List[str]] = None,
    gender: Optional[str] = None,   # "male" | "female" | None
    max_utts: Optional[int] = None,
    target_sr: int = 16000,
) -> List[Dict]:
    """Load JVS WAV files and transcriptions from the local corpus.

    Returns a list of dicts with keys:
      id, audio, audio_duration, reference, speaker, subset, gender.
    """
    try:
        import soundfile as sf  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("pip install soundfile") from exc

    jvs_dir = Path(jvs_dir)
    subsets_to_use = subsets or ["parallel100"]  # default to main read-speech subset

    # Determine speaker list
    if speakers:
        spk_dirs = [jvs_dir / spk for spk in speakers]
    else:
        spk_dirs = sorted(jvs_dir.glob("jvs*"))

    # Apply gender filter
    if gender == "female":
        spk_dirs = [d for d in spk_dirs if d.name in _FEMALE_SPEAKERS]
    elif gender == "male":
        spk_dirs = [d for d in spk_dirs if d.name in _MALE_SPEAKERS]

    samples: List[Dict] = []
    missing: List[str] = []

    for spk_dir in spk_dirs:
        if not spk_dir.is_dir():
            continue
        spk_name = spk_dir.name
        spk_gender = "female" if spk_name in _FEMALE_SPEAKERS else "male"

        for subset in subsets_to_use:
            subset_dir = spk_dir / subset
            wav_dir = subset_dir / "wav24kHz16bit"
            transcript_path = subset_dir / "transcripts_utf8.txt"
            if not wav_dir.exists():
                continue
            trans = _load_transcripts(transcript_path)

            for wav_path in sorted(wav_dir.glob("*.wav")):
                utt_id = wav_path.stem
                reference = trans.get(utt_id, "")
                if not reference:
                    missing.append(f"{spk_name}/{subset}/{utt_id}")

                audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
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
                    "id": f"{spk_name}_{subset}_{utt_id}",
                    "audio": audio,
                    "audio_duration": len(audio) / target_sr,
                    "reference": reference,
                    "speaker": spk_name,
                    "subset": subset,
                    "gender": spk_gender,
                })

                if max_utts and len(samples) >= max_utts:
                    if missing:
                        print(f"  Warning: {len(missing)} utterances had no transcription", flush=True)
                    return samples

    if missing:
        print(f"  Warning: {len(missing)} utterances had no transcription", flush=True)
    return samples


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    recognizer,
    samples: List[Dict],
    model_type: str,
    offline: bool = True,
    sample_rate: int = 16000,
    chunk_size: float = 0.1,
    verbose: bool = False,
) -> Tuple[List[Dict], Dict]:
    results: List[Dict] = []

    total_edit_dist = 0
    total_ref_chars = 0
    total_ker_edit = 0
    total_ker_ref = 0
    total_audio = 0.0
    total_proc = 0.0
    cer_pairs: List[Tuple[int, int]] = []
    ker_pairs: List[Tuple[int, int]] = []

    for i, s in enumerate(samples, 1):
        audio = s["audio"]
        utt_id = s["id"]
        reference = s["reference"]

        t_start = time.monotonic()
        try:
            if offline:
                hypothesis = transcribe_offline(
                    recognizer,
                    audio,
                    sample_rate,
                    model_type=model_type,
                )
            else:
                hypothesis = transcribe_online(recognizer, audio, sample_rate, chunk_size)
        except Exception as exc:
            print(f"  [{i:4d}] ERROR {utt_id}: {exc}", flush=True)
            hypothesis = ""
        proc_time = time.monotonic() - t_start

        duration = s["audio_duration"]
        rtf = proc_time / duration if duration > 0 else float("inf")

        cer, edit_dist, ref_len = _compute_cer(reference, hypothesis, [])
        ker, ker_edit_dist, ker_ref_len = _compute_ker(reference, hypothesis)

        total_edit_dist += edit_dist
        total_ref_chars += ref_len
        total_ker_edit += ker_edit_dist
        total_ker_ref += ker_ref_len
        total_audio += duration
        total_proc += proc_time
        cer_pairs.append((edit_dist, ref_len))
        ker_pairs.append((ker_edit_dist, ker_ref_len))

        result = {
            "id": utt_id,
            "speaker": s["speaker"],
            "subset": s["subset"],
            "gender": s["gender"],
            "reference": reference,
            "hypothesis": hypothesis,
            "audio_duration_s": duration,
            "processing_time_s": proc_time,
            "rtf": rtf,
            "latency_ms": proc_time * 1000.0,
            "char_edit_distance": edit_dist,
            "ref_chars": ref_len,
            "cer": cer,
            "ker_edit_distance": ker_edit_dist,
            "ker_ref_chars": ker_ref_len,
            "ker": ker,
        }
        results.append(result)

        marker = "✓" if ker == 0.0 else "✗"
        line = (
            f"  [{i:4d}/{len(samples)}] {marker}  RTF={rtf:.3f}  "
            f"CER={cer * 100:5.1f}%  KER={ker * 100:5.1f}%  Lat={proc_time * 1000:.0f}ms"
        )
        line += f"  [{s['speaker']}  {s['subset']}]" if verbose else f"  [{s['speaker']}]"
        print(line, flush=True)
        if verbose:
            print(f"    REF: {reference[:100]}")
            print(f"    HYP: {hypothesis[:100]}")

    micro_cer = min(total_edit_dist / total_ref_chars, 1.0) if total_ref_chars > 0 else 0.0
    micro_ker = min(total_ker_edit / total_ker_ref, 1.0) if total_ker_ref > 0 else 0.0
    mean_rtf = total_proc / total_audio if total_audio > 0 else float("inf")
    mean_lat = total_proc / len(results) * 1000.0 if results else float("inf")
    composite = (micro_ker + mean_rtf) / 2.0
    cer_ci = _bootstrap_ci(cer_pairs)
    ker_ci = _bootstrap_ci(ker_pairs)

    agg: Dict = {
        "n_utterances": len(results),
        "cer": micro_cer,
        "cer_ci_95": list(cer_ci),
        "ker": micro_ker,
        "ker_ci_95": list(ker_ci),
        "composite_score": composite,
        "mean_rtf": mean_rtf,
        "mean_latency_ms": mean_lat,
        "total_audio_s": total_audio,
        "total_proc_s": total_proc,
    }

    print(f"\n  ── Aggregate ──")
    print(f"  Utterances      : {len(results)}")
    print(f"  CER             : {micro_cer * 100:.2f}%"
          f"  (95% CI: {cer_ci[0] * 100:.1f}% – {cer_ci[1] * 100:.1f}%)")
    print(f"  KER             : {micro_ker * 100:.2f}%"
          f"  (95% CI: {ker_ci[0] * 100:.1f}% – {ker_ci[1] * 100:.1f}%)")
    print(f"  Mean RTF        : {mean_rtf:.4f}")
    print(f"  Mean Latency    : {mean_lat:.1f} ms")
    print(f"  Composite Score : {composite:.4f}  (KER + mean_RTF) / 2  (lower is better)")
    print(f"  Audio total     : {total_audio:.1f}s")
    print(f"  Proc total      : {total_proc:.1f}s")

    return results, agg


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(model_name: str, results: List[Dict], agg: Dict) -> None:
    print("\n" + "=" * 68)
    print("  SUMMARY — Japanese ASR Benchmark (JVS)")
    print("=" * 68)
    print(f"  Model            : {model_name}")
    print(f"  Utterances       : {agg['n_utterances']}")
    print(f"  CER              : {agg['cer'] * 100:.2f}%"
          f"  (95% CI: {agg['cer_ci_95'][0] * 100:.1f}% – {agg['cer_ci_95'][1] * 100:.1f}%)")
    print(f"  KER              : {agg['ker'] * 100:.2f}%"
          f"  (95% CI: {agg['ker_ci_95'][0] * 100:.1f}% – {agg['ker_ci_95'][1] * 100:.1f}%)")
    print(f"  Mean RTF         : {agg['mean_rtf']:.4f}")
    print(f"  Mean Latency(ms) : {agg['mean_latency_ms']:.1f}")
    print(f"  Composite Score  : {agg['composite_score']:.4f}  (KER + mean_RTF) / 2  (lower is better)")
    print("=" * 68)
    print_group_breakdown(results, "gender", "CER/KER by Gender")
    print_group_breakdown(results, "subset", "CER/KER by Subset")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark Japanese ASR on the JVS corpus (local dataset). "
            "Reports CER and KER (Kana Error Rate)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--jvs-dir", default=DEFAULT_JVS_DIR, metavar="PATH",
                   help="Path to the JVS corpus root directory (contains jvs001…jvs100).")
    p.add_argument(
        "--subset", default=None, nargs="+", metavar="SUBSET",
        choices=JVS_SUBSETS,
        help=f"Which JVS subsets to evaluate. Default: parallel100. Choices: {JVS_SUBSETS}",
    )
    p.add_argument("--speaker", default=None, nargs="+", metavar="SPK",
                   help="Restrict evaluation to specific speaker IDs (e.g. jvs001 jvs002).")
    p.add_argument("--gender", default=None, choices=["male", "female"],
                   help="Restrict to male (jvs051-100) or female (jvs001-050) speakers.")
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
    return p


def _validate_args(args: argparse.Namespace) -> None:
    if args.sample_rate <= 0:
        print("Error: --sample-rate must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.threads <= 0:
        print("Error: --threads must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.max_utts is not None and args.max_utts <= 0:
        print("Error: --max-utts must be > 0", file=sys.stderr)
        sys.exit(1)
    jvs_dir = Path(args.jvs_dir)
    if not jvs_dir.exists():
        print(f"Error: JVS directory not found: {jvs_dir}", file=sys.stderr)
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

    jvs_dir = Path(args.jvs_dir)
    subsets = args.subset or ["parallel100"]
    samples = load_jvs_samples(
        jvs_dir=jvs_dir,
        speakers=args.speaker,
        subsets=subsets,
        gender=args.gender,
        max_utts=args.max_utts,
        target_sr=args.sample_rate,
    )
    print(f"Loaded {len(samples)} utterances from {jvs_dir.name}\n")

    results, agg = run_benchmark(
        recognizer, samples,
        model_type=args.model_type,
        offline=args.offline,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        verbose=args.verbose,
    )

    print_summary(Path(model_dir).name, results, agg)

    if args.output:
        output_data = {
            "dataset": "JVS",
            "jvs_dir": str(jvs_dir),
            "subsets": subsets,
            "model_dir": model_dir,
            "model_type": args.model_type,
            "offline": args.offline,
            "language": args.language,
            "threads": cfg.num_threads,
            "aggregate": agg,
            "utterances": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
