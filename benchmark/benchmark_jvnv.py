#!/usr/bin/env python3
"""
Japanese ASR benchmark on the JVNV corpus (local dataset).

JVNV is a 4-speaker, 6-emotion emotional speech corpus with nonverbal
vocalizations (laughter, sobbing, etc.) embedded in each utterance.
  - 1,615 utterances, ~3.94 hours
  - Speakers: F1, F2, M1, M2
  - Emotions: anger, disgust, fear, happy, sad, surprise
  - Sessions: regular (designated NV phrase), free (speaker-chosen NV phrase)
  - Audio: 48 kHz mono WAV → resampled to 16 kHz for ASR

CER is computed at content level (same pipeline as benchmark_ja.py):
  NFKC + lowercase + strip JP punctuation + strip whitespace

Composite Score = (CER + mean_RTF) / 2  (lower is better)

Dataset: https://ss-takashi.sakura.ne.jp/corpus/jvnv/
Reference implementation: https://github.com/ouktlab/asr-ja_evalkit

Usage examples:
    # Full benchmark with default model
    python benchmark_jvnv.py --jvnv-dir /data/jvnv_v1 --offline

    # Smoke test: first 20 utterances, verbose
    python benchmark_jvnv.py --jvnv-dir /data/jvnv_v1 --offline --max-utts 20 --verbose

    # Filter by emotion or speaker
    python benchmark_jvnv.py --jvnv-dir /data/jvnv_v1 --offline --emotion happy
    python benchmark_jvnv.py --jvnv-dir /data/jvnv_v1 --offline --speaker F1

    # Save full results to JSON
    python benchmark_jvnv.py --jvnv-dir /data/jvnv_v1 --offline --output results_jvnv.json

    # Different model
    python benchmark_jvnv.py --jvnv-dir /data/jvnv_v1 --offline --model-type reazonspeech-ja
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
    print_group_breakdown,
    transcribe_offline,
    transcribe_online,
)

DEFAULT_JVNV_DIR = "/data/jvnv_v1"
DEFAULT_OFFLINE_MODEL_DIR = str(_PROJECT_DIR / "models" / "parakeet-ctc-ja-int8")
DEFAULT_OFFLINE_MODEL_TYPE = "parakeet-ctc-ja"
DEFAULT_ONLINE_MODEL_DIR = str(_PROJECT_DIR / "models" / "zipformer-ja")

SPEAKERS = ["F1", "F2", "M1", "M2"]
EMOTIONS = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
SESSIONS = ["regular", "free"]

_MODEL_TYPE_ALIASES = {
    "parakeet-ctc-ja": "nemo_ctc",
    "reazonspeech-ja": "",
    "reazonspeech-ja-en": "",
    "reazonspeech-ja-en-mls-5k": "",
}


# ---------------------------------------------------------------------------
# JVNV data loading
# ---------------------------------------------------------------------------

def load_transcriptions(jvnv_dir: Path) -> Dict[str, str]:
    """Parse transcription.csv → {emotion_session_id: full_transcription}.

    CSV format (pipe-delimited, no header):
        {emotion}_{session}_{id}|{nv_phrase}|{full_transcription}
    """
    trans: Dict[str, str] = {}
    csv_path = jvnv_dir / "transcription.csv"
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                trans[parts[0]] = parts[2]
    return trans


def load_jvnv_samples(
    jvnv_dir: Path,
    speakers: Optional[List[str]] = None,
    emotions: Optional[List[str]] = None,
    sessions: Optional[List[str]] = None,
    max_utts: Optional[int] = None,
    target_sr: int = 16000,
) -> List[Dict]:
    """Load JVNV WAV files and their transcriptions from the local dataset.

    Audio is 48 kHz mono; resampled to target_sr (16 kHz) for ASR.
    Returns list of dicts with keys: id, audio, audio_duration, reference,
    speaker, emotion, session.
    """
    try:
        import soundfile as sf  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("pip install soundfile") from exc

    trans = load_transcriptions(jvnv_dir)
    missing: List[str] = []

    speakers_to_use = speakers or SPEAKERS
    emotions_to_use = emotions or EMOTIONS
    sessions_to_use = sessions or SESSIONS

    samples: List[Dict] = []
    for speaker in speakers_to_use:
        for emotion in emotions_to_use:
            for session in sessions_to_use:
                wav_dir = jvnv_dir / speaker / emotion / session
                if not wav_dir.exists():
                    continue
                for wav_path in sorted(wav_dir.glob("*.wav")):
                    stem = wav_path.stem  # e.g. F1_anger_regular_01
                    # transcription key: emotion_session_id (drop speaker prefix)
                    parts = stem.split("_")
                    trans_key = "_".join(parts[1:])  # anger_regular_01
                    reference = trans.get(trans_key, "")
                    if not reference:
                        missing.append(stem)

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
                        "id": stem,
                        "audio": audio,
                        "audio_duration": len(audio) / target_sr,
                        "reference": reference,
                        "speaker": speaker,
                        "emotion": emotion,
                        "session": session,
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
    offline: bool = True,
    sample_rate: int = 16000,
    chunk_size: float = 0.1,
    verbose: bool = False,
) -> Tuple[List[Dict], Dict]:
    results: List[Dict] = []

    total_edit_dist = 0
    total_ref_chars = 0
    total_audio = 0.0
    total_proc = 0.0
    cer_pairs: List[Tuple[int, int]] = []

    for i, s in enumerate(samples, 1):
        audio = s["audio"]
        utt_id = s["id"]
        reference = s["reference"]

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

        # No term annotations in JVNV — pass empty terms list
        cer, edit_dist, ref_len = _compute_cer(reference, hypothesis, [])

        total_edit_dist += edit_dist
        total_ref_chars += ref_len
        total_audio += duration
        total_proc += proc_time
        cer_pairs.append((edit_dist, ref_len))

        result = {
            "id": utt_id,
            "speaker": s["speaker"],
            "emotion": s["emotion"],
            "session": s["session"],
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
        results.append(result)

        marker = "✓" if cer == 0.0 else "✗"
        line = (
            f"  [{i:4d}/{len(samples)}] {marker}  RTF={rtf:.3f}  "
            f"CER={cer * 100:5.1f}%  Lat={proc_time * 1000:.0f}ms"
            f"  [{s['emotion']}  {s['speaker']}  {s['session']}]"
        )
        if verbose:
            line = line.rstrip("]") + "]"
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

    print(f"\n  ── Aggregate ──")
    print(f"  Utterances      : {len(results)}")
    print(f"  CER             : {micro_cer * 100:.2f}%"
          f"  (95% CI: {cer_ci[0] * 100:.1f}% – {cer_ci[1] * 100:.1f}%)")
    print(f"  Mean RTF        : {mean_rtf:.4f}")
    print(f"  Mean Latency    : {mean_lat:.1f} ms")
    print(f"  Composite Score : {composite:.4f}  (CER + mean_RTF) / 2  (lower is better)")
    print(f"  Audio total     : {total_audio:.1f}s")
    print(f"  Proc total      : {total_proc:.1f}s")

    return results, agg


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(model_name: str, jvnv_dir: str, results: List[Dict], agg: Dict) -> None:
    print("\n" + "=" * 68)
    print("  SUMMARY — Japanese ASR Benchmark (JVNV)")
    print("=" * 68)
    print(f"  Model            : {model_name}")
    print(f"  Dataset          : JVNV v1  ({jvnv_dir})")
    print(f"  Utterances       : {agg['n_utterances']}")
    print(f"  CER              : {agg['cer'] * 100:.2f}%"
          f"  (95% CI: {agg['cer_ci_95'][0] * 100:.1f}% – {agg['cer_ci_95'][1] * 100:.1f}%)")
    print(f"  Mean RTF         : {agg['mean_rtf']:.4f}")
    print(f"  Mean Latency(ms) : {agg['mean_latency_ms']:.1f}")
    print(f"  Composite Score  : {agg['composite_score']:.4f}  (CER + mean_RTF) / 2  (lower is better)")
    print("=" * 68)
    print_group_breakdown(results, "speaker", "CER by Speaker")
    print_group_breakdown(results, "emotion", "CER by Emotion")
    print_group_breakdown(results, "session", "CER by Session")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark Japanese ASR on the JVNV emotional speech corpus. "
            "Evaluates CER across 4 speakers, 6 emotions, and 2 sessions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--jvnv-dir", default=DEFAULT_JVNV_DIR, metavar="PATH",
        help="Root directory of the JVNV corpus (contains transcription.csv)",
    )
    p.add_argument(
        "--speaker", default=None, metavar="SPK", choices=SPEAKERS,
        help="Filter to a single speaker",
    )
    p.add_argument(
        "--emotion", default=None, metavar="EMO", choices=EMOTIONS,
        help="Filter to a single emotion",
    )
    p.add_argument(
        "--session", default=None, metavar="SES", choices=SESSIONS,
        help="Filter to a single session (regular or free)",
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
    jvnv_dir = Path(args.jvnv_dir)
    if not jvnv_dir.exists():
        print(f"Error: JVNV directory not found: {jvnv_dir}", file=sys.stderr)
        sys.exit(1)
    if not (jvnv_dir / "transcription.csv").exists():
        print(f"Error: transcription.csv not found in {jvnv_dir}", file=sys.stderr)
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

    jvnv_dir = Path(args.jvnv_dir)
    print(f"Loading JVNV from {jvnv_dir} …")
    speakers = [args.speaker] if args.speaker else None
    emotions = [args.emotion] if args.emotion else None
    sessions = [args.session] if args.session else None
    samples = load_jvnv_samples(
        jvnv_dir=jvnv_dir,
        speakers=speakers,
        emotions=emotions,
        sessions=sessions,
        max_utts=args.max_utts,
        target_sr=args.sample_rate,
    )
    spk_set = sorted({s["speaker"] for s in samples})
    emo_set = sorted({s["emotion"] for s in samples})
    ses_set = sorted({s["session"] for s in samples})
    print(
        f"  {len(samples)} utterances  "
        f"({len(spk_set)} speakers, {len(emo_set)} emotions, {len(ses_set)} sessions)\n"
    )

    results, agg = run_benchmark(
        recognizer, samples,
        offline=args.offline,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        verbose=args.verbose,
    )

    print_summary(Path(model_dir).name, str(jvnv_dir), results, agg)

    if args.output:
        output_data = {
            "dataset": "JVNV",
            "jvnv_dir": str(jvnv_dir),
            "model_dir": model_dir,
            "model_type": args.model_type,
            "offline": args.offline,
            "language": args.language,
            "threads": cfg.num_threads,
            "filters": {
                "speaker": args.speaker,
                "emotion": args.emotion,
                "session": args.session,
            },
            "aggregate": agg,
            "utterances": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
