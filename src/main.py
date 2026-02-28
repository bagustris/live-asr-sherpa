"""Streaming ASR — entry point.

Usage:
    python3 main.py --mic
    python3 main.py --wav path/to/audio.wav [--model-dir model] [--threads 4]
"""

import argparse
import sys
import tarfile
import urllib.request
import wave
from pathlib import Path

from audio import mic_stream, read_wav
from asr_engine import build_recognizer
from config import Config
from streaming import run_streaming


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming ASR with Sherpa-ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mic", action="store_true", help="Stream from microphone")
    mode.add_argument("--wav", metavar="PATH", help="Transcribe a WAV file")

    parser.add_argument("--model-dir", default="model", help="Sherpa-ONNX model directory")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate (Hz)")
    parser.add_argument(
        "--chunk-size", type=float, default=0.16, help="Chunk size in seconds (0.1–0.2 recommended)"
    )
    parser.add_argument("--threads", type=int, default=4, help="CPU thread count for ONNX runtime")
    return parser.parse_args()


_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
)
_MODEL_ARCHIVE = "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
_MODEL_EXTRACTED = "sherpa-onnx-streaming-zipformer-en-2023-06-26"


def _download_model(model_dir: str) -> None:
    """Download and extract the default Zipformer model, then rename to model_dir."""
    model_dir = Path(model_dir)
    archive = model_dir.parent / _MODEL_ARCHIVE
    print(f"[info] Model not found. Downloading from:\n  {_MODEL_URL}")
    print("[info] This may take a few minutes…")

    def _progress(block: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            sys.stdout.write(f"\r  {pct}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(_MODEL_URL, archive, reporthook=_progress)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"\n[error] Download failed: {exc}")

    print("\n[info] Extracting…")
    try:
        with tarfile.open(archive, "r:bz2") as tf:
            tf.extractall(model_dir.parent, filter="data")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"[error] Extraction failed: {exc}")

    extracted = model_dir.parent / _MODEL_EXTRACTED
    if not extracted.is_dir():
        sys.exit(f"[error] Expected extracted directory '{_MODEL_EXTRACTED}' not found.")

    extracted.rename(model_dir)
    archive.unlink(missing_ok=True)
    print(f"[info] Model saved to '{model_dir}'.\n")


def _validate_model(model_dir: str) -> None:
    if not Path(model_dir).is_dir():
        _download_model(model_dir)


def _validate_wav(path: str, sample_rate: int) -> None:
    p = Path(path)
    if not p.exists():
        sys.exit(f"[error] WAV file not found: {path}")
    try:
        with wave.open(path, "rb") as wf:
            if wf.getnchannels() != 1:
                sys.exit(
                    f"[error] WAV must be mono (1 channel), got {wf.getnchannels()}.\n"
                    f"  Convert: ffmpeg -i {path} -ar {sample_rate} -ac 1 out.wav"
                )
            if wf.getframerate() != sample_rate:
                sys.exit(
                    f"[error] WAV sample rate must be {sample_rate} Hz, got {wf.getframerate()} Hz.\n"
                    f"  Convert: ffmpeg -i {path} -ar {sample_rate} -ac 1 out.wav"
                )
    except wave.Error as exc:
        sys.exit(f"[error] Cannot read WAV file: {exc}")


def _validate_mic() -> None:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        inputs = [d for d in devices if d["max_input_channels"] > 0]
        if not inputs:
            sys.exit("[error] No input audio device found.")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"[error] Microphone check failed: {exc}")


def main() -> None:
    args = parse_args()

    # Resolve model_dir relative to the script's directory so the model is
    # found regardless of the working directory the user invokes from.
    script_dir = Path(__file__).resolve().parent
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = script_dir / model_dir

    cfg = Config(
        model_dir=str(model_dir),
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        num_threads=args.threads,
    )

    _validate_model(cfg.model_dir)

    if args.wav:
        _validate_wav(args.wav, cfg.sample_rate)
    else:
        _validate_mic()

    print(f"[info] Loading model from '{cfg.model_dir}' ({cfg.num_threads} threads)…")
    recognizer = build_recognizer(cfg)
    print("[info] Model ready.\n")

    if args.wav:
        print(f"[info] Transcribing: {args.wav}\n")
        audio = read_wav(args.wav, target_sr=cfg.sample_rate, chunk_size=cfg.chunk_size)
    else:
        print("[info] Listening on microphone — press Ctrl+C to stop.\n")
        audio = mic_stream(sample_rate=cfg.sample_rate, chunk_size=cfg.chunk_size)

    run_streaming(recognizer, audio, sample_rate=cfg.sample_rate)


if __name__ == "__main__":
    main()
