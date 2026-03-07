"""Streaming ASR — entry point.

Usage:
    python3 main.py --mic
    python3 main.py --wav path/to/audio.wav [--model-dir model] [--threads 4]

    # Parakeet TDT (offline NeMo transducer):
    python3 main.py --mic --model-dir /path/to/parakeet --model-type nemo_transducer --vad-model silero_vad.onnx
    python3 main.py --wav audio.wav --model-dir /path/to/parakeet --model-type nemo_transducer --vad-model silero_vad.onnx
"""

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf

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
    parser.add_argument(
        "--model-type",
        default="online",
        choices=["online", "nemo_transducer"],
        help=(
            "Model architecture: 'online' for streaming transducers (e.g. Zipformer), "
            "'nemo_transducer' for offline NeMo transducers (e.g. Parakeet TDT)"
        ),
    )
    parser.add_argument(
        "--vad-model",
        default="",
        metavar="PATH",
        help="Path to silero_vad.onnx; required when --model-type=nemo_transducer",
    )
    return parser.parse_args()


_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
)
_MODEL_ARCHIVE = "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
_MODEL_EXTRACTED = "sherpa-onnx-streaming-zipformer-en-2023-06-26"

_PARAKEET_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16.tar.bz2"
)
_PARAKEET_MODEL_ARCHIVE = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16.tar.bz2"
_PARAKEET_MODEL_EXTRACTED = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16"

_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/silero_vad.onnx"
)


def _download_file(url: str, dest: Path) -> None:
    print(f"[info] Downloading from:\n  {url}")
    print("[info] This may take a few minutes…")

    def _progress(block: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            sys.stdout.write(f"\r  {pct}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"\n[error] Download failed: {exc}")
    print()


def _download_model(model_dir: str, model_type: str) -> None:
    """Download and extract the default model for the given model_type."""
    model_dir = Path(model_dir)

    if model_type == "nemo_transducer":
        url = _PARAKEET_MODEL_URL
        archive_name = _PARAKEET_MODEL_ARCHIVE
        extracted_name = _PARAKEET_MODEL_EXTRACTED
    else:
        url = _MODEL_URL
        archive_name = _MODEL_ARCHIVE
        extracted_name = _MODEL_EXTRACTED

    archive = model_dir.parent / archive_name
    print(f"[info] Model not found.")
    _download_file(url, archive)

    print("[info] Extracting…")
    try:
        with tarfile.open(archive, "r:bz2") as tf:
            tf.extractall(model_dir.parent, filter="data")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"[error] Extraction failed: {exc}")

    extracted = model_dir.parent / extracted_name
    if not extracted.is_dir():
        sys.exit(f"[error] Expected extracted directory '{extracted_name}' not found.")

    extracted.rename(model_dir)
    archive.unlink(missing_ok=True)
    print(f"[info] Model saved to '{model_dir}'.\n")


def _validate_model(model_dir: str, model_type: str) -> None:
    if not Path(model_dir).is_dir():
        _download_model(model_dir, model_type)


def _validate_vad(vad_model: str, model_type: str, script_dir: Path) -> str:
    if model_type != "nemo_transducer":
        return vad_model
    if not vad_model:
        vad_path = script_dir / "silero_vad.onnx"
        if not vad_path.exists():
            print("[info] VAD model not found, downloading silero_vad.onnx…")
            _download_file(_VAD_URL, vad_path)
        return str(vad_path)
    if not Path(vad_model).exists():
        sys.exit(f"[error] VAD model not found: {vad_model}")
    return vad_model


def _validate_wav(path: str, sample_rate: int) -> None:
    p = Path(path)
    if not p.exists():
        sys.exit(f"[error] Audio file not found: {path}")
    try:
        with sf.SoundFile(path) as f:
            if f.channels != 1:
                sys.exit(
                    f"[error] Audio must be mono (1 channel), got {f.channels}.\n"
                    f"  Convert: ffmpeg -i {path} -ar {sample_rate} -ac 1 out.wav"
                )
            if f.samplerate != sample_rate:
                sys.exit(
                    f"[error] Audio sample rate must be {sample_rate} Hz, got {f.samplerate} Hz.\n"
                    f"  Convert: ffmpeg -i {path} -ar {sample_rate} -ac 1 out.wav"
                )
    except Exception as exc:
        sys.exit(f"[error] Cannot read audio file: {exc}")


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
    # Use a type-specific default dir when the user didn't pass --model-dir explicitly
    raw_model_dir = args.model_dir
    if raw_model_dir == "model" and args.model_type == "nemo_transducer":
        raw_model_dir = "model-parakeet"
    model_dir = Path(raw_model_dir)
    if not model_dir.is_absolute():
        model_dir = script_dir / model_dir

    cfg = Config(
        model_dir=str(model_dir),
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        num_threads=args.threads,
        model_type=args.model_type,
        vad_model=args.vad_model,
    )

    _validate_model(cfg.model_dir, cfg.model_type)
    cfg.vad_model = _validate_vad(cfg.vad_model, cfg.model_type, script_dir)

    if args.wav:
        _validate_wav(args.wav, cfg.sample_rate)
    else:
        _validate_mic()

    print(f"[info] Loading model '{cfg.model_type}' ({cfg.num_threads} threads)…")

    if args.wav:
        audio = read_wav(args.wav, target_sr=cfg.sample_rate, chunk_size=cfg.chunk_size)
    else:
        audio = mic_stream(sample_rate=cfg.sample_rate, chunk_size=cfg.chunk_size)

    if cfg.model_type == "nemo_transducer":
        from asr_engine import build_offline_recognizer, build_vad
        from streaming import run_offline_vad_streaming

        recognizer = build_offline_recognizer(cfg)
        vad = build_vad(cfg)
        print("[info] Model ready.\n")

        if args.wav:
            print(f"[info] Transcribing: {args.wav}\n")
        else:
            print("[info] Listening on microphone — press Ctrl+C to stop.\n")

        run_offline_vad_streaming(
            recognizer=recognizer,
            vad=vad,
            audio_gen=audio,
            sample_rate=cfg.sample_rate,
        )
    else:
        recognizer = build_recognizer(cfg)
        print("[info] Model ready.\n")

        if args.wav:
            print(f"[info] Transcribing: {args.wav}\n")
        else:
            print("[info] Listening on microphone — press Ctrl+C to stop.\n")

        run_streaming(recognizer, audio, sample_rate=cfg.sample_rate)


if __name__ == "__main__":
    main()
