"""Streaming ASR — entry point.

Usage:
    # Zipformer streaming (default, auto-detects model type):
    python3 main.py --mic
    python3 main.py --wav path/to/audio.wav

    # Hint the architecture explicitly (speeds up model loading):
    python3 main.py --mic --model-type zipformer2

    # Parakeet TDT fp16 (offline, better accuracy):
    python3 main.py --mic --offline --model-type nemo_transducer

    # Parakeet TDT int8 (offline, smaller & faster, slightly less accurate):
    python3 main.py --mic --offline --model-type nemo_transducer --model-dir models/parakeet-tdt-0.6b-v2-int8

    # Whisper (offline):
    python3 main.py --mic --offline --model-type whisper --language en

    # SenseVoice (offline):
    python3 main.py --mic --offline --model-type sense_voice

    # Custom model directory:
    python3 main.py --mic --model-dir models/my-model --offline --model-type nemo_transducer

    Models are stored under  models/<model-name>/  at the project root:
      models/zipformer-en-2023/            (online transducer, default)
      models/parakeet-tdt-0.6b-v2/         (offline, fp16 — larger, more accurate)
      models/parakeet-tdt-0.6b-v2-int8/    (offline, int8 — smaller & faster)
      models/silero_vad.onnx               (VAD, shared for offline use)

    Online --model-type values:  (blank), transducer, zipformer, zipformer2,
                                 conformer, lstm, paraformer, ctc, wenet_ctc,
                                 zipformer2_ctc
    Offline --model-type values: (blank), transducer, nemo_transducer, paraformer,
                                 whisper, ctc, nemo_ctc, sense_voice, moonshine,
                                 fire_red_asr
"""

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf

from asr_engine import build_offline_recognizer, build_recognizer, build_vad
from audio import mic_stream, read_wav
from config import Config
from streaming import run_offline_vad_streaming, run_streaming


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming ASR with Sherpa-ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mic", action="store_true", help="Stream from microphone")
    mode.add_argument("--wav", metavar="PATH", help="Transcribe a WAV file")

    parser.add_argument(
        "--model-dir",
        default=None,
        metavar="PATH",
        help=(
            "Path to the model directory. "
            "Default: models/zipformer-en-2023 (online) or models/parakeet-tdt-0.6b-v2 (--offline)."
        ),
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate (Hz)")
    parser.add_argument(
        "--chunk-size", type=float, default=0.16, help="Chunk size in seconds (0.1–0.2 recommended)"
    )
    parser.add_argument("--threads", type=int, default=4, help="CPU thread count for ONNX runtime")
    parser.add_argument(
        "--model-type",
        default="",
        metavar="TYPE",
        help=(
            "Model architecture hint passed to sherpa-onnx. Leave blank for auto-detect. "
            "Online: transducer, zipformer, zipformer2, conformer, lstm, paraformer, "
            "ctc, wenet_ctc, zipformer2_ctc. "
            "Offline: transducer, nemo_transducer, paraformer, whisper, ctc, nemo_ctc, "
            "sense_voice, moonshine, fire_red_asr. "
            "See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/"
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the offline (VAD-segmented) pipeline instead of streaming (required for Whisper, NeMo, SenseVoice, etc.)",
    )
    parser.add_argument(
        "--capture-rate",
        type=int,
        default=16000,
        metavar="HZ",
        help="Microphone capture sample rate (Hz). Use 48000 for better device compatibility; sherpa-onnx resamples internally.",
    )
    parser.add_argument(
        "--vad-model",
        default="",
        metavar="PATH",
        help="Path to silero_vad.onnx; auto-downloaded when needed for offline model types",
    )
    parser.add_argument(
        "--language",
        default="en",
        metavar="LANG",
        help="Language code for Whisper and SenseVoice models (e.g. en, zh, ja)",
    )
    parser.add_argument(
        "--listening",
        action="store_true",
        help="Show a live RMS energy bar for microphone level calibration",
    )
    return parser.parse_args()


_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
)
_MODEL_ARCHIVE = "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
_MODEL_EXTRACTED = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
_MODEL_TARGET = "zipformer-en-2023"

_PARAKEET_FP16_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16.tar.bz2"
)
_PARAKEET_FP16_ARCHIVE = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16.tar.bz2"
_PARAKEET_FP16_EXTRACTED = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16"
_PARAKEET_FP16_TARGET = "parakeet-tdt-0.6b-v2"

_PARAKEET_INT8_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
)
_PARAKEET_INT8_ARCHIVE = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
_PARAKEET_INT8_EXTRACTED = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
_PARAKEET_INT8_TARGET = "parakeet-tdt-0.6b-v2-int8"

# Default offline model (fp16)
_PARAKEET_TARGET = _PARAKEET_FP16_TARGET

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

    # Use parakeet as the default offline model download target
    if model_type == "nemo_transducer" or model_dir.name in (
        _PARAKEET_FP16_TARGET, _PARAKEET_INT8_TARGET
    ):
        # Choose variant based on directory name
        if "int8" in model_dir.name:
            url = _PARAKEET_INT8_URL
            archive_name = _PARAKEET_INT8_ARCHIVE
            extracted_name = _PARAKEET_INT8_EXTRACTED
        else:
            url = _PARAKEET_FP16_URL
            archive_name = _PARAKEET_FP16_ARCHIVE
            extracted_name = _PARAKEET_FP16_EXTRACTED
    else:
        url = _MODEL_URL
        archive_name = _MODEL_ARCHIVE
        extracted_name = _MODEL_EXTRACTED

    # Download into models/ alongside the target directory
    models_dir = model_dir.parent
    models_dir.mkdir(parents=True, exist_ok=True)
    archive = models_dir / archive_name
    print(f"[info] Model not found.")
    _download_file(url, archive)

    print("[info] Extracting…")
    try:
        with tarfile.open(archive, "r:bz2") as tf:
            tf.extractall(models_dir, filter="data")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"[error] Extraction failed: {exc}")

    extracted = models_dir / extracted_name
    if not extracted.is_dir():
        sys.exit(f"[error] Expected extracted directory '{extracted_name}' not found.")

    extracted.rename(model_dir)
    archive.unlink(missing_ok=True)
    print(f"[info] Model saved to '{model_dir}'.\n")


def _validate_model(model_dir: str, model_type: str) -> None:
    if not Path(model_dir).is_dir():
        _download_model(model_dir, model_type)


def _validate_vad(vad_model: str, offline: bool, project_dir: Path) -> str:
    if not offline:
        return vad_model
    if not vad_model:
        vad_path = project_dir / "models" / "silero_vad.onnx"
        if not vad_path.exists():
            vad_path.parent.mkdir(parents=True, exist_ok=True)
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

    # Resolve paths relative to the project root (one level above src/).
    project_dir = Path(__file__).resolve().parent.parent
    # Use a type-specific default dir when the user didn't pass --model-dir explicitly.
    if args.model_dir is None:
        raw_model_dir = (
            f"models/{_PARAKEET_TARGET}"
            if args.offline
            else f"models/{_MODEL_TARGET}"
        )
    else:
        raw_model_dir = args.model_dir
    model_dir = Path(raw_model_dir)
    if not model_dir.is_absolute():
        model_dir = project_dir / model_dir

    cfg = Config(
        model_dir=str(model_dir),
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        num_threads=args.threads,
        model_type=args.model_type,
        offline=args.offline,
        vad_model=args.vad_model,
        language=args.language,
        show_mic_level=args.listening,
    )

    _validate_model(cfg.model_dir, cfg.model_type)

    # Auto-detect offline-only models and switch automatically.
    _OFFLINE_ONLY_TYPES = {"nemo_transducer", "whisper", "nemo_ctc", "sense_voice", "moonshine", "fire_red_asr"}
    _OFFLINE_ONLY_NAME_PATTERNS = ("parakeet", "nemo", "whisper", "sense_voice", "moonshine", "fire_red_asr")
    model_name_lower = Path(cfg.model_dir).name.lower()
    if not cfg.offline and (
        cfg.model_type.lower() in _OFFLINE_ONLY_TYPES
        or any(pat in model_name_lower for pat in _OFFLINE_ONLY_NAME_PATTERNS)
    ):
        print(
            f"[info] Model '{Path(cfg.model_dir).name}' is offline-only — "
            "enabling --offline automatically."
        )
        cfg.offline = True

    cfg.vad_model = _validate_vad(cfg.vad_model, cfg.offline, project_dir)

    if args.wav:
        _validate_wav(args.wav, cfg.sample_rate)
    else:
        _validate_mic()

    model_name = Path(cfg.model_dir).name
    print(f"[info] Loading model '{model_name}' ({cfg.num_threads} threads)…")

    if args.wav:
        audio = read_wav(args.wav, target_sr=cfg.sample_rate, chunk_size=cfg.chunk_size)
        capture_rate = cfg.sample_rate
    else:
        capture_rate = args.capture_rate
        audio = mic_stream(capture_rate=capture_rate, chunk_size=cfg.chunk_size)

    if cfg.offline:
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
            sample_rate=capture_rate,
            show_mic_level=cfg.show_mic_level,
        )
    else:
        recognizer = build_recognizer(cfg)
        print("[info] Model ready.\n")

        if args.wav:
            print(f"[info] Transcribing: {args.wav}\n")
        else:
            print("[info] Listening on microphone — press Ctrl+C to stop.\n")

        run_streaming(recognizer, audio, sample_rate=capture_rate, show_mic_level=cfg.show_mic_level)


if __name__ == "__main__":
    main()
