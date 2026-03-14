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

    # Speaker diarization (offline, auto-downloads lightweight models):
    python3 main.py --mic --offline --diarization

    # Speaker diarization with known speaker count:
    python3 main.py --mic --offline --diarization --num-speakers 2

    # Speaker diarization with [Speaker N] tag prefix:
    python3 main.py --mic --offline --diarization --speaker-tag

    Models are stored under  models/<model-name>/  at the project root:
      models/zipformer-en-2023/            (online transducer, default)
      models/parakeet-tdt-0.6b-v2/         (offline, fp16 — larger, more accurate)
      models/parakeet-tdt-0.6b-v2-int8/    (offline, int8 — smaller & faster)
      models/silero_vad.onnx               (VAD, shared for offline use)
      models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx  (diarization segmentation)
      models/nemo_en_speakerverification_speakernet.onnx        (diarization embedding)

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
from rich.console import Console

from asr_engine import build_diarization, build_offline_recognizer, build_recognizer, build_vad
from audio import mic_stream, read_wav
from config import Config
from streaming import run_offline_vad_streaming, run_streaming

_console = Console()
_err_console = Console(stderr=True)


def _info(msg: str) -> None:
    _console.print(f"[bold green]\\[info][/bold green] {msg}")


def _error(msg: str) -> None:
    _err_console.print(f"[bold red]\\[error][/bold red] {msg}")
    sys.exit(1)


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
        dest="vad_type",
        default="silero",
        choices=["silero", "ten-vad"],
        help="VAD model type to use for offline segmentation (default: silero).",
    )
    parser.add_argument(
        "--ten-vad-model",
        default="ten-vad.int8.onnx",
        choices=["ten-vad.onnx", "ten-vad.int8.onnx"],
        help=(
            "Ten-VAD model variant to use when --vad-model is ten-vad "
            "(default: ten-vad.int8.onnx)."
        ),
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
    parser.add_argument(
        "--diarization",
        action="store_true",
        help=(
            "Enable speaker diarization. Colours each speaker's output differently. "
            "Works with both online and offline pipelines. "
            "Diarization models are auto-downloaded on first use."
        ),
    )
    parser.add_argument(
        "--speaker-tag",
        action="store_true",
        help=(
            "Show a [Speaker N] prefix before each diarized utterance "
            "(requires --diarization). By default only the text colour differs per speaker."
        ),
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=-1,
        metavar="N",
        help=(
            "Known number of speakers for diarization (-1 = auto-detect via "
            "clustering threshold). Providing the correct count improves accuracy."
        ),
    )
    parser.add_argument(
        "--diarization-seg-model",
        default="",
        metavar="PATH",
        help="Path to pyannote segmentation model.onnx (auto-downloaded if not provided)",
    )
    parser.add_argument(
        "--diarization-emb-model",
        default="",
        metavar="PATH",
        help="Path to speaker embedding extractor .onnx (auto-downloaded if not provided)",
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

_TEN_VAD_MODEL_URLS = {
    "ten-vad.onnx": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "asr-models/ten-vad.onnx"
    ),
    "ten-vad.int8.onnx": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "asr-models/ten-vad.int8.onnx"
    ),
}

# ── Diarization model URLs (lightest available models) ───────────────────────
_DIAR_SEG_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
_DIAR_SEG_ARCHIVE = "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
_DIAR_SEG_EXTRACTED = "sherpa-onnx-pyannote-segmentation-3-0"
_DIAR_SEG_MODEL_FILE = "model.onnx"

# Lightest speaker embedding extractor (~22 MB)
_DIAR_EMB_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recognition-models/nemo_en_speakerverification_speakernet.onnx"
)
_DIAR_EMB_FILE = "nemo_en_speakerverification_speakernet.onnx"


def _download_file(url: str, dest: Path) -> None:
    _info(f"Downloading from:\n  {url}")
    _info("This may take a few minutes…")

    def _progress(block: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            sys.stdout.write(f"\r  {pct}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
    except Exception as exc:  # noqa: BLE001
        _error(f"Download failed: {exc}")
    print()


def _safe_tar_members(tf: tarfile.TarFile, dest_dir: Path):
    """Yield only safe members for extraction into dest_dir.

    This emulates the safety guarantees of `filter="data"` on Python < 3.12:
    - prevent path traversal (no entries may escape dest_dir)
    - skip device files
    """
    dest_dir_resolved = dest_dir.resolve()
    for member in tf.getmembers():
        # Skip device files and other special devices
        if member.isdev():
            continue

        member_path = (dest_dir / member.name).resolve()
        try:
            member_path.relative_to(dest_dir_resolved)
        except ValueError:
            # This member would escape dest_dir (e.g., via .. or absolute path)
            continue

        yield member


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
    _info("Model not found.")
    _download_file(url, archive)

    _info("Extracting…")
    try:
        with tarfile.open(archive, "r:bz2") as tf:
            if sys.version_info >= (3, 12):
                tf.extractall(models_dir, filter="data")
            else:
                tf.extractall(models_dir, members=_safe_tar_members(tf, models_dir))
    except Exception as exc:  # noqa: BLE001
        _error(f"Extraction failed: {exc}")

    extracted = models_dir / extracted_name
    if not extracted.is_dir():
        _error(f"Expected extracted directory '{extracted_name}' not found.")

    extracted.rename(model_dir)
    archive.unlink(missing_ok=True)
    _info(f"Model saved to '{model_dir}'.\n")


def _validate_model(model_dir: str, model_type: str) -> None:
    if not Path(model_dir).is_dir():
        _download_model(model_dir, model_type)


def _validate_vad(vad_type: str, ten_vad_model: str, offline: bool, project_dir: Path) -> str:
    if not offline:
        return ""
    if vad_type not in {"silero", "ten-vad"}:
        _error(f"Unknown --vad-model type '{vad_type}'. Supported: silero, ten-vad.")
    if vad_type == "ten-vad":
        if ten_vad_model not in _TEN_VAD_MODEL_URLS:
            _error(
                f"Unknown --ten-vad-model '{ten_vad_model}'. "
                f"Supported: {', '.join(_TEN_VAD_MODEL_URLS)}."
            )
        vad_path = project_dir / "models" / ten_vad_model
        if not vad_path.exists():
            vad_path.parent.mkdir(parents=True, exist_ok=True)
            _info(f"VAD model not found, downloading {ten_vad_model}…")
            _download_file(_TEN_VAD_MODEL_URLS[ten_vad_model], vad_path)
        return str(vad_path)
    # silero
    vad_path = project_dir / "models" / "silero_vad.onnx"
    if not vad_path.exists():
        vad_path.parent.mkdir(parents=True, exist_ok=True)
        _info("VAD model not found, downloading silero_vad.onnx…")
        _download_file(_VAD_URL, vad_path)
    return str(vad_path)


def _safe_extract_tar(tar: tarfile.TarFile, path: Path) -> None:
    """Safely extract tar contents to `path`, preventing path traversal and links.

    This emulates the behavior of `filter="data"` available in Python 3.12+,
    but is compatible with Python 3.8+.
    """
    base_path = path.resolve()
    for member in tar.getmembers():
        # Skip symlinks and hard links for safety.
        if member.issym() or member.islnk():
            continue

        member_path = (base_path / member.name).resolve()
        try:
            # Ensure the target path is within the intended base directory.
            member_path.relative_to(base_path)
        except ValueError:
            # Path traversal attempt or otherwise outside base directory; skip.
            continue

        tar.extract(member, path=base_path)


def _validate_diarization_models(
    seg_model: str, emb_model: str, project_dir: Path
) -> tuple[str, str]:
    """Return paths to diarization models, downloading them if necessary."""
    models_dir = project_dir / "models"

    # Segmentation model
    if not seg_model:
        seg_dir = models_dir / _DIAR_SEG_EXTRACTED
        seg_path = seg_dir / _DIAR_SEG_MODEL_FILE
        if not seg_path.exists():
            seg_dir.parent.mkdir(parents=True, exist_ok=True)
            _info("Diarization segmentation model not found, downloading…")
            archive = models_dir / _DIAR_SEG_ARCHIVE
            _download_file(_DIAR_SEG_URL, archive)
            _info("Extracting segmentation model…")
            try:
                with tarfile.open(archive, "r:bz2") as tf:
                    _safe_extract_tar(tf, models_dir)
            except Exception as exc:  # noqa: BLE001
                _error(f"Extraction failed: {exc}")
            archive.unlink(missing_ok=True)
            if not seg_path.exists():
                _error(f"Segmentation model not found after extraction: {seg_path}")
            _info(f"Segmentation model saved to '{seg_path}'.")
        seg_model = str(seg_path)
    elif not Path(seg_model).exists():
        _error(f"Diarization segmentation model not found: {seg_model}")

    # Embedding model
    if not emb_model:
        emb_path = models_dir / _DIAR_EMB_FILE
        if not emb_path.exists():
            models_dir.mkdir(parents=True, exist_ok=True)
            _info("Diarization embedding model not found, downloading…")
            _download_file(_DIAR_EMB_URL, emb_path)
            _info(f"Embedding model saved to '{emb_path}'.")
        emb_model = str(emb_path)
    elif not Path(emb_model).exists():
        _error(f"Diarization embedding model not found: {emb_model}")

    return seg_model, emb_model


def _validate_wav(path: str, sample_rate: int) -> None:
    p = Path(path)
    if not p.exists():
        _error(f"Audio file not found: {path}")
    try:
        with sf.SoundFile(path) as f:
            if f.channels != 1:
                _error(
                    f"Audio must be mono (1 channel), got {f.channels}.\n"
                    f"  Convert: ffmpeg -i {path} -ar {sample_rate} -ac 1 out.wav"
                )
            if f.samplerate != sample_rate:
                _error(
                    f"Audio sample rate must be {sample_rate} Hz, got {f.samplerate} Hz.\n"
                    f"  Convert: ffmpeg -i {path} -ar {sample_rate} -ac 1 out.wav"
                )
    except Exception as exc:
        _error(f"Cannot read audio file: {exc}")


def _validate_mic() -> None:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        inputs = [d for d in devices if d["max_input_channels"] > 0]
        if not inputs:
            _error("No input audio device found.")
    except Exception as exc:  # noqa: BLE001
        _error(f"Microphone check failed: {exc}")


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
        vad_type=args.vad_type,
        ten_vad_model=args.ten_vad_model,
        language=args.language,
        show_mic_level=args.listening,
        diarization=args.diarization,
        diarization_seg_model=args.diarization_seg_model,
        diarization_emb_model=args.diarization_emb_model,
        diarization_num_speakers=args.num_speakers,
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
        _info(
            f"Model '{Path(cfg.model_dir).name}' is offline-only — "
            "enabling --offline automatically."
        )
        cfg.offline = True

    cfg.vad_model = _validate_vad(cfg.vad_type, cfg.ten_vad_model, cfg.offline, project_dir)

    if args.wav:
        _validate_wav(args.wav, cfg.sample_rate)
    else:
        _validate_mic()

    # Validate / download diarization models if requested.
    diarizer = None
    if cfg.diarization:
        cfg.diarization_seg_model, cfg.diarization_emb_model = _validate_diarization_models(
            cfg.diarization_seg_model, cfg.diarization_emb_model, project_dir
        )

    model_name = Path(cfg.model_dir).name
    _info(f"Loading model '{model_name}' ({cfg.num_threads} threads)…")

    if args.wav:
        audio = read_wav(args.wav, target_sr=cfg.sample_rate, chunk_size=cfg.chunk_size)
        capture_rate = cfg.sample_rate
    else:
        capture_rate = args.capture_rate
        audio = mic_stream(capture_rate=capture_rate, chunk_size=cfg.chunk_size)

    if cfg.offline:
        recognizer = build_offline_recognizer(cfg)
        vad = build_vad(cfg)
        if cfg.diarization:
            _info("Loading diarization models…")
            diarizer = build_diarization(cfg)
        _info("Model ready.\n")

        if args.wav:
            _info(f"Transcribing: {args.wav}\n")
        else:
            _info("Listening on microphone — press Ctrl+C to stop.\n")

        run_offline_vad_streaming(
            recognizer=recognizer,
            vad=vad,
            audio_gen=audio,
            sample_rate=capture_rate,
            show_mic_level=cfg.show_mic_level,
            diarization=diarizer,
            show_speaker_tag=args.speaker_tag,
        )
    else:
        recognizer = build_recognizer(cfg)
        if cfg.diarization:
            _info("Loading diarization models…")
            diarizer = build_diarization(cfg)
        _info("Model ready.\n")

        if args.wav:
            _info(f"Transcribing: {args.wav}\n")
        else:
            _info("Listening on microphone — press Ctrl+C to stop.\n")

        run_streaming(
            recognizer,
            audio,
            sample_rate=capture_rate,
            show_mic_level=cfg.show_mic_level,
            diarization=diarizer,
            show_speaker_tag=args.speaker_tag,
        )


if __name__ == "__main__":
    main()

