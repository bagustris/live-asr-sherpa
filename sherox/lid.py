"""Spoken Language Identification — entry point.

Uses multilingual Whisper encoder/decoder ONNX models (tiny / base / small /
medium) published by sherpa-onnx to detect the language spoken in audio.

Usage:
    # Identify language in a WAV file:
    sherox.lid --wav audio.wav

    # Continuously identify language from microphone (VAD-segmented):
    sherox.lid --mic

    # Use a larger Whisper variant for better accuracy:
    sherox.lid --wav audio.wav --size base

    # Point at a custom encoder/decoder pair:
    sherox.lid --wav audio.wav \
        --encoder models/whisper-tiny/tiny-encoder.int8.onnx \
        --decoder models/whisper-tiny/tiny-decoder.int8.onnx

Default model: sherpa-onnx-whisper-tiny (encoder+decoder, ~75 MB,
auto-downloaded on first run).
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path
from typing import Tuple

import numpy as np
from rich.console import Console

from .asr_engine import _require_sherpa_onnx, build_vad
from .audio import mic_stream
from .config import Config as _AsrConfig, LidConfig
from .utils import download_file as _download_file

_console = Console()
_err_console = Console(stderr=True)

_PREFIX = "  "

_SUPPORTED_SIZES = ("tiny", "base", "small", "medium")

_MODEL_URL_TEMPLATE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-whisper-{size}.tar.bz2"
)

_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/silero_vad.onnx"
)

# Whisper's full set of supported language codes (ISO 639-1 + a few 3-letter
# codes that Whisper uses verbatim — "haw", "jw"). Names are Title Case so
# they display nicely in the terminal.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish",
    "ru": "Russian", "ko": "Korean", "fr": "French", "ja": "Japanese",
    "pt": "Portuguese", "tr": "Turkish", "pl": "Polish", "ca": "Catalan",
    "nl": "Dutch", "ar": "Arabic", "sv": "Swedish", "it": "Italian",
    "id": "Indonesian", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese",
    "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay",
    "cs": "Czech", "ro": "Romanian", "da": "Danish", "hu": "Hungarian",
    "ta": "Tamil", "no": "Norwegian", "th": "Thai", "ur": "Urdu",
    "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian", "la": "Latin",
    "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
    "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali",
    "sr": "Serbian", "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada",
    "et": "Estonian", "mk": "Macedonian", "br": "Breton", "eu": "Basque",
    "is": "Icelandic", "hy": "Armenian", "ne": "Nepali", "mn": "Mongolian",
    "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili",
    "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala",
    "km": "Khmer", "sn": "Shona", "yo": "Yoruba", "so": "Somali",
    "af": "Afrikaans", "oc": "Occitan", "ka": "Georgian", "be": "Belarusian",
    "tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati", "am": "Amharic",
    "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek", "fo": "Faroese",
    "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen", "nn": "Nynorsk",
    "mt": "Maltese", "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Myanmar",
    "bo": "Tibetan", "tl": "Tagalog", "mg": "Malagasy", "as": "Assamese",
    "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala", "ha": "Hausa",
    "ba": "Bashkir", "jw": "Javanese", "su": "Sundanese", "yue": "Cantonese",
}


def language_name(code: str) -> str:
    """Return the human-readable name for a Whisper language code.

    Falls back to the code itself when the language is unknown so callers
    always get something printable.
    """
    return LANGUAGE_NAMES.get(code.lower(), code)


def _format_language(code: str) -> str:
    """Format a detected language code for display, e.g. 'id (Indonesian)'."""
    if not code or code == "unknown":
        return "unknown"
    name = LANGUAGE_NAMES.get(code.lower())
    return f"{code} ({name})" if name else code


def _info(msg: str) -> None:
    _console.print(f"[bold green]\\[info][/bold green] {msg}")


def _error(msg: str) -> None:
    _err_console.print(f"[bold red]\\[error][/bold red] {msg}")
    sys.exit(1)


def _safe_tar_members(tf: tarfile.TarFile, dest_dir: Path):
    """Yield only safe members, preventing path traversal on Python < 3.12."""
    dest_resolved = dest_dir.resolve()
    for member in tf.getmembers():
        if member.isdev():
            continue
        member_path = (dest_dir / member.name).resolve()
        try:
            member_path.relative_to(dest_resolved)
        except ValueError:
            continue
        yield member


def _resolve_model(cfg: LidConfig, project_dir: Path) -> Tuple[str, str]:
    """Resolve encoder/decoder paths, downloading and extracting if absent.

    Returns (encoder_path, decoder_path).
    """
    if cfg.encoder and cfg.decoder:
        enc, dec = Path(cfg.encoder), Path(cfg.decoder)
        if not enc.is_file():
            _error(f"Encoder model not found: {enc}")
        if not dec.is_file():
            _error(f"Decoder model not found: {dec}")
        return str(enc), str(dec)

    if cfg.size not in _SUPPORTED_SIZES:
        _error(
            f"Unsupported --size '{cfg.size}'. "
            f"Choose one of: {', '.join(_SUPPORTED_SIZES)}."
        )

    models_root = project_dir / "models"
    target_dir = models_root / f"sherpa-onnx-whisper-{cfg.size}"
    encoder_path = target_dir / f"{cfg.size}-encoder.int8.onnx"
    decoder_path = target_dir / f"{cfg.size}-decoder.int8.onnx"

    if encoder_path.is_file() and decoder_path.is_file():
        return str(encoder_path), str(decoder_path)

    models_root.mkdir(parents=True, exist_ok=True)
    archive = models_root / f"sherpa-onnx-whisper-{cfg.size}.tar.bz2"
    _info(f"Whisper-{cfg.size} LID model not found. Downloading…")
    _download_file(_MODEL_URL_TEMPLATE.format(size=cfg.size), archive)

    _info("Extracting…")
    try:
        with tarfile.open(archive, "r:bz2") as tf:
            if sys.version_info >= (3, 12):
                tf.extractall(models_root, filter="data")
            else:  # pragma: no cover
                tf.extractall(models_root, members=_safe_tar_members(tf, models_root))
    except Exception as exc:  # noqa: BLE001
        _error(f"Extraction failed: {exc}")

    archive.unlink(missing_ok=True)

    if not encoder_path.is_file() or not decoder_path.is_file():
        _error(
            "Expected encoder/decoder files not found after extraction in "
            f"'{target_dir}'."
        )

    _info(f"Model saved to '{target_dir}'.\n")
    return str(encoder_path), str(decoder_path)


def _validate_vad(project_dir: Path) -> str:
    vad_path = project_dir / "models" / "silero_vad.onnx"
    if not vad_path.exists():
        vad_path.parent.mkdir(parents=True, exist_ok=True)
        _info("VAD model not found. Downloading silero_vad.onnx…")
        _download_file(_VAD_URL, vad_path)
    return str(vad_path)


def _load_wav_flat(path: str) -> Tuple[np.ndarray, int]:
    """Load an audio file and return (float32 mono samples, sample_rate)."""
    try:
        import soundfile as sf  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("soundfile is required. pip install soundfile") from exc
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return np.ascontiguousarray(data[:, 0]), sr


def _build_slid(cfg: LidConfig):
    sherpa_onnx = _require_sherpa_onnx()
    config = sherpa_onnx.SpokenLanguageIdentificationConfig(
        whisper=sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
            encoder=cfg.encoder,
            decoder=cfg.decoder,
        ),
        num_threads=cfg.num_threads,
        debug=False,
        provider=cfg.provider,
    )
    return sherpa_onnx.SpokenLanguageIdentification(config)


def _identify(slid, samples: np.ndarray, sample_rate: int) -> str:
    stream = slid.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
    lang = slid.compute(stream)
    return lang if lang else "unknown"


def _render(lang: str) -> str:
    """Build a Rich-formatted line for a detected language code."""
    if lang == "unknown":
        return f"{_PREFIX}[bold yellow]unknown[/bold yellow]"
    name = LANGUAGE_NAMES.get(lang.lower())
    if name:
        return (
            f"{_PREFIX}[bold bright_cyan]{lang}[/bold bright_cyan] "
            f"[dim]({name})[/dim]"
        )
    return f"{_PREFIX}[bold bright_cyan]{lang}[/bold bright_cyan]"


def run_wav(cfg: LidConfig) -> None:
    slid = _build_slid(cfg)

    _info(f"Processing: {cfg.wav}\n")
    samples, sr = _load_wav_flat(cfg.wav)
    lang = _identify(slid, samples, sr)
    _console.print(_render(lang))


def run_mic(cfg: LidConfig) -> None:
    slid = _build_slid(cfg)

    asr_cfg = _AsrConfig(
        vad_model=cfg.vad_model,
        vad_type="silero",
        sample_rate=cfg.capture_rate,
        num_threads=cfg.num_threads,
    )
    vad = build_vad(asr_cfg)

    def _process(samples: np.ndarray) -> None:
        lang = _identify(slid, samples, cfg.capture_rate)
        sys.stdout.write(f"\r{' ' * 44}\r")
        sys.stdout.flush()
        _console.print(_render(lang))

    audio = mic_stream(capture_rate=cfg.capture_rate, chunk_size=cfg.chunk_size)
    _info("Listening on microphone — press Ctrl+C to stop.\n")

    try:
        for chunk in audio:
            vad.accept_waveform(chunk)

            if cfg.show_mic_level:
                energy = float(np.sqrt(np.mean(chunk ** 2)))
                bar = "█" * min(int(energy * 500), 40)
                sys.stdout.write(f"\r{_PREFIX}mic: {bar:<40} {energy:.4f}")
                sys.stdout.flush()

            while not vad.empty():
                segment = vad.front
                samples = np.array(segment.samples, dtype=np.float32)
                vad.pop()
                _process(samples)

    except KeyboardInterrupt:
        pass
    finally:
        vad.flush()
        while not vad.empty():
            segment = vad.front
            samples = np.array(segment.samples, dtype=np.float32)
            vad.pop()
            _process(samples)
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spoken Language Identification with Sherpa-ONNX (Whisper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mic", action="store_true", help="Stream from microphone")
    mode.add_argument("--wav", metavar="PATH", help="Identify language in a WAV file")

    parser.add_argument(
        "--size",
        default="tiny",
        choices=_SUPPORTED_SIZES,
        help="Whisper variant (controls which model archive is auto-downloaded)",
    )
    parser.add_argument(
        "--encoder",
        default="",
        metavar="PATH",
        help="Path to a custom Whisper encoder .onnx (overrides --size)",
    )
    parser.add_argument(
        "--decoder",
        default="",
        metavar="PATH",
        help="Path to a custom Whisper decoder .onnx (overrides --size)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Expected sample rate for WAV input (Hz)",
    )
    parser.add_argument(
        "--capture-rate", type=int, default=16000, metavar="HZ",
        help="Microphone capture rate (use 48000 for device compatibility)",
    )
    parser.add_argument(
        "--chunk-size", type=float, default=0.1,
        help="Mic audio chunk size in seconds",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="CPU thread count for ONNX runtime",
    )
    parser.add_argument(
        "--provider", default="cpu",
        choices=["cpu", "cuda", "coreml"],
        help="ONNX Runtime execution provider",
    )
    parser.add_argument(
        "--listening", action="store_true",
        help="Show a live RMS energy bar for microphone level calibration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent.parent

    if bool(args.encoder) ^ bool(args.decoder):
        _error("--encoder and --decoder must be provided together.")

    cfg = LidConfig(
        encoder=args.encoder,
        decoder=args.decoder,
        size=args.size,
        num_threads=args.threads,
        provider=args.provider,
        sample_rate=args.sample_rate,
        capture_rate=args.capture_rate,
        chunk_size=args.chunk_size,
        wav=args.wav or "",
        show_mic_level=args.listening,
    )

    cfg.encoder, cfg.decoder = _resolve_model(cfg, project_dir)

    if args.mic:
        cfg.vad_model = _validate_vad(project_dir)

    _info("Loading Whisper LID model…")

    if args.wav:
        run_wav(cfg)
    else:
        run_mic(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
