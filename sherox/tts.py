"""Text-to-speech synthesis — entry point.

Usage:
    # Synthesise from inline text (Indonesian, default):
    sherox.tts --text "Selamat pagi, apa kabar?"

    # Read from file:
    sherox.tts --file input.txt --lang ind

    # Read from stdin:
    echo "Halo dunia" | sherox.tts --lang ind

    # Save to a specific output file:
    sherox.tts --text "Halo" --output halo.wav

    # Play through the system speaker (requires sounddevice):
    sherox.tts --text "Halo" --play

    # Control speech speed:
    sherox.tts --text "Halo" --speed 0.85

Supported languages (ISO 639-3):
    ind   Indonesian  — vits-piper-id_ID-news_tts-medium  (22050 Hz, 1 speaker)

Models are auto-downloaded on first use into  models/<model-dir>/  at the project root.
"""

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
from rich.console import Console

from .config import TtsConfig

sf = SimpleNamespace(write=None)

_console = Console()
_err_console = Console(stderr=True)

# ── Model registry (ISO 639-3 → model metadata) ──────────────────────────────

_TTS_MODELS: dict[str, dict] = {
    "ind": {
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/vits-piper-id_ID-news_tts-medium.tar.bz2"
        ),
        "archive": "vits-piper-id_ID-news_tts-medium.tar.bz2",
        "extracted": "vits-piper-id_ID-news_tts-medium",
        "model": "id_ID-news_tts-medium.onnx",
        "tokens": "tokens.txt",
        "data_dir": "espeak-ng-data",
        "sample_rate": 22050,
        "description": "Indonesian (Piper VITS, medium quality)",
    },
}

_SUPPORTED_LANGS = ", ".join(
    f"{code} ({meta['description']})" for code, meta in _TTS_MODELS.items()
)


def _info(msg: str) -> None:
    _console.print(f"[bold green]\\[info][/bold green] {msg}")


def _error(msg: str) -> None:
    _err_console.print(f"[bold red]\\[error][/bold red] {msg}")
    sys.exit(1)


def _require_soundfile():
    global sf
    if getattr(sf, "write", None) is not None:
        return sf
    try:
        import soundfile as _soundfile  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        _error(
            "soundfile is required for writing synthesized audio. "
            "Install it with: pip install soundfile"
        )
        raise AssertionError("unreachable") from exc
    sf = _soundfile
    return sf


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if args.speaker_id < 0:
        _error(f"--speaker-id must be >= 0, got {args.speaker_id}")
    if args.speed <= 0:
        _error(f"--speed must be > 0, got {args.speed}")
    if args.threads <= 0:
        _error(f"--threads must be > 0, got {args.threads}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text-to-speech synthesis with Sherpa-ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    text_src = parser.add_mutually_exclusive_group()
    text_src.add_argument("--text", metavar="TEXT", help="Text to synthesise")
    text_src.add_argument("--file", metavar="PATH", help="Read text from a file")

    parser.add_argument(
        "--lang",
        default="ind",
        metavar="LANG",
        help=f"ISO 639-3 language code. Supported: {_SUPPORTED_LANGS}",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        metavar="PATH",
        help=(
            "Path to a custom TTS model directory. "
            "If omitted, the default model for --lang is used."
        ),
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        metavar="N",
        help="Speaker identity index for multi-speaker models",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="F",
        help="Speech rate multiplier (0.5 = slower, 2.0 = faster)",
    )
    parser.add_argument(
        "--output",
        default="output.wav",
        metavar="PATH",
        help="Output WAV file path",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play audio through the default output device after synthesis",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU thread count for ONNX runtime",
    )
    return parser.parse_args()


# ── Model download helpers ────────────────────────────────────────────────────

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
    except Exception as exc:
        _error(f"Download failed: {exc}")
    print()


def _safe_tar_members(tf: tarfile.TarFile, dest_dir: Path):
    """Yield only safe members, preventing path traversal."""
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


def _ensure_model(lang: str, model_dir: Optional[Path], project_dir: Path) -> Path:
    """Return the resolved TTS model directory, downloading if needed."""
    if lang not in _TTS_MODELS:
        _error(
            f"Unsupported language '{lang}'. Supported: {list(_TTS_MODELS.keys())}"
        )

    meta = _TTS_MODELS[lang]
    if model_dir is not None:
        if not model_dir.is_dir():
            _error(f"Model directory not found: {model_dir}")
        return model_dir

    models_root = project_dir / "models"
    target_dir = models_root / meta["extracted"]

    if target_dir.is_dir():
        return target_dir

    # Download and extract
    models_root.mkdir(parents=True, exist_ok=True)
    archive = models_root / meta["archive"]
    _info(f"TTS model for '{lang}' not found.")
    _download_file(meta["url"], archive)

    _info("Extracting…")
    try:
        with tarfile.open(archive, "r:bz2") as tf:
            if sys.version_info >= (3, 12):
                tf.extractall(models_root, filter="data")
            else:
                tf.extractall(models_root, members=_safe_tar_members(tf, models_root))
    except Exception as exc:
        _error(f"Extraction failed: {exc}")

    archive.unlink(missing_ok=True)

    if not target_dir.is_dir():
        _error(f"Expected model directory not found after extraction: {target_dir}")

    _info(f"Model saved to '{target_dir}'.\n")
    return target_dir


# ── TTS engine ────────────────────────────────────────────────────────────────

def build_tts(cfg: TtsConfig, project_dir: Path):
    """Build a sherpa_onnx.OfflineTts from *cfg*, downloading the model if needed."""
    import sherpa_onnx  # noqa: PLC0415

    lang = cfg.language
    if lang not in _TTS_MODELS:
        _error(
            f"Unsupported language '{lang}'. Supported: {list(_TTS_MODELS.keys())}"
        )

    meta = _TTS_MODELS[lang]
    model_dir_override = Path(cfg.model_dir) if cfg.model_dir else None
    model_dir = _ensure_model(lang, model_dir_override, project_dir)

    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(model_dir / meta["model"]),
                lexicon="",
                data_dir=str(model_dir / meta["data_dir"]),
                tokens=str(model_dir / meta["tokens"]),
            ),
            num_threads=cfg.num_threads,
        ),
    )

    if not config.validate():
        _error(
            "TTS config is invalid — check that all model files exist and are valid."
        )

    return sherpa_onnx.OfflineTts(config)


def synthesise(tts, text: str, cfg: TtsConfig) -> tuple[np.ndarray, int]:
    """Synthesise *text* and return (samples, sample_rate)."""
    audio = tts.generate(text=text, sid=cfg.speaker_id, speed=cfg.speed)
    samples = np.array(audio.samples, dtype=np.float32)
    return samples, audio.sample_rate


def _play(samples: np.ndarray, sample_rate: int) -> None:
    try:
        import sounddevice as sd  # noqa: PLC0415
    except ImportError:
        _error(
            "sounddevice is required for --play. "
            "Install it with: pip install sounddevice"
        )
    sd.play(samples, samplerate=sample_rate)
    sd.wait()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    _validate_runtime_args(args)

    project_dir = Path(__file__).resolve().parent.parent

    # Resolve text input
    if args.text:
        text = args.text
    elif args.file:
        p = Path(args.file)
        if not p.exists():
            _error(f"Input file not found: {args.file}")
        text = p.read_text(encoding="utf-8").strip()
    else:
        if sys.stdin.isatty():
            _console.print("[dim]Reading text from stdin (Ctrl+D to finish)…[/dim]")
        text = sys.stdin.read().strip()

    if not text:
        _error("No text provided. Use --text, --file, or pipe text via stdin.")

    model_dir_arg = Path(args.model_dir) if args.model_dir else None
    cfg = TtsConfig(
        model_dir=str(model_dir_arg) if model_dir_arg else "",
        language=args.lang,
        speaker_id=args.speaker_id,
        speed=args.speed,
        output=args.output,
        play=args.play,
        num_threads=args.threads,
    )

    _info(f"Language: {cfg.language}  |  speed: {cfg.speed}  |  speaker: {cfg.speaker_id}")
    _info("Loading TTS model…")
    tts = build_tts(cfg, project_dir)
    _info("Synthesising…")

    samples, sample_rate = synthesise(tts, text, cfg)

    if cfg.play:
        _info("Playing audio…")
        _play(samples, sample_rate)

    sf = _require_soundfile()
    sf.write(cfg.output, samples, samplerate=sample_rate)
    _info(f"Saved → {cfg.output}")


if __name__ == "__main__":
    main()
