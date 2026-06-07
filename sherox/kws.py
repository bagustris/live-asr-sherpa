"""Keyword Spotting — entry point.

Listens for one or more trigger words in real-time microphone audio or a WAV
file and prints each match with a timestamp.

Usage::

    # Microphone — watch for two keywords:
    sherox.kws --mic --keywords "hey sherpa, ok google"

    # WAV file — same keywords:
    sherox.kws --wav audio.wav --keywords "hey sherpa"

    # Load keywords from a file (one per line):
    sherox.kws --mic --keywords-file my_keywords.txt

    # Custom model directory:
    sherox.kws --mic --keywords "hey sherpa" --model-dir models/my-kws-model

Keyword file format (plain text, one keyword per line)::

    hey sherpa
    ok google
    stop recording

Notes
-----
* The default model is ``sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01``
  (English, ~3.3 MB, auto-downloaded on first run).
* Keywords are **case-insensitive** at the specification level — the model
  internally uses its own token vocabulary.
* Each keyword hit is printed as ``[HH:MM:SS] keyword: <matched text>``.
* After a hit the stream is automatically reset so the same keyword can
  trigger again on subsequent utterances.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from .config import KwsConfig
from .utils import _error, _info, download_file, render_mic_level, run_cli as _run_cli, safe_tar_members as _safe_tar_members

# ── Model constants ───────────────────────────────────────────────────────────

_MODEL_NAME = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2"
)


# ── Lazy imports ──────────────────────────────────────────────────────────────

def _require_sherpa_onnx():
    try:
        import sherpa_onnx  # noqa: PLC0415
        return sherpa_onnx
    except ImportError:
        _error(
            "sherpa-onnx is not installed. Run: pip install sherpa-onnx"
        )


def _require_sounddevice():
    try:
        import sounddevice as sd  # noqa: PLC0415
        return sd
    except ImportError:
        _error(
            "sounddevice is not installed. Run: pip install sounddevice"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_model(model_dir_override: str, project_dir: Path) -> Path:
    """Return the resolved model directory, auto-downloading if needed."""
    if model_dir_override:
        p = Path(model_dir_override)
        if not p.is_dir():
            _error(f"Model directory not found: {model_dir_override}")
        return p

    model_dir = project_dir / "models" / _MODEL_NAME
    if model_dir.is_dir():
        return model_dir

    archive = project_dir / "models" / f"{_MODEL_NAME}.tar.bz2"
    _info(f"Downloading KWS model → {archive}")
    download_file(_MODEL_URL, archive)

    _info("Extracting…")
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "r:bz2") as tf:
            if sys.version_info >= (3, 12):
                tf.extractall(models_dir, filter="data")
            else:
                tf.extractall(models_dir, members=_safe_tar_members(tf, models_dir))
    except Exception as exc:  # noqa: BLE001
        _error(f"Extraction failed: {exc}")
    archive.unlink(missing_ok=True)

    if not model_dir.is_dir():
        _error(f"Model directory not found after extraction: {model_dir}")
    return model_dir


def _find(directory: Path, pattern: str) -> Path:
    """Return the first file matching *pattern* inside *directory*."""
    matches = sorted(directory.rglob(pattern))
    if not matches:
        _error(f"No file matching '{pattern}' found in {directory}")
    return matches[0]


def _resolve_keywords(cfg: KwsConfig, model_dir: Path) -> str:
    """Return a path to a keywords file, creating a temp file if needed.

    sherpa-onnx's ``KeywordSpotter`` takes a ``keywords_file`` parameter — a
    path to a plain-text file with one keyword per line.  When the caller
    supplies keywords as a comma-separated string we write them to a temporary
    file and return that path.
    """
    if cfg.keywords_str:
        words = [w.strip() for w in cfg.keywords_str.split(",") if w.strip()]
        if not words:
            _error("--keywords produced an empty keyword list.")

        bpe_model = model_dir / "bpe.model"
        if bpe_model.is_file():
            import sentencepiece as spm  # noqa: PLC0415
            sp = spm.SentencePieceProcessor(model_file=str(bpe_model))
            lines = [" ".join(sp.encode(w.upper(), out_type=str)) for w in words]
        else:
            lines = words

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write("\n".join(lines) + "\n")
            return tmp.name

    if cfg.keywords_file:
        p = Path(cfg.keywords_file)
        if not p.is_file():
            _error(f"Keywords file not found: {cfg.keywords_file}")
        return str(p)

    _error("Provide --keywords or --keywords-file.")


def _build_spotter(model_dir: Path, keywords_file_path: str, cfg: KwsConfig):
    """Construct and return a ``sherpa_onnx.KeywordSpotter`` instance."""
    sherpa_onnx = _require_sherpa_onnx()

    return sherpa_onnx.KeywordSpotter(
        tokens=str(_find(model_dir, "tokens.txt")),
        encoder=str(_find(model_dir, "encoder*.onnx")),
        decoder=str(_find(model_dir, "decoder*.onnx")),
        joiner=str(_find(model_dir, "joiner*.onnx")),
        keywords_file=keywords_file_path,
        num_threads=cfg.num_threads,
        sample_rate=cfg.sample_rate,
        max_active_paths=cfg.max_active_paths,
    )


# ── Runtime loops ─────────────────────────────────────────────────────────────

def run_mic(spotter, cfg: KwsConfig) -> None:
    """Stream microphone audio into *spotter* and print keyword hits."""
    sd = _require_sounddevice()
    stream = spotter.create_stream()
    samples_per_chunk = int(cfg.sample_rate * cfg.chunk_size)

    _info("Listening… (press Ctrl+C to stop)")
    start_time = time.time()

    import numpy as np  # noqa: PLC0415

    with sd.InputStream(
        samplerate=cfg.capture_rate,
        channels=1,
        dtype="float32",
        blocksize=int(cfg.capture_rate * cfg.chunk_size),
    ) as mic:
        try:
            while True:
                audio, _ = mic.read(int(cfg.capture_rate * cfg.chunk_size))
                samples = audio.flatten()

                # Resample if capture_rate ≠ model sample_rate.
                if cfg.capture_rate != cfg.sample_rate:
                    ratio = cfg.sample_rate / cfg.capture_rate
                    new_len = int(len(samples) * ratio)
                    samples = np.interp(
                        np.linspace(0, len(samples) - 1, new_len),
                        np.arange(len(samples)),
                        samples,
                    ).astype(np.float32)

                if cfg.show_mic_level:
                    render_mic_level(samples)

                stream.accept_waveform(cfg.sample_rate, samples)

                while spotter.is_ready(stream):
                    spotter.decode_stream(stream)

                result = spotter.get_result(stream)
                if result:
                    elapsed = time.time() - start_time
                    h, rem = divmod(int(elapsed), 3600)
                    m, s = divmod(rem, 60)
                    if cfg.show_mic_level:
                        sys.stdout.write(f"\r{' ' * 54}\r")
                    print(f"[{h:02d}:{m:02d}:{s:02d}] keyword: {result}")
                    # Reset stream so the same keyword can trigger again.
                    spotter.reset_stream(stream)

        except KeyboardInterrupt:
            pass


def run_wav(spotter, cfg: KwsConfig) -> None:
    """Feed a WAV file through *spotter* and print keyword hits."""
    try:
        import soundfile as sf  # noqa: PLC0415
    except ImportError:
        _error("soundfile is not installed. Run: pip install soundfile")

    import numpy as np  # noqa: PLC0415

    audio, file_sr = sf.read(cfg.wav, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio[:, 0]  # take first channel

    # Resample to model sample rate if needed.
    if file_sr != cfg.sample_rate:
        ratio = cfg.sample_rate / file_sr
        new_len = int(len(audio) * ratio)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    stream = spotter.create_stream()
    chunk = int(cfg.sample_rate * cfg.chunk_size)
    offset = 0
    hits: list[str] = []

    while offset < len(audio):
        segment = audio[offset : offset + chunk]
        stream.accept_waveform(cfg.sample_rate, segment)

        while spotter.is_ready(stream):
            spotter.decode_stream(stream)

        result = spotter.get_result(stream)
        if result:
            timestamp = offset / cfg.sample_rate
            h, rem = divmod(int(timestamp), 3600)
            m, s = divmod(rem, 60)
            print(f"[{h:02d}:{m:02d}:{s:02d}] keyword: {result}")
            hits.append(result)
            spotter.reset_stream(stream)

        offset += chunk

    if not hits:
        _info("No keywords detected.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keyword Spotting with Sherpa-ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mic", action="store_true", help="Stream from microphone")
    mode.add_argument("--wav", metavar="PATH", help="Spot keywords in a WAV file")

    kw_src = parser.add_mutually_exclusive_group(required=True)
    kw_src.add_argument(
        "--keywords",
        metavar="LIST",
        help=(
            "Comma-separated keyword phrases to detect. "
            "Example: --keywords 'hey sherpa, ok google'"
        ),
    )
    kw_src.add_argument(
        "--keywords-file",
        metavar="PATH",
        help="Plain-text file with one keyword per line",
    )

    parser.add_argument(
        "--model-dir",
        metavar="PATH",
        default="",
        help=f"Custom KWS model directory (default: auto-download {_MODEL_NAME})",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Model input sample rate (Hz)",
    )
    parser.add_argument(
        "--capture-rate", type=int, default=16000, metavar="HZ",
        help="Microphone capture rate — use 48000 for some devices",
    )
    parser.add_argument(
        "--chunk-size", type=float, default=0.1,
        help="Audio chunk duration per decode call (seconds)",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="CPU thread count for ONNX runtime",
    )
    parser.add_argument(
        "--max-active-paths", type=int, default=4,
        help="Beam width for keyword search (higher = more sensitive, slower)",
    )
    parser.add_argument(
        "--no-mic-level",
        action="store_true",
        help="Suppress the live RMS energy bar during microphone capture",
    )
    return parser.parse_args()


def main() -> None:
    _run_cli(_main_impl)


def _main_impl() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent.parent

    cfg = KwsConfig(
        model_dir=args.model_dir,
        keywords_str=args.keywords or "",
        keywords_file=args.keywords_file or "",
        sample_rate=args.sample_rate,
        capture_rate=args.capture_rate,
        chunk_size=args.chunk_size,
        num_threads=args.threads,
        max_active_paths=args.max_active_paths,
        show_mic_level=not args.no_mic_level,
    )
    if args.wav:
        cfg.wav = args.wav  # type: ignore[attr-defined]

    model_dir = _validate_model(cfg.model_dir, project_dir)
    keywords_file_path = _resolve_keywords(cfg, model_dir)

    try:
        spotter = _build_spotter(model_dir, keywords_file_path, cfg)

        if args.mic:
            run_mic(spotter, cfg)
        else:
            run_wav(spotter, cfg)
    finally:
        # Clean up temp file if we created one from --keywords string.
        if cfg.keywords_str:
            Path(keywords_file_path).unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover
    main()
