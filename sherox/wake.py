"""Wake-word detection — entry point.

Listens for one or more wake-word trigger phrases in real-time microphone
audio or a WAV file and prints each match with a timestamp.  Built on top of
``livekit-wakeword`` which is fully open-source and supports custom ONNX
models for any language (including non-English names like "Hey Bagus").

Usage::

    # Pre-trained or custom ONNX model (user-trained via livekit-wakeword):
    sherox.wake --mic --model models/hey_livekit.onnx

    # Multiple wake-word models at once:
    sherox.wake --mic --model models/hey_livekit.onnx --model models/hey_jarvis.onnx

    # Process a WAV file (offline scoring):
    sherox.wake --wav audio.wav --model models/hey_livekit.onnx

    # Adjust detection threshold (0.0 - 1.0; higher = fewer false positives):
    sherox.wake --mic --model models/hey_livekit.onnx --threshold 0.7

    # Custom chunk size in seconds (default 2.0):
    sherox.wake --mic --model models/hey_livekit.onnx --chunk-size 1.5

Training a custom wake-word model
---------------------------------
``livekit-wakeword`` ships with a single-command training pipeline.  See
https://github.com/livekit/livekit-wakeword for full docs.  Quickstart::

    uv pip install livekit-wakeword[train,eval,export]
    livekit-wakeword setup
    # write configs/hey_bagus.yaml with target_phrases: ["hey bagus"]
    livekit-wakeword run configs/hey_bagus.yaml
    # model is exported to ./output/hey_bagus/hey_bagus.onnx

Notes
-----
* Audio is captured at 16 kHz mono int16 (livekit-wakeword's expected format).
* Detections are debounced: a wake-word model can only fire once every
  ``--debounce`` seconds.
* Output format: ``[HH:MM:SS] <model_name>: <confidence>`` per detection.
* Requires the ``livekit-wakeword`` package: ``uv pip install livekit-wakeword``.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .config import WakeConfig
from .utils import _info, _error, run_cli as _run_cli


# ── Lazy imports ──────────────────────────────────────────────────────────────

def _require_livekit_wakeword():
    try:
        from livekit.wakeword import WakeWordModel  # noqa: PLC0415
        return WakeWordModel
    except ImportError:
        _error(
            "livekit-wakeword is not installed. "
            "Run: uv pip install livekit-wakeword"
        )


def _require_sounddevice():
    try:
        import sounddevice as sd  # noqa: PLC0415
        return sd
    except ImportError:
        _error("sounddevice is not installed. Run: uv pip install sounddevice")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_model_paths(paths: list[str], project_dir: Path) -> list[str]:
    """Resolve a list of model paths; allow auto-discovery under models/."""
    if not paths:
        _error("Provide at least one --model PATH.")
    resolved: list[str] = []
    for p in paths:
        path = Path(p).expanduser()
        if path.is_file():
            if path.suffix.lower() != ".onnx":
                _error(f"Wake-word model must be an .onnx file: {p}")
            resolved.append(str(path))
            continue
        # Allow lookup under <project>/models/<name>.onnx
        candidate = project_dir / "models" / p
        candidates = [candidate]
        if candidate.suffix == "":
            candidates.append(candidate.with_suffix(".onnx"))
        for candidate in candidates:
            if candidate.is_file():
                resolved.append(str(candidate))
                break
        else:
            _error(f"Wake-word model not found: {p}")
    return resolved


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.threshold <= 1.0:
        _error(f"--threshold must be between 0.0 and 1.0, got {args.threshold}")
    if args.debounce < 0:
        _error(f"--debounce must be >= 0, got {args.debounce}")
    if args.chunk_size <= 0:
        _error(f"--chunk-size must be > 0, got {args.chunk_size}")
    if int(_SAMPLE_RATE * args.chunk_size) <= 0:
        _error(f"--chunk-size is too small for {_SAMPLE_RATE} Hz audio: {args.chunk_size}")
    if args.wav and not Path(args.wav).expanduser().is_file():
        _error(f"WAV file not found: {args.wav}")


def _load_model(model_paths: list[str]):
    WakeWordModel = _require_livekit_wakeword()
    try:
        return WakeWordModel(models=model_paths)
    except Exception as exc:  # noqa: BLE001
        joined = ", ".join(model_paths)
        detail = " ".join(str(exc).split())
        _error(f"Failed to load wake-word model(s) {joined}: {detail}")


def _require_soundfile():
    try:
        import soundfile as sf  # noqa: PLC0415
        return sf
    except ImportError:
        _error("soundfile is not installed. Run: uv pip install soundfile")


# ── Runtime loops ─────────────────────────────────────────────────────────────

_SAMPLE_RATE = 16000


def run_mic(model, cfg: WakeConfig) -> None:
    """Stream microphone audio into *model* and print wake-word hits."""
    sd = _require_sounddevice()
    chunk_samples = int(_SAMPLE_RATE * cfg.chunk_size)

    _info(f"Listening for {len(cfg.model_paths)} wake-word model(s) "
          f"@ threshold={cfg.threshold} (press Ctrl+C to stop)")
    start_time = time.monotonic()
    last_detection: dict[str, float] = {}

    with sd.InputStream(
        samplerate=_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=chunk_samples,
    ) as mic:
        try:
            while True:
                audio, _ = mic.read(chunk_samples)
                samples = (np.clip(audio.flatten(), -1.0, 1.0) * 32767).astype(np.int16)

                scores = model.predict(samples)
                for name, score in scores.items():
                    if score < cfg.threshold:
                        continue
                    now = time.monotonic()
                    if now - last_detection.get(name, 0.0) < cfg.debounce:
                        continue
                    last_detection[name] = now
                    elapsed = now - start_time
                    h, rem = divmod(int(elapsed), 3600)
                    m, s = divmod(rem, 60)
                    print(f"[{h:02d}:{m:02d}:{s:02d}] {name}: {score:.3f}")

        except KeyboardInterrupt:
            pass


def run_wav(model, cfg: WakeConfig) -> None:
    """Feed a WAV file through *model* and print wake-word hits."""
    sf = _require_soundfile()

    audio, file_sr = sf.read(cfg.wav, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio[:, 0]
    if len(audio) == 0:
        _info("No wake-word detected.")
        return
    if file_sr <= 0:
        _error(f"Invalid WAV sample rate: {file_sr}")

    # livekit-wakeword expects 16 kHz
    if file_sr != _SAMPLE_RATE:
        ratio = _SAMPLE_RATE / file_sr
        new_len = max(1, int(round(len(audio) * ratio)))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    chunk_samples = int(_SAMPLE_RATE * cfg.chunk_size)
    offset = 0
    hits: list[tuple[str, float, float]] = []  # (name, score, timestamp_sec)

    while offset < len(audio):
        segment = audio[offset : offset + chunk_samples]
        if len(segment) < chunk_samples:
            segment = np.pad(segment, (0, chunk_samples - len(segment)))
        segment_i16 = (np.clip(segment, -1.0, 1.0) * 32767).astype(np.int16)
        scores = model.predict(segment_i16)
        timestamp = offset / _SAMPLE_RATE
        for name, score in scores.items():
            if score >= cfg.threshold:
                hits.append((name, score, timestamp))
        offset += chunk_samples

    # Print all hits (group consecutive duplicates from the same chunk)
    last_emit: dict[str, float] = {}
    for name, score, ts in hits:
        if ts - last_emit.get(name, -cfg.debounce) < cfg.debounce:
            continue
        last_emit[name] = ts
        h, rem = divmod(int(ts), 3600)
        m, s = divmod(rem, 60)
        print(f"[{h:02d}:{m:02d}:{s:02d}] {name}: {score:.3f}")

    if not hits:
        _info("No wake-word detected.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wake-word detection with livekit-wakeword",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mic", action="store_true", help="Stream from microphone")
    mode.add_argument("--wav", metavar="PATH", help="Scan a WAV file")

    parser.add_argument(
        "--model", metavar="PATH", action="append", required=True,
        help="Path to a wake-word ONNX model (may be repeated for multiple models)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Detection threshold (0.0 - 1.0; higher = fewer false positives)",
    )
    parser.add_argument(
        "--debounce", type=float, default=2.0, metavar="SECS",
        help="Minimum seconds between detections of the same wake word",
    )
    parser.add_argument(
        "--chunk-size", type=float, default=2.0, metavar="SECS",
        help="Audio chunk duration per inference call (seconds)",
    )
    return parser.parse_args()


def main() -> None:
    _run_cli(_main_impl)


def _main_impl() -> None:
    args = parse_args()
    _validate_runtime_args(args)
    project_dir = Path(__file__).resolve().parent.parent

    model_paths = _validate_model_paths(args.model, project_dir)

    cfg = WakeConfig(
        model_paths=model_paths,
        threshold=args.threshold,
        debounce=args.debounce,
        chunk_size=args.chunk_size,
    )
    if args.wav:
        cfg.wav = args.wav  # type: ignore[attr-defined]

    model = _load_model(model_paths)

    if args.mic:
        run_mic(model, cfg)
    else:
        run_wav(model, cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
