"""Speaker Identification — entry point.

Usage:
    # Identify speaker from microphone (VAD-segmented real-time):
    sherox.sid --mic --speaker-file speakers.txt

    # Identify speaker from a WAV file:
    sherox.sid --wav audio.wav --speaker-file speakers.txt

    # Custom model and threshold:
    sherox.sid --mic --speaker-file speakers.txt --threshold 0.75

    # Enroll a new speaker (append entries to the speaker file):
    sherox.sid --enroll alice alice1.wav alice2.wav --speaker-file speakers.txt

    # Create a new speaker file from scratch by enrolling:
    sherox.sid --enroll bob ref.wav --speaker-file new_speakers.txt

    # Enroll a new speaker directly from microphone:
    sherox.sid --enroll-mic alice

    # --speaker-file defaults to speakers.txt (omitted above):

Speaker file format (one 'name /absolute/path/wav' per line):
    alice /path/to/alice1.wav
    alice /path/to/alice2.wav
    bob   /path/to/bob1.wav

Multiple entries for the same name are averaged into a single embedding.
Enroll writes absolute paths so the file works regardless of working directory.

Default model: models/nemo_en_titanet_large.onnx (~96 MB, auto-downloaded).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rich.console import Console

from . import ConfigError
from .asr_engine import _require_sherpa_onnx, build_vad
from .utils import download_file as _download_file
from .utils import render_mic_level as _render_mic_level
from .utils import run_cli as _run_cli
from .audio import mic_stream
from .config import Config as _AsrConfig, SidConfig

_console = Console()
_err_console = Console(stderr=True)

_PREFIX = "  "


def _require_soundfile():
    try:
        import soundfile as sf  # noqa: PLC0415
        return sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "soundfile is required for audio I/O. "
            "Install it with: pip install soundfile"
        ) from exc


def _make_sid_vad_cfg(
    vad_model: str,
    vad_threshold: float,
    vad_min_silence_duration: float,
    vad_min_speech_duration: float,
    sample_rate: int,
    num_threads: int = 4,
) -> _AsrConfig:
    return _AsrConfig(
        vad_model=vad_model,
        vad_type="silero",
        vad_threshold=vad_threshold,
        vad_min_silence_duration=vad_min_silence_duration,
        vad_min_speech_duration=vad_min_speech_duration,
        sample_rate=sample_rate,
        num_threads=num_threads,
    )


_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/nemo_en_titanet_large.onnx"
)
_MODEL_FILE = "nemo_en_titanet_large.onnx"

_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/silero_vad.onnx"
)

_PALETTE = [
    "bright_cyan", "bright_magenta", "bright_yellow",
    "bright_green", "bright_blue", "bright_red",
    "cyan", "magenta", "yellow", "green",
]


def _info(msg: str) -> None:
    _console.print(f"[bold green]\\[info][/bold green] {msg}")


def _error(msg: str) -> None:
    _err_console.print(f"[bold red]\\[error][/bold red] {msg}")
    raise ConfigError(msg)


def _validate_model(model_path: str, project_dir: Path) -> str:
    p = Path(model_path)
    if not p.is_absolute():
        p = project_dir / p
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        _info(f"Model not found. Downloading {_MODEL_FILE}…")
        _download_file(_MODEL_URL, p)
    return str(p)


def _validate_vad(project_dir: Path) -> str:
    vad_path = project_dir / "models" / "silero_vad.onnx"
    if not vad_path.exists():
        vad_path.parent.mkdir(parents=True, exist_ok=True)
        _info("VAD model not found. Downloading silero_vad.onnx…")
        _download_file(_VAD_URL, vad_path)
    return str(vad_path)


def _load_speaker_file(path: str) -> Dict[str, List[str]]:
    p = Path(path)
    if not p.is_file():
        msg = f"Speaker file not found: {path}"
        if Path(path).name == "speakers.txt":
            msg += (
                "\n\nNo speaker file found at the default location. "
                "You can:\n"
                "  • Enroll a new speaker with: --enroll-mic NAME\n"
                "  • Specify an existing file with: --speaker-file PATH"
            )
        _error(msg)
    speakers: Dict[str, List[str]] = defaultdict(list)
    with open(p) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                _error(f"Speaker file line {i}: expected 'name /path/to/wav', got: {line!r}")
            name, wav_path = parts
            if not Path(wav_path).is_file():
                _error(f"Speaker file line {i}: WAV not found: {wav_path}")
            speakers[name].append(wav_path)
    if not speakers:
        _error(f"Speaker file {path!r} is empty.")
    return dict(speakers)


def enroll_speaker(name: str, wav_paths: List[str], speaker_file: str) -> None:
    """Append *name* / *wav* entries to *speaker_file*, creating it if needed.

    Each WAV path is stored as an absolute path so the file is portable
    regardless of the caller's working directory.

    Duplicate ``name + absolute_path`` pairs are skipped with a warning so
    re-running enroll never double-counts the same recording.

    Example — add two recordings for *alice* to speakers.txt::

        enroll_speaker("alice", ["alice1.wav", "alice2.wav"], "speakers.txt")

    The file will contain::

        alice /abs/path/alice1.wav
        alice /abs/path/alice2.wav
    """
    speaker_path = Path(speaker_file)

    # Read existing entries to detect duplicates.
    existing: set[tuple[str, str]] = set()
    if speaker_path.is_file():
        with open(speaker_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    existing.add((parts[0], parts[1]))

    speaker_path.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with open(speaker_path, "a") as f:
        for wav in wav_paths:
            abs_wav = str(Path(wav).resolve())
            if not Path(abs_wav).is_file():
                _error(f"WAV file not found: {wav}")
            key = (name, abs_wav)
            if key in existing:
                _info(f"Skipping duplicate: {name} {abs_wav}")
                continue
            f.write(f"{name} {abs_wav}\n")
            existing.add(key)
            added += 1

    _info(
        f"Enrolled {added} recording(s) for '{name}' in '{speaker_path}'. "
        f"(Total entries in file: {len(existing)})"
    )


def enroll_speaker_mic(
    name: str,
    speaker_file: str,
    vad_model: str = "",
    capture_rate: int = 16000,
    chunk_size: float = 0.1,
    show_mic_level: bool = True,
    vad_threshold: float = 0.3,
    vad_min_silence_duration: float = 1.0,
    vad_min_speech_duration: float = 1.0,
) -> None:
    """Enroll a speaker by recording from the microphone.

    Captures audio via :func:`.mic_stream`, uses Silero VAD to segment
    speech into utterances, saves each segment as a numbered WAV file
    alongside *speaker_file*, then delegates to :func:`enroll_speaker`.

    Press Ctrl+C to stop recording — any segments captured so far are
    saved and enrolled.
    """
    sf = _require_soundfile()

    speaker_path = Path(speaker_file)
    wav_dir = speaker_path.parent

    vad = build_vad(_make_sid_vad_cfg(
        vad_model, vad_threshold, vad_min_silence_duration,
        vad_min_speech_duration, capture_rate,
    ))

    _console.print(f"\n[bold yellow]Enrolling '{name}' from microphone.[/bold yellow]")
    _console.print("Press [bold]Ctrl+C[/bold] when done speaking.\n")

    audio = mic_stream(capture_rate=capture_rate, chunk_size=chunk_size)
    segments: list[np.ndarray] = []

    try:
        for chunk in audio:
            vad.accept_waveform(chunk)

            if show_mic_level:
                _render_mic_level(chunk)

            while not vad.empty():
                seg = vad.front
                samples = np.array(seg.samples, dtype=np.float32)
                vad.pop()
                segments.append(samples)
                sys.stdout.write(f"\r{_PREFIX}Captured {len(segments)} segment(s)...")
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        vad.flush()
        while not vad.empty():
            seg = vad.front
            samples = np.array(seg.samples, dtype=np.float32)
            vad.pop()
            segments.append(samples)
        print()

    if not segments:
        _error("No speech detected. Enrollment cancelled.")

    def _next_wav() -> Path:
        n = 1
        while True:
            p = wav_dir / f"{name}_mic_enroll_{n:03d}.wav"
            if not p.exists():
                return p
            n += 1

    wav_paths: list[str] = []
    try:
        for samples in segments:
            wav_path = _next_wav()
            sf.write(str(wav_path), samples, samplerate=capture_rate)
            wav_paths.append(str(wav_path))
    except Exception:
        for p in wav_paths:
            Path(p).unlink(missing_ok=True)
        raise

    _info(f"Saved {len(wav_paths)} recording(s) to '{wav_dir}'")
    enroll_speaker(name, wav_paths, speaker_file)


def _load_wav_flat(path: str) -> Tuple[np.ndarray, int]:
    """Load an audio file and return (float32 mono samples, sample_rate)."""
    sf = _require_soundfile()
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return np.ascontiguousarray(data[:, 0]), sr


def _build_extractor(cfg: SidConfig):
    sherpa_onnx = _require_sherpa_onnx()
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=cfg.model,
        num_threads=cfg.num_threads,
        debug=False,
        provider="cpu",
    )
    if not config.validate():
        _error(f"Invalid embedding extractor config. Check model path: {cfg.model}")
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


def _build_manager(extractor, speakers: Dict[str, List[str]]):
    sherpa_onnx = _require_sherpa_onnx()
    manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)
    for name, wavs in speakers.items():
        _info(f"Registering '{name}' ({len(wavs)} file(s))…")
        acc = None
        for wav in wavs:
            samples, sr = _load_wav_flat(wav)
            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate=sr, waveform=samples)
            stream.input_finished()
            emb = np.array(extractor.compute(stream))
            acc = emb if acc is None else acc + emb
        if not manager.add(name, acc / len(wavs)):
            _error(f"Failed to register speaker: {name}")
    return manager


def _identify(extractor, manager, samples: np.ndarray, sample_rate: int, threshold: float) -> str:
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
    stream.input_finished()
    name = manager.search(np.array(extractor.compute(stream)), threshold=threshold)
    return name if name else "unknown"


def _colour_for(name: str, colour_map: Dict[str, str], next_idx: List[int]) -> str:
    if name == "unknown":
        return "yellow"
    if name not in colour_map:
        colour_map[name] = _PALETTE[next_idx[0] % len(_PALETTE)]
        next_idx[0] += 1
    return colour_map[name]


def run_wav(cfg: SidConfig, speakers: Dict[str, List[str]]) -> None:
    extractor = _build_extractor(cfg)
    manager = _build_manager(extractor, speakers)

    _info(f"Processing: {cfg.wav}\n")
    samples, sr = _load_wav_flat(cfg.wav)
    name = _identify(extractor, manager, samples, sr, cfg.threshold)
    colour = "bright_cyan" if name != "unknown" else "yellow"
    _console.print(f"{_PREFIX}[bold {colour}]{name}[/bold {colour}]")


def run_mic(cfg: SidConfig, speakers: Dict[str, List[str]]) -> None:
    extractor = _build_extractor(cfg)
    manager = _build_manager(extractor, speakers)

    vad = build_vad(_make_sid_vad_cfg(
        cfg.vad_model, cfg.vad_threshold, cfg.vad_min_silence_duration,
        cfg.vad_min_speech_duration, cfg.capture_rate, cfg.num_threads,
    ))

    colour_map: Dict[str, str] = {}
    next_idx: List[int] = [0]

    def _process(samples: np.ndarray) -> None:
        name = _identify(extractor, manager, samples, cfg.capture_rate, cfg.threshold)
        c = _colour_for(name, colour_map, next_idx)
        sys.stdout.write(f"\r{' ' * 44}\r")
        sys.stdout.flush()
        _console.print(f"{_PREFIX}[bold {c}]{name}[/bold {c}]")

    audio = mic_stream(capture_rate=cfg.capture_rate, chunk_size=cfg.chunk_size)
    _info("Listening on microphone — press Ctrl+C to stop.\n")

    try:
        for chunk in audio:
            vad.accept_waveform(chunk)

            if cfg.show_mic_level:
                _render_mic_level(chunk)

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
        description="Speaker Identification with Sherpa-ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mic", action="store_true", help="Stream from microphone")
    mode.add_argument("--wav", metavar="PATH", help="Identify speaker in a WAV file")
    mode.add_argument(
        "--enroll",
        nargs="+",
        metavar=("NAME", "WAV"),
        help=(
            "Enroll a new speaker. Provide the speaker name followed by one or more "
            "WAV files: --enroll alice ref1.wav ref2.wav. "
            "Entries are appended to --speaker-file (created if absent). "
            "Absolute paths are stored so the file works from any directory. "
            "Duplicate name+path pairs are silently skipped."
        ),
    )

    mode.add_argument(
        "--enroll-mic",
        metavar="NAME",
        help=(
            "Enroll a new speaker by recording from the microphone. "
            "Provide the speaker name: --enroll-mic alice. "
            "Press Ctrl+C when done speaking. "
            "Recordings are saved alongside --speaker-file (default: speakers.txt)."
        ),
    )

    parser.add_argument(
        "--speaker-file",
        default="speakers.txt",
        metavar="PATH",
        help="Text file with 'name /path/to/ref.wav' entries (one per line)",
    )
    parser.add_argument(
        "--model",
        default=f"models/{_MODEL_FILE}",
        metavar="PATH",
        help="Speaker embedding ONNX model (auto-downloaded if absent)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Cosine similarity threshold for a match (0–1, higher = stricter)",
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
        "--no-mic-level",
        action="store_true",
        default=False,
        help="Suppress the live RMS energy bar during microphone capture",
    )
    return parser.parse_args()


def main() -> None:
    _run_cli(_main_impl)


def _main_impl() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent.parent

    # --enroll is handled before any model loading — it only writes to the
    # speaker file and needs no ONNX runtime, VAD, or audio hardware.
    if args.enroll:
        if len(args.enroll) < 2:
            _error("--enroll requires a NAME followed by at least one WAV file.")
        name, *wavs = args.enroll
        enroll_speaker(name, wavs, args.speaker_file)
        return

    if args.enroll_mic:
        vad_model = _validate_vad(project_dir)
        enroll_speaker_mic(
            args.enroll_mic,
            args.speaker_file,
            vad_model=vad_model,
            capture_rate=args.capture_rate,
            chunk_size=args.chunk_size,
            show_mic_level=not args.no_mic_level,
            vad_threshold=0.3,
            vad_min_silence_duration=1.0,
            vad_min_speech_duration=1.0,
        )
        return

    model_path = _validate_model(args.model, project_dir)

    vad_model = ""
    if args.mic:
        vad_model = _validate_vad(project_dir)

    cfg = SidConfig(
        model=model_path,
        threshold=args.threshold,
        sample_rate=args.sample_rate,
        capture_rate=args.capture_rate,
        chunk_size=args.chunk_size,
        num_threads=args.threads,
        vad_model=vad_model,
        wav=args.wav or "",
        show_mic_level=not args.no_mic_level,
    )

    speakers = _load_speaker_file(args.speaker_file)
    _info(f"Loaded {len(speakers)} speaker(s): {', '.join(sorted(speakers))}")
    _info("Loading embedding model…")

    if args.wav:
        run_wav(cfg, speakers)
    else:
        run_mic(cfg, speakers)


if __name__ == "__main__":  # pragma: no cover
    main()
