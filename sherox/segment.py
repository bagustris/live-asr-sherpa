"""VAD-based audio segmentation — entry point.

Usage:
    # Segment microphone input (Silero VAD, default):
    sherox.segment --mic

    # Segment a WAV file:
    sherox.segment --wav path/to/audio.wav

    # Use Ten-VAD instead of Silero:
    sherox.segment --mic --vad-model ten-vad

    # Save each detected speech segment to a directory:
    sherox.segment --wav audio.wav --output-dir segments/

    # Show live microphone level:
    sherox.segment --mic --listening

Output format (stdout):
    [00:00.120 – 00:02.560]  (speech detected)
    [00:05.040 – 00:08.210]
    ...
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console

from .asr_engine import build_vad
from .audio import mic_stream, read_wav
from .config import SegmentConfig

_console = Console()
_err_console = Console(stderr=True)

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


def _info(msg: str) -> None:
    _console.print(f"[bold green]\\[info][/bold green] {msg}")


def _error(msg: str) -> None:
    _err_console.print(f"[bold red]\\[error][/bold red] {msg}")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VAD-based audio segmentation with Sherpa-ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mic", action="store_true", help="Segment microphone input")
    mode.add_argument("--wav", metavar="PATH", help="Segment a WAV file")

    parser.add_argument(
        "--vad-model",
        dest="vad_type",
        default="silero",
        choices=["silero", "ten-vad"],
        help="VAD model to use for segmentation (default: silero)",
    )
    parser.add_argument(
        "--ten-vad-model",
        default="ten-vad.int8.onnx",
        choices=["ten-vad.onnx", "ten-vad.int8.onnx"],
        help="Ten-VAD model variant (default: ten-vad.int8.onnx)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        metavar="F",
        help="VAD speech probability threshold (0.0–1.0)",
    )
    parser.add_argument(
        "--min-silence",
        type=float,
        default=0.5,
        metavar="SEC",
        help="Minimum silence duration to end a segment (seconds)",
    )
    parser.add_argument(
        "--min-speech",
        type=float,
        default=0.25,
        metavar="SEC",
        help="Minimum speech duration to start a segment (seconds)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        metavar="HZ",
        help="Audio sample rate expected by the VAD model",
    )
    parser.add_argument(
        "--capture-rate",
        type=int,
        default=16000,
        metavar="HZ",
        help="Microphone capture rate (Hz). Use 48000 for better device compatibility.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU thread count for ONNX runtime",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        metavar="DIR",
        help="Save each detected segment as a WAV file into this directory",
    )
    parser.add_argument(
        "--listening",
        action="store_true",
        help="Show a live RMS energy bar for mic level calibration",
    )
    return parser.parse_args()


def _download_file(url: str, dest: Path) -> None:
    _info(f"Downloading from:\n  {url}")
    _info("This may take a moment…")

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


def _resolve_vad(cfg: SegmentConfig, project_dir: Path) -> str:
    """Return the path to the VAD model file, downloading it if necessary."""
    models_dir = project_dir / "models"
    if cfg.vad_type == "ten-vad":
        if cfg.ten_vad_model not in _TEN_VAD_MODEL_URLS:
            _error(
                f"Unknown --ten-vad-model '{cfg.ten_vad_model}'. "
                f"Supported: {', '.join(_TEN_VAD_MODEL_URLS)}."
            )
        vad_path = models_dir / cfg.ten_vad_model
        if not vad_path.exists():
            vad_path.parent.mkdir(parents=True, exist_ok=True)
            _info(f"VAD model not found, downloading {cfg.ten_vad_model}…")
            _download_file(_TEN_VAD_MODEL_URLS[cfg.ten_vad_model], vad_path)
        return str(vad_path)
    # silero (default)
    vad_path = models_dir / "silero_vad.onnx"
    if not vad_path.exists():
        vad_path.parent.mkdir(parents=True, exist_ok=True)
        _info("VAD model not found, downloading silero_vad.onnx…")
        _download_file(_VAD_URL, vad_path)
    return str(vad_path)


def _fmt_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m:02d}:{s:06.3f}"


def run_segment(
    vad,
    audio_gen,
    cfg: SegmentConfig,
    sample_rate: int,
    output_dir: Path | None = None,
    segment_counter: list | None = None,
) -> None:
    """Consume *audio_gen* chunks, detect speech segments, and print timestamps.

    Optionally saves each segment as a WAV file into *output_dir*.
    *segment_counter* is a mutable list used to persist the segment index
    across calls (pass ``[0]`` to start from zero).
    """
    if segment_counter is None:
        segment_counter = [0]

    elapsed_samples = 0
    prefix = "  "

    try:
        for chunk in audio_gen:
            vad.accept_waveform(chunk)
            elapsed_samples += len(chunk)

            if cfg.show_mic_level:
                energy = float(np.sqrt(np.mean(chunk ** 2)))
                bar = "█" * min(int(energy * 500), 40)
                sys.stdout.write(f"\r{prefix}mic: {bar:<40} {energy:.4f}")
                sys.stdout.flush()

            while not vad.empty():
                seg = vad.front
                samples = np.array(seg.samples, dtype=np.float32)
                start_sec = seg.start / sample_rate
                end_sec = (seg.start + len(samples)) / sample_rate
                vad.pop()

                label = (
                    f"{prefix}[{_fmt_time(start_sec)} – {_fmt_time(end_sec)}]"
                    f"  ({end_sec - start_sec:.2f}s)"
                )
                sys.stdout.write(f"\r{' ' * 20}\r")
                sys.stdout.flush()
                _console.print(f"[bold cyan]{label}[/bold cyan]")

                if output_dir is not None:
                    idx = segment_counter[0]
                    out_path = output_dir / f"segment_{idx:04d}.wav"
                    sf.write(str(out_path), samples, samplerate=sample_rate)
                    segment_counter[0] += 1

    except KeyboardInterrupt:
        pass
    finally:
        # Flush remaining buffered audio
        vad.flush()
        while not vad.empty():
            seg = vad.front
            samples = np.array(seg.samples, dtype=np.float32)
            start_sec = seg.start / sample_rate
            end_sec = (seg.start + len(samples)) / sample_rate
            vad.pop()

            label = (
                f"{prefix}[{_fmt_time(start_sec)} – {_fmt_time(end_sec)}]"
                f"  ({end_sec - start_sec:.2f}s)"
            )
            _console.print(f"[bold cyan]{label}[/bold cyan]")

            if output_dir is not None:
                idx = segment_counter[0]
                out_path = output_dir / f"segment_{idx:04d}.wav"
                sf.write(str(out_path), samples, samplerate=sample_rate)
                segment_counter[0] += 1

        sys.stdout.write("\n")
        sys.stdout.flush()


def main() -> None:
    args = parse_args()

    project_dir = Path(__file__).resolve().parent.parent

    cfg = SegmentConfig(
        vad_type=args.vad_type,
        ten_vad_model=args.ten_vad_model,
        vad_threshold=args.threshold,
        vad_min_silence_duration=args.min_silence,
        vad_min_speech_duration=args.min_speech,
        sample_rate=args.sample_rate,
        capture_rate=args.capture_rate,
        num_threads=args.threads,
        show_mic_level=args.listening,
        output_dir=args.output_dir,
    )

    cfg.vad_model = _resolve_vad(cfg, project_dir)

    output_dir: Path | None = None
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.wav:
        if not Path(args.wav).exists():
            _error(f"Audio file not found: {args.wav}")
        audio = read_wav(args.wav, target_sr=cfg.sample_rate, chunk_size=cfg.chunk_size)
        capture_rate = cfg.sample_rate
        _info(f"Segmenting: {args.wav}\n")
    else:
        audio = mic_stream(capture_rate=cfg.capture_rate, chunk_size=cfg.chunk_size)
        capture_rate = cfg.capture_rate
        _info("Listening on microphone — press Ctrl+C to stop.\n")

    # Build VAD configured for the actual input sample rate (wav uses sample_rate,
    # mic uses capture_rate — VadModelConfig.sample_rate must match the audio fed in).
    from .config import Config as AsrConfig  # noqa: PLC0415
    vad_cfg = AsrConfig(
        vad_type=cfg.vad_type,
        vad_model=cfg.vad_model,
        ten_vad_model=cfg.ten_vad_model,
        vad_threshold=cfg.vad_threshold,
        vad_min_silence_duration=cfg.vad_min_silence_duration,
        vad_min_speech_duration=cfg.vad_min_speech_duration,
        sample_rate=capture_rate,
        num_threads=cfg.num_threads,
    )
    vad = build_vad(vad_cfg)

    run_segment(vad, audio, cfg, sample_rate=capture_rate, output_dir=output_dir)


if __name__ == "__main__":
    main()
