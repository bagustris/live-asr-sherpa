import queue
from types import SimpleNamespace
from typing import Generator

import numpy as np

sd = SimpleNamespace(InputStream=None)
sf = SimpleNamespace(SoundFile=None)


def _require_soundfile():
    global sf
    if getattr(sf, "SoundFile", None) is not None:
        return sf
    try:
        import soundfile as _soundfile  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "soundfile is required for reading audio files. "
            "Install it with: pip install soundfile"
        ) from exc
    sf = _soundfile
    return sf


def _require_sounddevice():
    global sd
    if getattr(sd, "InputStream", None) is not None:
        return sd
    try:
        import sounddevice as _sounddevice  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "sounddevice is required for microphone input. "
            "Install it with: pip install sounddevice"
        ) from exc
    sd = _sounddevice
    return sd


def read_wav(
    path: str,
    target_sr: int = 16000,
    chunk_size: float = 0.16,
) -> Generator[np.ndarray, None, None]:
    """Read a mono audio file (WAV, FLAC, etc.) and yield float32 chunks."""
    sf = _require_soundfile()
    with sf.SoundFile(path) as f:
        if f.channels != 1:
            raise ValueError(
                f"Expected mono audio, got {f.channels} channels. "
                f"Re-sample with: ffmpeg -i <in> -ar {target_sr} -ac 1 out.wav"
            )
        if f.samplerate != target_sr:
            raise ValueError(
                f"Expected {target_sr} Hz audio, got {f.samplerate} Hz. "
                f"Re-sample with: ffmpeg -i <in> -ar {target_sr} -ac 1 out.wav"
            )
        chunk_frames = int(target_sr * chunk_size)
        while True:
            samples = f.read(chunk_frames, dtype="float32")
            if len(samples) == 0:
                break
            yield samples


def mic_stream(
    capture_rate: int = 16000,
    chunk_size: float = 0.1,
) -> Generator[np.ndarray, None, None]:
    """Capture microphone audio and yield float32 chunks via a queue.

    capture_rate: sample rate for the microphone (default 16 kHz).
    Use 48000 for better compatibility with system microphones that prefer
    48/44.1 kHz — sherpa-onnx resamples to the model rate internally.

    Uses a callback-based InputStream so audio capture never blocks the
    decoding loop — chunks are queued and consumed independently.
    """
    sd = _require_sounddevice()
    chunk_frames = int(capture_rate * chunk_size)
    q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(
        indata: np.ndarray, frames: int, time, status  # noqa: ANN001
    ) -> None:
        if status:
            print(f"[audio] {status}", flush=True)
        # indata shape: (frames, 1) — flatten to 1-D
        q.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=capture_rate,
        channels=1,
        dtype="float32",
        blocksize=chunk_frames,
        callback=_callback,
    ):
        while True:
            yield q.get()
