import queue
from typing import Generator

import numpy as np
import sounddevice as sd
import soundfile as sf


def read_wav(
    path: str,
    target_sr: int = 16000,
    chunk_size: float = 0.16,
) -> Generator[np.ndarray, None, None]:
    """Read a mono audio file (WAV, FLAC, etc.) and yield float32 chunks."""
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
    sample_rate: int = 16000,
    chunk_size: float = 0.16,
) -> Generator[np.ndarray, None, None]:
    """Capture microphone audio and yield float32 chunks via a queue.

    Uses a callback-based InputStream so audio capture never blocks the
    decoding loop — chunks are queued and consumed independently.
    """
    chunk_frames = int(sample_rate * chunk_size)
    q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(
        indata: np.ndarray, frames: int, time, status  # noqa: ANN001
    ) -> None:
        if status:
            print(f"[audio] {status}", flush=True)
        # indata shape: (frames, 1) — flatten to 1-D
        q.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=chunk_frames,
        callback=_callback,
    ):
        while True:
            yield q.get()
