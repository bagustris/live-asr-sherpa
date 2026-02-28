import queue
import wave
from typing import Generator

import numpy as np
import sounddevice as sd


def read_wav(
    path: str,
    target_sr: int = 16000,
    chunk_size: float = 0.16,
) -> Generator[np.ndarray, None, None]:
    """Read a mono 16-bit WAV file and yield float32 chunks."""
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(
                f"Expected mono WAV, got {wf.getnchannels()} channels. "
                f"Re-sample with: ffmpeg -i <in> -ar {target_sr} -ac 1 out.wav"
            )
        if wf.getframerate() != target_sr:
            raise ValueError(
                f"Expected {target_sr} Hz WAV, got {wf.getframerate()} Hz. "
                f"Re-sample with: ffmpeg -i <in> -ar {target_sr} -ac 1 out.wav"
            )
        chunk_frames = int(target_sr * chunk_size)
        while True:
            raw = wf.readframes(chunk_frames)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
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
