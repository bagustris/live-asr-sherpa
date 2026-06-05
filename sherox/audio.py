import logging
import queue
from math import gcd
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


def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-quality polyphase resample from orig_sr to target_sr.

    Uses scipy.signal.resample_poly with a Kaiser window (β=14, ~90 dB stopband
    attenuation).  This is preferred over scipy.signal.resample (FFT-based) for
    audio because FFT resampling assumes a periodic signal and produces
    wrap-around ringing artifacts at the edges; polyphase FIR filtering is
    non-circular and avoids those artifacts entirely.

    Falls back to linear interpolation if scipy is somehow unavailable.
    """
    if orig_sr == target_sr:
        return data
    try:
        from scipy.signal import resample_poly  # noqa: PLC0415
        g = gcd(target_sr, orig_sr)
        return resample_poly(
            data,
            target_sr // g,
            orig_sr // g,
            window=("kaiser", 14.0),  # ~90 dB stopband; default β=5 gives only ~60 dB
            padtype="line",           # linear edge extrapolation reduces end-of-signal transients
        ).astype(np.float32)
    except ImportError:  # pragma: no cover - scipy is a core dep
        n_orig = len(data)
        n_new = int(n_orig * target_sr / orig_sr)
        return np.interp(
            np.linspace(0, n_orig - 1, n_new),
            np.arange(n_orig),
            data,
        ).astype(np.float32)


def read_wav(
    path: str,
    target_sr: int = 16000,
    chunk_size: float = 0.16,
) -> Generator[np.ndarray, None, None]:
    """Read a mono audio file (WAV, FLAC, etc.) and yield float32 chunks.

    When the file's sample rate matches *target_sr*, audio is streamed
    directly in chunks without loading the full file into RAM.  When
    resampling is required, the file is loaded once and resampled with
    scipy.signal.resample_poly before chunking, avoiding boundary
    artifacts that block-wise resampling would introduce.
    """
    sf = _require_soundfile()
    with sf.SoundFile(path) as f:
        if f.channels != 1:
            raise ValueError(
                f"Expected mono audio, got {f.channels} channels. "
                f"Convert with: ffmpeg -i <in> -ar {target_sr} -ac 1 out.wav"
            )
        orig_sr = f.samplerate
        chunk_frames = int(target_sr * chunk_size)

        if orig_sr == target_sr:
            # Stream directly: no resampling, no full-file load.
            while True:
                block = f.read(frames=chunk_frames, dtype="float32")
                if len(block) == 0:
                    break
                yield block
        else:
            # Rates differ: load full file, resample once, then chunk.
            data = f.read(dtype="float32")

    if orig_sr != target_sr:
        data = _resample(data, orig_sr, target_sr)
        offset = 0
        while offset < len(data):
            yield data[offset : offset + chunk_frames]
            offset += chunk_frames


def wav_duration(path: str) -> float:
    """Return the duration of a WAV/audio file in seconds."""
    sf = _require_soundfile()
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate


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

    PortAudio status messages (overflows, underflows) are logged as
    warnings rather than printed to stdout.
    """
    sd = _require_sounddevice()
    chunk_frames = int(capture_rate * chunk_size)
    q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(
        indata: np.ndarray, frames: int, time, status  # noqa: ANN001
    ) -> None:
        if status:
            logging.warning("[audio] %s", status)
        # indata shape: (frames, 1) — flatten to 1-D
        q.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=capture_rate,
        channels=1,
        dtype="float32",
        blocksize=chunk_frames,
        callback=_callback,
    )
    stream.start()
    try:
        while True:
            yield q.get()
    finally:
        stream.stop()
        stream.close()


def pipe_stream(
    capture_rate: int = 16000,
    chunk_size: float = 0.16,
) -> Generator[np.ndarray, None, None]:
    """Read raw 16-bit little-endian mono PCM from stdin and yield float32 chunks.

    Stops cleanly when stdin reaches EOF so the ASR loop terminates naturally.

    Typical usage::

        arecord -f S16_LE -r 16000 -c 1 | sherox.asr --pipe
        ffmpeg -i audio.mp4 -f s16le -ar 16000 -ac 1 - | sherox.asr --pipe
    """
    import sys  # noqa: PLC0415
    chunk_frames = int(capture_rate * chunk_size)
    bytes_per_chunk = chunk_frames * 2  # int16 = 2 bytes per sample
    stdin_buf = sys.stdin.buffer
    while True:
        data = stdin_buf.read(bytes_per_chunk)
        if not data:
            break
        if len(data) < bytes_per_chunk:
            # Pad the last incomplete chunk so array shape is consistent.
            data = data + b"\x00" * (bytes_per_chunk - len(data))
        yield np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def denoise_gen(
    audio_gen: Generator[np.ndarray, None, None],
    sample_rate: int,
    min_duration: float = 1.0,
) -> Generator[np.ndarray, None, None]:
    """Wrap an audio generator with noisereduce denoising.

    Audio is accumulated for at least *min_duration* seconds before
    denoising, which gives noisereduce enough context to estimate the
    noise profile.  The denoised audio is then re-chunked at the
    original chunk size and yielded.

    Only suitable for offline (WAV file) processing where the added
    latency is acceptable.
    """
    try:
        import noisereduce as nr  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "noisereduce is required for --denoise. "
            "Install it with: pip install 'sherox[denoise]'"
        ) from exc

    min_samples = int(sample_rate * min_duration)
    buf: list[np.ndarray] = []
    buf_len = 0

    def _flush(buffer: list[np.ndarray], chunk_size: int) -> Generator[np.ndarray, None, None]:
        combined = np.concatenate(buffer)
        denoised = nr.reduce_noise(y=combined, sr=sample_rate).astype(np.float32)
        offset = 0
        while offset < len(denoised):
            yield denoised[offset : offset + chunk_size]
            offset += chunk_size

    chunk_size = 0
    for chunk in audio_gen:
        chunk_size = len(chunk)
        buf.append(chunk)
        buf_len += chunk_size
        if buf_len >= min_samples:
            yield from _flush(buf, chunk_size)
            buf = []
            buf_len = 0

    if buf:
        yield from _flush(buf, chunk_size or 1)
