import sys
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Generator, Optional

import numpy as np
import sherpa_onnx
from rich.console import Console
from rich.text import Text
import shutil

# ── Rich console (no markup auto-escaping needed; we build Text objects) ────
_console = Console(highlight=False, markup=False)

# Colours cycled through for speakers 0, 1, 2, …
_SPEAKER_COLOURS = [
    "cyan",
    "magenta",
    "yellow",
    "green",
    "blue",
    "red",
    "bright_cyan",
    "bright_magenta",
    "bright_yellow",
    "bright_green",
]

_PREFIX = "  "


def _speaker_colour(speaker_id: int) -> str:
    return _SPEAKER_COLOURS[speaker_id % len(_SPEAKER_COLOURS)]


def _rich_print(text: str, speaker_id: Optional[int] = None) -> None:
    """Print a finalised line, optionally coloured by speaker."""
    if speaker_id is not None:
        label = f"[Speaker {speaker_id:02d}] "
        colour = _speaker_colour(speaker_id)
        t = Text()
        t.append(f"{_PREFIX}{label}", style=f"bold {colour}")
        t.append(text)
        _console.print(t)
    else:
        _console.print(f"{_PREFIX}{text}")


def _dominant_speaker(result: sherpa_onnx.OfflineSpeakerDiarizationResult) -> int:
    """Return the speaker id that covers the most time in this segment."""
    segments = result.sort_by_start_time()
    if not segments:
        return 0
    duration: dict[int, float] = {}
    for seg in segments:
        duration[seg.speaker] = duration.get(seg.speaker, 0.0) + (seg.end - seg.start)
    return max(duration, key=duration.__getitem__)


# ── Online (streaming) recogniser loop ──────────────────────────────────────

def run_streaming(
    recognizer: sherpa_onnx.OnlineRecognizer,
    audio_gen: Generator[np.ndarray, None, None],
    sample_rate: int = 16000,
    show_mic_level: bool = False,
    diarization: Optional[sherpa_onnx.OfflineSpeakerDiarization] = None,
) -> None:
    """Feed incremental audio chunks into the recognizer and render output.

    Display strategy:
      - Partial hypotheses: overwrite the current terminal line with \\r
        (near-zero latency feedback, avoids line spam).
      - Finalized segments: printed on a new line when an endpoint is detected.

    When *diarization* is provided the accumulated audio for each utterance is
    sent to the diarization pipeline in a background thread so that it runs
    concurrently with the next ASR utterance, keeping added latency near zero.
    """
    stream = recognizer.create_stream()
    last_partial = ""
    # Buffer raw audio for the current utterance (used for diarization).
    audio_buf: list[np.ndarray] = []
    executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1) if diarization is not None else None
    pending: Optional[Future] = None  # diarization future for the *previous* utterance

    def _submit_diarization(samples: np.ndarray) -> Optional[Future]:
        if executor is None or diarization is None:
            return None
        return executor.submit(diarization.process, samples)

    def _flush_pending(pending_future: Optional[Future], pending_text: str) -> None:
        """Print the pending utterance with its diarization label (if available)."""
        if not pending_text:
            return
        speaker_id = None
        if pending_future is not None:
            try:
                result = pending_future.result(timeout=5.0)
                speaker_id = _dominant_speaker(result)
            except Exception:
                pass
        _rich_print(pending_text, speaker_id)

    pending_text = ""

    try:
        for chunk in audio_gen:
            stream.accept_waveform(sample_rate, chunk)
            if diarization is not None:
                audio_buf.append(chunk)

            # Decode all queued frames immediately
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            text = recognizer.get_result(stream).strip()

            if show_mic_level and not text:
                energy = float(np.sqrt(np.mean(chunk ** 2)))
                bar = "█" * min(int(energy * 500), 40)
                sys.stdout.write(f"\r{_PREFIX}mic: {bar:<40} {energy:.4f}")
                sys.stdout.flush()

            if recognizer.is_endpoint(stream):
                if text:
                    if show_mic_level:
                        try:
                            width = shutil.get_terminal_size(fallback=(80, 20)).columns
                        except OSError:
                            width = 80
                        sys.stdout.write("\r" + " " * width + "\r")
                        sys.stdout.flush()
                    _clear_line(last_partial)
                    # Flush the previous utterance (diarization may now be done).
                    _flush_pending(pending, pending_text)
                    # Submit diarization for this utterance.
                    if diarization is not None and audio_buf:
                        seg_audio = np.concatenate(audio_buf)
                        pending = _submit_diarization(seg_audio)
                        pending_text = text
                    else:
                        pending = None
                        pending_text = text
                        _flush_pending(None, pending_text)
                        pending_text = ""
                recognizer.reset(stream)
                audio_buf.clear()
                last_partial = ""
            elif text != last_partial:
                sys.stdout.write(f"\r{_PREFIX}{text}")
                sys.stdout.flush()
                last_partial = text

    except KeyboardInterrupt:
        pass
    finally:
        _flush_tail(recognizer, stream, sample_rate, last_partial)
        # Flush the last pending diarization result.
        _flush_pending(pending, pending_text)
        if executor is not None:
            executor.shutdown(wait=True)


# ── Offline VAD-segmented loop ───────────────────────────────────────────────

def run_offline_vad_streaming(
    recognizer: sherpa_onnx.OfflineRecognizer,
    vad: sherpa_onnx.VoiceActivityDetector,
    audio_gen: Generator[np.ndarray, None, None],
    sample_rate: int = 48000,
    show_mic_level: bool = False,
    diarization: Optional[sherpa_onnx.OfflineSpeakerDiarization] = None,
) -> None:
    """VAD-segmented offline ASR with optional concurrent speaker diarization.

    For each speech segment the ASR and diarization models run concurrently
    in a ThreadPoolExecutor so that neither doubles the per-segment latency.
    """
    executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=2) if diarization else None

    try:
        for chunk in audio_gen:
            vad.accept_waveform(chunk)

            if show_mic_level:
                energy = float(np.sqrt(np.mean(chunk ** 2)))
                bar = "█" * min(int(energy * 500), 40)
                sys.stdout.write(f"\r{_PREFIX}mic: {bar:<40} {energy:.4f}")
                sys.stdout.flush()

            while not vad.empty():
                segment = vad.front
                samples = np.array(segment.samples, dtype=np.float32)
                vad.pop()
                _decode_and_print(recognizer, samples, sample_rate, diarization, executor)

    except KeyboardInterrupt:
        pass
    finally:
        vad.flush()
        while not vad.empty():
            segment = vad.front
            samples = np.array(segment.samples, dtype=np.float32)
            vad.pop()
            _decode_and_print(recognizer, samples, sample_rate, diarization, executor)
        sys.stdout.write("\n")
        sys.stdout.flush()
        if executor is not None:
            executor.shutdown(wait=True)


def _decode_and_print(
    recognizer: sherpa_onnx.OfflineRecognizer,
    samples: np.ndarray,
    sample_rate: int,
    diarization: Optional[sherpa_onnx.OfflineSpeakerDiarization] = None,
    executor: Optional[ThreadPoolExecutor] = None,
) -> None:
    """Run ASR (and optionally diarization) on *samples* and print the result.

    ASR and diarization are submitted concurrently to *executor* so that the
    combined latency is max(asr_time, diarization_time) rather than the sum.
    """
    if diarization is not None and executor is not None:
        asr_future = executor.submit(_run_asr, recognizer, samples, sample_rate)
        diar_future = executor.submit(diarization.process, samples)
        text = asr_future.result()
        diar_result = diar_future.result()
        speaker_id: Optional[int] = _dominant_speaker(diar_result) if text else None
    else:
        text = _run_asr(recognizer, samples, sample_rate)
        speaker_id = None

    if text:
        sys.stdout.write(f"\r{' ' * 20}\r")
        sys.stdout.flush()
        _rich_print(text, speaker_id)


def _run_asr(
    recognizer: sherpa_onnx.OfflineRecognizer,
    samples: np.ndarray,
    sample_rate: int,
) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def _clear_line(partial: str) -> None:
    """Overwrite the partial hypothesis line with spaces to prevent leftover text."""
    if partial:
        width = len(_PREFIX) + len(partial)
        sys.stdout.write(f"\r{' ' * width}\r")
        sys.stdout.flush()


def _flush_tail(
    recognizer: sherpa_onnx.OnlineRecognizer,
    stream: sherpa_onnx.OnlineStream,
    sample_rate: int,
    last_partial: str,
) -> None:
    """Flush any audio left in the recognizer pipeline after the loop ends."""
    tail = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail)
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    text = recognizer.get_result(stream).strip()
    if text:
        _clear_line(last_partial)
        _console.print(f"{_PREFIX}{text}")
    sys.stdout.write("\n")
    sys.stdout.flush()
