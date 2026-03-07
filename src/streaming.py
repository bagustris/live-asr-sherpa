import sys
from typing import Generator

import numpy as np
import sherpa_onnx


def run_streaming(
    recognizer: sherpa_onnx.OnlineRecognizer,
    audio_gen: Generator[np.ndarray, None, None],
    sample_rate: int = 16000,
) -> None:
    """Feed incremental audio chunks into the recognizer and render output.

    Display strategy:
      - Partial hypotheses: overwrite the current terminal line with \\r
        (near-zero latency feedback, avoids line spam).
      - Finalized segments: printed on a new line when an endpoint is detected.

    Tradeoff: endpoint detection adds ~1–2 s silence at segment boundaries
    but keeps each printed line readable. Disable via enable_endpoint_detection
    in asr_engine.py for continuous single-line output instead.
    """
    stream = recognizer.create_stream()
    last_partial = ""

    try:
        for chunk in audio_gen:
            stream.accept_waveform(sample_rate, chunk)

            # Decode all queued frames immediately — avoid internal buffer build-up
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            text = recognizer.get_result(stream).strip()

            if recognizer.is_endpoint(stream):
                # Segment finalized: clear partial line and print the full segment
                if text:
                    _clear_line(last_partial)
                    print(f"{_PREFIX}{text}")
                    sys.stdout.flush()
                recognizer.reset(stream)
                last_partial = ""
            elif text != last_partial:
                # Show latest partial hypothesis in-place
                sys.stdout.write(f"\r{_PREFIX}{text}")
                sys.stdout.flush()
                last_partial = text

    except KeyboardInterrupt:
        pass
    finally:
        _flush_tail(recognizer, stream, sample_rate, last_partial)


def run_offline_vad_streaming(
    recognizer: sherpa_onnx.OfflineRecognizer,
    vad: sherpa_onnx.VoiceActivityDetector,
    audio_gen: Generator[np.ndarray, None, None],
    sample_rate: int = 16000,
) -> None:
    """VAD-segmented offline ASR for live audio (microphone or WAV file).

    Strategy: feed audio chunks into the VAD; when the VAD marks a speech
    segment as complete (silence detected after speech), run the offline
    recognizer on the accumulated samples and print the result.

    Latency tradeoff: adds ~0.5 s silence at segment boundaries (VAD
    min_silence_duration) but gives higher accuracy than streaming models.
    """
    was_speech = False

    try:
        for chunk in audio_gen:
            vad.accept_waveform(chunk)

            # Show a live indicator while speech is being captured
            now_speech = vad.is_speech_detected()
            if now_speech and not was_speech:
                sys.stdout.write(f"\r{_PREFIX}[listening…]")
                sys.stdout.flush()
            was_speech = now_speech

            # Process completed speech segments
            while not vad.empty():
                segment = vad.front
                samples = np.array(segment.samples, dtype=np.float32)
                vad.pop()
                _decode_and_print(recognizer, samples, sample_rate)

    except KeyboardInterrupt:
        pass
    finally:
        # Flush any speech still buffered in the VAD
        vad.flush()
        while not vad.empty():
            segment = vad.front
            samples = np.array(segment.samples, dtype=np.float32)
            vad.pop()
            _decode_and_print(recognizer, samples, sample_rate)
        sys.stdout.write("\n")
        sys.stdout.flush()


def _decode_and_print(
    recognizer: sherpa_onnx.OfflineRecognizer,
    samples: np.ndarray,
    sample_rate: int,
) -> None:
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    text = stream.result.text.strip()
    if text:
        sys.stdout.write(f"\r{' ' * 20}\r")
        print(f"{_PREFIX}{text}")
        sys.stdout.flush()


_PREFIX = "  "


def _clear_line(previous: str) -> None:
    sys.stdout.write(f"\r{' ' * max(len(_PREFIX) + len(previous), 1)}\r")
    sys.stdout.flush()


def _flush_tail(
    recognizer: sherpa_onnx.OnlineRecognizer,
    stream: sherpa_onnx.OnlineStream,
    sample_rate: int,
    last_partial: str,
) -> None:
    """Flush any audio left in the recognizer pipeline after the loop ends."""
    # Feed a short silence tail to trigger decoding of the final frames
    tail = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail)
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    text = recognizer.get_result(stream).strip()
    if text:
        _clear_line(last_partial)
        print(f"{_PREFIX}{text}")
    sys.stdout.write("\n")
    sys.stdout.flush()
