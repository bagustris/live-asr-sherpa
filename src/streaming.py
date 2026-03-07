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
