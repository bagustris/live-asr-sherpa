from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))

import benchmark_utils  # noqa: E402


class _FakeStream:
    def __init__(self):
        self.audio = None
        self.result = SimpleNamespace(text="")

    def accept_waveform(self, sample_rate, audio):
        self.audio = np.asarray(audio)


class _FakeRecognizer:
    def __init__(self):
        self.streams = []

    def create_stream(self):
        stream = _FakeStream()
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream):
        stream.result.text = f"chunk{len(self.streams)}"


def test_transcribe_offline_keeps_non_whisper_single_pass():
    recognizer = _FakeRecognizer()
    audio = np.zeros(64000, dtype=np.float32)

    text = benchmark_utils.transcribe_offline(recognizer, audio, 16000, model_type="nemo_ctc")

    assert text == "chunk1"
    assert len(recognizer.streams) == 1


def test_transcribe_offline_chunks_long_whisper_audio():
    recognizer = _FakeRecognizer()
    max_samples = int(16000 * benchmark_utils._WHISPER_MAX_OFFLINE_SECONDS)
    audio = np.zeros(max_samples * 2 + 123, dtype=np.float32)

    text = benchmark_utils.transcribe_offline(recognizer, audio, 16000, model_type="whisper")

    assert text == "chunk1 chunk2 chunk3"
    assert len(recognizer.streams) == 3
    assert all(len(stream.audio) <= max_samples for stream in recognizer.streams)
