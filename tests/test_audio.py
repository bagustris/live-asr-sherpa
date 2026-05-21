from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sherox.audio import mic_stream, pipe_stream, read_wav


# ---------------------------------------------------------------------------
# read_wav
# ---------------------------------------------------------------------------

def _make_sf_mock(channels: int, samplerate: int, data: np.ndarray):
    """Return a context-manager mock for sf.SoundFile that simulates streaming reads.

    Calls with ``frames=N`` return the next *N* samples (streaming path).
    Calls without ``frames`` return all remaining data (full-load path used when resampling).
    """
    mock_file = MagicMock()
    mock_file.__enter__ = MagicMock(return_value=mock_file)
    mock_file.__exit__ = MagicMock(return_value=False)
    mock_file.channels = channels
    mock_file.samplerate = samplerate

    pos = [0]

    def _read(*args, **kwargs):
        n = kwargs.get("frames", None)
        if n is None:
            chunk = data[pos[0]:]
            pos[0] = len(data)
        else:
            chunk = data[pos[0] : pos[0] + n]
            pos[0] += n
        return chunk

    mock_file.read.side_effect = _read
    return mock_file


class TestReadWav:
    def test_yields_chunks(self):
        data = np.concatenate([
            np.ones(2560, dtype="float32"),
            np.ones(1000, dtype="float32") * 0.5,
        ])
        mock_file = _make_sf_mock(1, 16000, data)

        with patch("sherox.audio.sf.SoundFile", return_value=mock_file):
            result = list(read_wav("dummy.wav", chunk_size=0.16))  # chunk_frames=2560

        assert len(result) == 2
        np.testing.assert_array_equal(result[0], np.ones(2560, dtype="float32"))
        np.testing.assert_array_equal(result[1], np.ones(1000, dtype="float32") * 0.5)

    def test_yields_nothing_for_empty_file(self):
        mock_file = _make_sf_mock(1, 16000, np.array([], dtype="float32"))

        with patch("sherox.audio.sf.SoundFile", return_value=mock_file):
            result = list(read_wav("empty.wav"))

        assert result == []

    def test_raises_on_multichannel(self):
        mock_file = _make_sf_mock(2, 16000, np.array([], dtype="float32"))

        with patch("sherox.audio.sf.SoundFile", return_value=mock_file):
            with pytest.raises(ValueError, match="mono"):
                list(read_wav("stereo.wav"))

    def test_resamples_wrong_sample_rate(self):
        # 44100 → 16000: roughly 36% of original length
        orig = np.ones(44100, dtype="float32")
        mock_file = _make_sf_mock(1, 44100, orig)

        with patch("sherox.audio.sf.SoundFile", return_value=mock_file):
            result = list(read_wav("wrong_rate.wav", target_sr=16000))

        total_samples = sum(len(c) for c in result)
        assert total_samples == int(44100 * 16000 / 44100)

    def test_chunk_frames_derived_from_chunk_size(self):
        """Output chunks must be at most chunk_frames = int(target_sr * chunk_size)."""
        data = np.zeros(8000, dtype="float32")  # exactly one chunk at chunk_size=0.5
        mock_file = _make_sf_mock(1, 16000, data)

        with patch("sherox.audio.sf.SoundFile", return_value=mock_file):
            result = list(read_wav("dummy.wav", target_sr=16000, chunk_size=0.5))

        assert len(result) == 1
        assert len(result[0]) == 8000

    def test_custom_target_sr_and_chunk_size(self):
        data = np.zeros(9600, dtype="float32")  # 48000 * 0.2
        mock_file = _make_sf_mock(1, 48000, data)

        with patch("sherox.audio.sf.SoundFile", return_value=mock_file):
            result = list(read_wav("dummy.wav", target_sr=48000, chunk_size=0.2))

        assert len(result) == 1
        assert len(result[0]) == 9600

    def test_chunks_are_float32(self):
        chunk = np.zeros(2560, dtype="float32")
        mock_file = _make_sf_mock(1, 16000, chunk)

        with patch("sherox.audio.sf.SoundFile", return_value=mock_file):
            result = list(read_wav("dummy.wav"))

        assert result[0].dtype == np.float32


# ---------------------------------------------------------------------------
# mic_stream
# ---------------------------------------------------------------------------

class TestMicStream:
    def test_yields_audio_from_callback(self):
        """Verify that audio put into the queue by the callback flows out."""
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)

        expected = np.ones(1600, dtype="float32")

        def fake_input_stream(**kwargs):
            # Call the callback synchronously so q already has data when q.get() runs
            indata = expected.reshape(-1, 1)
            kwargs["callback"](indata, 1600, None, None)
            return mock_stream_ctx

        with patch("sherox.audio.sd.InputStream", side_effect=fake_input_stream):
            result = next(mic_stream(capture_rate=16000, chunk_size=0.1))

        np.testing.assert_array_equal(result, expected)

    def test_input_stream_created_with_correct_params(self):
        """InputStream must receive the right samplerate, channels, and blocksize."""
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)
        captured_kwargs = {}

        def fake_input_stream(**kwargs):
            captured_kwargs.update(kwargs)
            # Immediately put something in the queue via the callback so next() returns
            indata = np.zeros((9600, 1), dtype="float32")
            kwargs["callback"](indata, 9600, None, None)
            return mock_stream_ctx

        with patch("sherox.audio.sd.InputStream", side_effect=fake_input_stream):
            gen = mic_stream(capture_rate=48000, chunk_size=0.2)
            next(gen)

        assert captured_kwargs["samplerate"] == 48000
        assert captured_kwargs["channels"] == 1
        assert captured_kwargs["dtype"] == "float32"
        assert captured_kwargs["blocksize"] == 9600  # int(48000 * 0.2)

    def test_callback_flattens_2d_input(self):
        """Callback receives (frames, 1) array and must yield 1-D chunks."""
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)

        def fake_input_stream(**kwargs):
            indata = np.ones((1600, 1), dtype="float32") * 0.7
            kwargs["callback"](indata, 1600, None, None)
            return mock_stream_ctx

        with patch("sherox.audio.sd.InputStream", side_effect=fake_input_stream):
            chunk = next(mic_stream())

        assert chunk.ndim == 1
        assert len(chunk) == 1600
        assert np.allclose(chunk, 0.7)


# ---------------------------------------------------------------------------
# _require_soundfile — success path
# ---------------------------------------------------------------------------

class TestRequireSoundfileAudio:
    def test_imports_when_sentinel_is_none(self):
        import types
        import sherox.audio as audio_module
        fake_sf = MagicMock()
        fake_sf.SoundFile = MagicMock()
        initial = types.SimpleNamespace(SoundFile=None)
        with patch.object(audio_module, "sf", initial):
            with patch.dict("sys.modules", {"soundfile": fake_sf}):
                result = audio_module._require_soundfile()
        assert result is fake_sf


# ---------------------------------------------------------------------------
# _require_sounddevice — success path
# ---------------------------------------------------------------------------

class TestRequireSounddeviceAudio:
    def test_imports_when_sentinel_is_none(self):
        import types
        import sherox.audio as audio_module
        fake_sd = MagicMock()
        fake_sd.InputStream = MagicMock()
        initial = types.SimpleNamespace(InputStream=None)
        with patch.object(audio_module, "sd", initial):
            with patch.dict("sys.modules", {"sounddevice": fake_sd}):
                result = audio_module._require_sounddevice()
        assert result is fake_sd


# ---------------------------------------------------------------------------
# mic_stream — status callback
# ---------------------------------------------------------------------------

class TestMicStreamStatus:
    def test_status_printed_when_nonzero(self, caplog):
        import logging
        import sherox.audio as audio_module
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)

        callback_ref = {}

        def fake_input_stream(**kwargs):
            callback_ref["cb"] = kwargs["callback"]
            # Put something so next() returns
            indata = MagicMock()
            indata.__getitem__ = MagicMock(return_value=MagicMock(copy=MagicMock(return_value=MagicMock())))
            kwargs["callback"](
                __import__("numpy").ones((1600, 1), dtype="float32"),
                1600, None, None
            )
            return mock_stream_ctx

        with patch("sherox.audio.sd.InputStream", side_effect=fake_input_stream):
            next(audio_module.mic_stream())

        # Now call the callback again with a status
        import numpy as np
        indata = np.ones((1600, 1), dtype="float32")
        with caplog.at_level(logging.WARNING):
            callback_ref["cb"](indata, 1600, None, "input overflow")
        assert "input overflow" in caplog.text


# ---------------------------------------------------------------------------
# pipe_stream
# ---------------------------------------------------------------------------

class TestPipeStream:
    def _make_stdin(self, arrays: list) -> MagicMock:
        """Build a fake sys.stdin.buffer that returns byte sequences then b""."""
        chunks = [a.tobytes() for a in arrays] + [b""]
        mock_buf = MagicMock()
        mock_buf.read.side_effect = chunks
        return mock_buf

    def test_yields_float32_chunks(self):
        """Bytes from stdin are converted to float32 in [-1, 1]."""
        data = np.array([16384, -16384], dtype=np.int16)
        mock_buf = self._make_stdin([data])
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.buffer = mock_buf
            result = list(pipe_stream(capture_rate=16000, chunk_size=len(data) / 16000))
        assert len(result) == 1
        assert result[0].dtype == np.float32
        np.testing.assert_allclose(result[0], np.array([0.5, -0.5], dtype=np.float32), atol=1e-5)

    def test_stops_on_eof(self):
        """Empty read from stdin must end the generator."""
        mock_buf = MagicMock()
        mock_buf.read.return_value = b""
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.buffer = mock_buf
            result = list(pipe_stream(capture_rate=16000, chunk_size=0.1))
        assert result == []

    def test_pads_incomplete_last_chunk(self):
        """If the last chunk is shorter than bytes_per_chunk, it gets zero-padded."""
        chunk_frames = int(16000 * 0.1)  # 1600
        short_data = np.ones(800, dtype=np.int16)  # half a chunk
        mock_buf = self._make_stdin([short_data])
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.buffer = mock_buf
            result = list(pipe_stream(capture_rate=16000, chunk_size=0.1))
        assert len(result) == 1
        assert len(result[0]) == chunk_frames
        # first 800 samples from data, last 800 are zeros
        np.testing.assert_allclose(result[0][800:], np.zeros(800, dtype=np.float32))

    def test_multiple_chunks(self):
        """Multiple stdin reads each become a separate yielded chunk."""
        chunk_frames = int(16000 * 0.1)
        chunk1 = np.zeros(chunk_frames, dtype=np.int16)
        chunk2 = np.ones(chunk_frames, dtype=np.int16) * 1000
        mock_buf = self._make_stdin([chunk1, chunk2])
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.buffer = mock_buf
            result = list(pipe_stream(capture_rate=16000, chunk_size=0.1))
        assert len(result) == 2
        assert result[0].dtype == np.float32
        assert result[1].dtype == np.float32
