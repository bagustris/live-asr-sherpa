from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sherox.audio import mic_stream, read_wav


# ---------------------------------------------------------------------------
# read_wav
# ---------------------------------------------------------------------------

def _make_sf_mock(channels: int, samplerate: int, chunks: list):
    """Return a context-manager mock for sf.SoundFile."""
    mock_file = MagicMock()
    mock_file.__enter__ = MagicMock(return_value=mock_file)
    mock_file.__exit__ = MagicMock(return_value=False)
    mock_file.channels = channels
    mock_file.samplerate = samplerate
    mock_file.read.side_effect = chunks
    return mock_file


class TestReadWav:
    def test_yields_chunks(self):
        chunk1 = np.ones(2560, dtype="float32")
        chunk2 = np.ones(1000, dtype="float32") * 0.5
        mock_file = _make_sf_mock(1, 16000, [chunk1, chunk2, np.array([], dtype="float32")])

        with patch("sherox.audio.sf.SoundFile", return_value=mock_file):
            result = list(read_wav("dummy.wav"))

        assert len(result) == 2
        np.testing.assert_array_equal(result[0], chunk1)
        np.testing.assert_array_equal(result[1], chunk2)

    def test_yields_nothing_for_empty_file(self):
        mock_file = _make_sf_mock(1, 16000, [np.array([], dtype="float32")])

        with patch("sherox.audio.sf.SoundFile", return_value=mock_file):
            result = list(read_wav("empty.wav"))

        assert result == []

    def test_raises_on_multichannel(self):
        mock_file = _make_sf_mock(2, 16000, [])

        with patch("sherox.audio.sf.SoundFile", return_value=mock_file):
            with pytest.raises(ValueError, match="mono"):
                list(read_wav("stereo.wav"))

    def test_raises_on_wrong_sample_rate(self):
        mock_file = _make_sf_mock(1, 44100, [])

        with patch("sherox.audio.sf.SoundFile", return_value=mock_file):
            with pytest.raises(ValueError, match="44100"):
                list(read_wav("wrong_rate.wav"))

    def test_error_message_includes_resample_hint(self):
        mock_file = _make_sf_mock(1, 8000, [])

        with patch("sherox.audio.sf.SoundFile", return_value=mock_file):
            with pytest.raises(ValueError, match="ffmpeg"):
                list(read_wav("low_rate.wav"))

    def test_chunk_frames_derived_from_chunk_size(self):
        """chunk_frames = int(target_sr * chunk_size) must be passed to read()."""
        mock_file = _make_sf_mock(1, 16000, [np.array([], dtype="float32")])

        with patch("sherox.audio.sf.SoundFile", return_value=mock_file):
            list(read_wav("dummy.wav", target_sr=16000, chunk_size=0.5))

        mock_file.read.assert_called_with(8000, dtype="float32")

    def test_custom_target_sr_and_chunk_size(self):
        chunk = np.zeros(9600, dtype="float32")  # 48000 * 0.2
        mock_file = _make_sf_mock(1, 48000, [chunk, np.array([], dtype="float32")])

        with patch("sherox.audio.sf.SoundFile", return_value=mock_file):
            result = list(read_wav("dummy.wav", target_sr=48000, chunk_size=0.2))

        assert len(result) == 1
        mock_file.read.assert_called_with(9600, dtype="float32")

    def test_chunks_are_float32(self):
        chunk = np.zeros(2560, dtype="float32")
        mock_file = _make_sf_mock(1, 16000, [chunk, np.array([], dtype="float32")])

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
