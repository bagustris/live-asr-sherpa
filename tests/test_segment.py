import argparse
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sherox.segment as seg_module
from sherox.config import SegmentConfig


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_mic_mode(self):
        with patch("sys.argv", ["sherox.segment", "--mic"]):
            args = seg_module.parse_args()
        assert args.mic is True
        assert args.wav is None

    def test_wav_mode(self):
        with patch("sys.argv", ["sherox.segment", "--wav", "audio.wav"]):
            args = seg_module.parse_args()
        assert args.wav == "audio.wav"
        assert args.mic is False

    def test_mutually_exclusive(self):
        with patch("sys.argv", ["sherox.segment", "--mic", "--wav", "x.wav"]):
            with pytest.raises(SystemExit):
                seg_module.parse_args()

    def test_requires_mic_or_wav(self):
        with patch("sys.argv", ["sherox.segment"]):
            with pytest.raises(SystemExit):
                seg_module.parse_args()

    def test_defaults(self):
        with patch("sys.argv", ["sherox.segment", "--mic"]):
            args = seg_module.parse_args()
        assert args.vad_type == "silero"
        assert args.threshold == 0.5
        assert args.min_silence == 0.5
        assert args.min_speech == 0.25
        assert args.sample_rate == 16000
        assert args.capture_rate == 16000
        assert args.threads == 4
        assert args.output_dir == ""
        assert args.listening is False

    def test_ten_vad_mode(self):
        with patch("sys.argv", ["sherox.segment", "--mic", "--vad-model", "ten-vad"]):
            args = seg_module.parse_args()
        assert args.vad_type == "ten-vad"

    def test_ten_vad_model_variant(self):
        with patch("sys.argv", ["sherox.segment", "--mic", "--ten-vad-model", "ten-vad.onnx"]):
            args = seg_module.parse_args()
        assert args.ten_vad_model == "ten-vad.onnx"

    def test_custom_threshold(self):
        with patch("sys.argv", ["sherox.segment", "--mic", "--threshold", "0.7"]):
            args = seg_module.parse_args()
        assert args.threshold == 0.7

    def test_output_dir(self):
        with patch("sys.argv", ["sherox.segment", "--mic", "--output-dir", "/tmp/segs"]):
            args = seg_module.parse_args()
        assert args.output_dir == "/tmp/segs"

    def test_listening_flag(self):
        with patch("sys.argv", ["sherox.segment", "--mic", "--listening"]):
            args = seg_module.parse_args()
        assert args.listening is True

    def test_invalid_vad_type_exits(self):
        with patch("sys.argv", ["sherox.segment", "--mic", "--vad-model", "invalid"]):
            with pytest.raises(SystemExit):
                seg_module.parse_args()

    def test_custom_sample_rate(self):
        with patch("sys.argv", ["sherox.segment", "--mic", "--sample-rate", "48000"]):
            args = seg_module.parse_args()
        assert args.sample_rate == 48000

    def test_custom_capture_rate(self):
        with patch("sys.argv", ["sherox.segment", "--mic", "--capture-rate", "48000"]):
            args = seg_module.parse_args()
        assert args.capture_rate == 48000

    def test_custom_threads(self):
        with patch("sys.argv", ["sherox.segment", "--mic", "--threads", "8"]):
            args = seg_module.parse_args()
        assert args.threads == 8


# ---------------------------------------------------------------------------
# _validate_runtime_args
# ---------------------------------------------------------------------------

class TestValidateRuntimeArgs:
    def _args(self, **kwargs):
        defaults = dict(
            threshold=0.5, min_silence=0.5, min_speech=0.25,
            sample_rate=16000, capture_rate=16000, threads=4,
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_valid_args_pass(self):
        seg_module._validate_runtime_args(self._args())  # no exception

    def test_threshold_below_0_exits(self):
        with pytest.raises(SystemExit):
            seg_module._validate_runtime_args(self._args(threshold=-0.1))

    def test_threshold_above_1_exits(self):
        with pytest.raises(SystemExit):
            seg_module._validate_runtime_args(self._args(threshold=1.1))

    def test_threshold_at_boundaries_passes(self):
        seg_module._validate_runtime_args(self._args(threshold=0.0))
        seg_module._validate_runtime_args(self._args(threshold=1.0))

    def test_negative_min_silence_exits(self):
        with pytest.raises(SystemExit):
            seg_module._validate_runtime_args(self._args(min_silence=-0.1))

    def test_negative_min_speech_exits(self):
        with pytest.raises(SystemExit):
            seg_module._validate_runtime_args(self._args(min_speech=-0.1))

    def test_zero_sample_rate_exits(self):
        with pytest.raises(SystemExit):
            seg_module._validate_runtime_args(self._args(sample_rate=0))

    def test_zero_capture_rate_exits(self):
        with pytest.raises(SystemExit):
            seg_module._validate_runtime_args(self._args(capture_rate=0))

    def test_zero_threads_exits(self):
        with pytest.raises(SystemExit):
            seg_module._validate_runtime_args(self._args(threads=0))


# ---------------------------------------------------------------------------
# _download_file
# ---------------------------------------------------------------------------

class TestDownloadFile:
    def test_success(self, tmp_path):
        dest = tmp_path / "file.bin"
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "100"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b""]

        with patch("sherox.utils.urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            seg_module._download_file("http://example.com/file.bin", dest)

        mock_urlopen.assert_called_once()

    def test_failure_exits(self, tmp_path):
        dest = tmp_path / "file.bin"
        with patch("sherox.utils.urllib.request.urlopen", side_effect=Exception("network error")):
            with pytest.raises(SystemExit):
                seg_module._download_file("http://example.com/file.bin", dest)

    def test_progress_callback_with_positive_total(self, tmp_path):
        dest = tmp_path / "file.bin"
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "2048"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b"x" * 1024, b""]

        with patch("sherox.utils.urllib.request.urlopen", return_value=mock_response):
            seg_module._download_file("http://example.com/file.bin", dest)

        assert dest.exists()

    def test_progress_skipped_when_total_zero(self, tmp_path):
        dest = tmp_path / "file.bin"
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "0"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b""]

        with patch("sherox.utils.urllib.request.urlopen", return_value=mock_response):
            seg_module._download_file("http://example.com/file.bin", dest)


# ---------------------------------------------------------------------------
# _resolve_vad
# ---------------------------------------------------------------------------

class TestResolveVad:
    def test_downloads_silero_when_missing(self, tmp_path):
        cfg = SegmentConfig(vad_type="silero")
        with patch.object(seg_module, "_download_file") as mock_dl:
            result = seg_module._resolve_vad(cfg, tmp_path)
        assert result == str(tmp_path / "models" / "silero_vad.onnx")
        mock_dl.assert_called_once()

    def test_returns_existing_silero(self, tmp_path):
        vad_path = tmp_path / "models" / "silero_vad.onnx"
        vad_path.parent.mkdir()
        vad_path.touch()
        cfg = SegmentConfig(vad_type="silero")
        with patch.object(seg_module, "_download_file") as mock_dl:
            result = seg_module._resolve_vad(cfg, tmp_path)
        mock_dl.assert_not_called()
        assert result == str(vad_path)

    def test_downloads_ten_vad_when_missing(self, tmp_path):
        cfg = SegmentConfig(vad_type="ten-vad", ten_vad_model="ten-vad.int8.onnx")
        with patch.object(seg_module, "_download_file") as mock_dl:
            result = seg_module._resolve_vad(cfg, tmp_path)
        assert "ten-vad.int8.onnx" in result
        mock_dl.assert_called_once()

    def test_returns_existing_ten_vad(self, tmp_path):
        vad_path = tmp_path / "models" / "ten-vad.int8.onnx"
        vad_path.parent.mkdir()
        vad_path.touch()
        cfg = SegmentConfig(vad_type="ten-vad", ten_vad_model="ten-vad.int8.onnx")
        with patch.object(seg_module, "_download_file") as mock_dl:
            result = seg_module._resolve_vad(cfg, tmp_path)
        mock_dl.assert_not_called()
        assert result == str(vad_path)

    def test_unknown_ten_vad_model_exits(self, tmp_path):
        cfg = SegmentConfig(vad_type="ten-vad", ten_vad_model="unknown.onnx")
        with pytest.raises(SystemExit):
            seg_module._resolve_vad(cfg, tmp_path)


# ---------------------------------------------------------------------------
# _fmt_time
# ---------------------------------------------------------------------------

class TestFmtTime:
    def test_zero(self):
        assert seg_module._fmt_time(0.0) == "00:00.000"

    def test_sub_minute(self):
        assert seg_module._fmt_time(2.56) == "00:02.560"

    def test_over_minute(self):
        result = seg_module._fmt_time(65.5)
        assert result.startswith("01:")

    def test_exact_minute(self):
        assert seg_module._fmt_time(60.0) == "01:00.000"


# ---------------------------------------------------------------------------
# run_segment
# ---------------------------------------------------------------------------

def _make_seg(samples, start=0):
    seg = MagicMock()
    seg.samples = np.array(samples, dtype=np.float32).tolist()
    seg.start = start
    return seg


class TestRunSegment:
    def test_processes_speech_segment(self, capsys):
        vad = MagicMock()
        seg = _make_seg(np.ones(8000), start=0)
        vad.front = seg
        vad.empty.side_effect = [False, True, True]
        cfg = SegmentConfig()
        seg_module.run_segment(vad, iter([np.zeros(1600, dtype="float32")]), cfg, 16000)
        assert "00:" in capsys.readouterr().out

    def test_vad_accept_waveform_called_per_chunk(self):
        vad = MagicMock()
        vad.empty.return_value = True
        cfg = SegmentConfig()
        chunks = [np.zeros(1600, dtype="float32")] * 3
        seg_module.run_segment(vad, iter(chunks), cfg, 16000)
        assert vad.accept_waveform.call_count == 3

    def test_saves_wav_when_output_dir(self, tmp_path):
        vad = MagicMock()
        seg = _make_seg(np.ones(8000), start=0)
        vad.front = seg
        vad.empty.side_effect = [False, True, True]
        cfg = SegmentConfig()
        mock_sf = MagicMock()
        counter = [0]
        with patch.object(seg_module, "_require_soundfile", return_value=mock_sf):
            seg_module.run_segment(
                vad, iter([np.zeros(1600, dtype="float32")]),
                cfg, 16000, output_dir=tmp_path, segment_counter=counter,
            )
        mock_sf.write.assert_called_once()
        written_path = mock_sf.write.call_args[0][0]
        assert "segment_0000.wav" in written_path
        assert counter[0] == 1

    def test_handles_keyboard_interrupt(self):
        vad = MagicMock()
        vad.empty.return_value = True
        cfg = SegmentConfig()
        def interrupt_gen():
            yield np.zeros(1600, dtype="float32")
            raise KeyboardInterrupt
        seg_module.run_segment(vad, interrupt_gen(), cfg, 16000)  # must not propagate

    def test_flush_called_in_finally(self):
        vad = MagicMock()
        vad.empty.return_value = True
        cfg = SegmentConfig()
        seg_module.run_segment(vad, iter([]), cfg, 16000)
        vad.flush.assert_called_once()

    def test_flushes_remaining_segments_in_finally(self, capsys):
        vad = MagicMock()
        seg = _make_seg(np.ones(8000), start=16000)
        vad.front = seg
        vad.empty.side_effect = [False, True]  # only reached in finally
        cfg = SegmentConfig()
        seg_module.run_segment(vad, iter([]), cfg, 16000)
        assert ":" in capsys.readouterr().out

    def test_saves_wav_in_finally_with_output_dir(self, tmp_path):
        vad = MagicMock()
        seg = _make_seg(np.ones(8000), start=0)
        vad.front = seg
        vad.empty.side_effect = [False, True]  # only in finally
        cfg = SegmentConfig()
        mock_sf = MagicMock()
        with patch.object(seg_module, "_require_soundfile", return_value=mock_sf):
            seg_module.run_segment(
                vad, iter([]), cfg, 16000,
                output_dir=tmp_path, segment_counter=[0],
            )
        mock_sf.write.assert_called_once()

    def test_mic_level_bar_shown(self, capsys):
        vad = MagicMock()
        vad.empty.return_value = True
        cfg = SegmentConfig(show_mic_level=True)
        seg_module.run_segment(
            vad, iter([np.ones(1600, dtype="float32") * 0.1]), cfg, 16000
        )
        assert "mic:" in capsys.readouterr().out

    def test_default_segment_counter_starts_at_zero(self, tmp_path):
        vad = MagicMock()
        seg = _make_seg(np.ones(8000), start=0)
        vad.front = seg
        vad.empty.side_effect = [False, True, True]
        cfg = SegmentConfig()
        mock_sf = MagicMock()
        with patch.object(seg_module, "_require_soundfile", return_value=mock_sf):
            seg_module.run_segment(
                vad, iter([np.zeros(1600, dtype="float32")]),
                cfg, 16000, output_dir=tmp_path,
            )
        written_path = mock_sf.write.call_args[0][0]
        assert "segment_0000.wav" in written_path

    def test_no_output_when_no_output_dir(self, tmp_path):
        vad = MagicMock()
        seg = _make_seg(np.ones(8000), start=0)
        vad.front = seg
        vad.empty.side_effect = [False, True, True]
        cfg = SegmentConfig()
        mock_sf = MagicMock()
        with patch.object(seg_module, "_require_soundfile", return_value=mock_sf):
            seg_module.run_segment(
                vad, iter([np.zeros(1600, dtype="float32")]),
                cfg, 16000, output_dir=None,
            )
        mock_sf.write.assert_not_called()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def _make_vad(self):
        vad = MagicMock()
        vad.empty.return_value = True
        return vad

    def test_main_with_wav(self, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.touch()
        with patch("sys.argv", ["sherox.segment", "--wav", str(wav)]), \
             patch.object(seg_module, "_resolve_vad", return_value="silero_vad.onnx"), \
             patch("sherox.segment.build_vad", return_value=self._make_vad()), \
             patch("sherox.segment.read_wav", return_value=iter([])), \
             patch.object(seg_module, "run_segment"):
            seg_module.main()

    def test_main_with_mic(self):
        with patch("sys.argv", ["sherox.segment", "--mic"]), \
             patch.object(seg_module, "_resolve_vad", return_value="silero_vad.onnx"), \
             patch("sherox.segment.build_vad", return_value=self._make_vad()), \
             patch("sherox.segment.mic_stream", return_value=iter([])), \
             patch.object(seg_module, "run_segment"):
            seg_module.main()

    def test_main_exits_if_wav_not_found(self, tmp_path):
        with patch("sys.argv", ["sherox.segment", "--wav", str(tmp_path / "missing.wav")]), \
             patch.object(seg_module, "_resolve_vad", return_value="silero_vad.onnx"), \
             pytest.raises(SystemExit):
            seg_module.main()

    def test_main_with_output_dir(self, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.touch()
        out_dir = str(tmp_path / "segs")
        with patch("sys.argv", ["sherox.segment", "--wav", str(wav), "--output-dir", out_dir]), \
             patch.object(seg_module, "_resolve_vad", return_value="silero_vad.onnx"), \
             patch("sherox.segment.build_vad", return_value=self._make_vad()), \
             patch("sherox.segment.read_wav", return_value=iter([])), \
             patch.object(seg_module, "run_segment"):
            seg_module.main()
        assert Path(out_dir).is_dir()


# ---------------------------------------------------------------------------
# _require_soundfile — success path
# ---------------------------------------------------------------------------

class TestRequireSoundfileSegment:
    def test_imports_when_sentinel_is_none(self):
        import types
        import sherox.segment as segment_module
        fake_sf = MagicMock()
        fake_sf.write = MagicMock()
        initial = types.SimpleNamespace(write=None)
        with patch.object(segment_module, "sf", initial):
            with patch.dict("sys.modules", {"soundfile": fake_sf}):
                result = segment_module._require_soundfile()
        assert result is fake_sf

    def test_returns_early_when_already_loaded(self):
        import sherox.segment as segment_module
        fake_sf = MagicMock()
        fake_sf.write = MagicMock()
        with patch.object(segment_module, "sf", fake_sf):
            result = segment_module._require_soundfile()
        assert result is fake_sf
