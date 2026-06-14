"""Tests for sherox.wake — wake-word detection module."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sherox.wake as wake_module
from sherox import SherpaError
from sherox.config import WakeConfig


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_mic_mode(self):
        with patch("sys.argv", ["sherox.wake", "--mic", "--model", "m.onnx"]):
            args = wake_module.parse_args()
        assert args.mic is True
        assert args.wav is None

    def test_wav_mode(self):
        with patch("sys.argv", ["sherox.wake", "--wav", "audio.wav", "--model", "m.onnx"]):
            args = wake_module.parse_args()
        assert args.wav == "audio.wav"
        assert args.mic is False

    def test_mic_and_wav_mutually_exclusive(self):
        with patch("sys.argv", [
            "sherox.wake", "--mic", "--wav", "a.wav", "--model", "m.onnx"
        ]):
            with pytest.raises(SystemExit):
                wake_module.parse_args()

    def test_requires_mic_or_wav(self):
        with patch("sys.argv", ["sherox.wake", "--model", "m.onnx"]):
            with pytest.raises(SystemExit):
                wake_module.parse_args()

    def test_model_argument(self):
        with patch("sys.argv", ["sherox.wake", "--mic", "--model", "m.onnx"]):
            args = wake_module.parse_args()
        assert args.model == ["m.onnx"]

    def test_model_can_be_repeated(self):
        with patch("sys.argv", [
            "sherox.wake", "--mic", "--model", "a.onnx", "--model", "b.onnx"
        ]):
            args = wake_module.parse_args()
        assert args.model == ["a.onnx", "b.onnx"]

    def test_default_threshold(self):
        with patch("sys.argv", ["sherox.wake", "--mic", "--model", "m.onnx"]):
            args = wake_module.parse_args()
        assert args.threshold == 0.5

    def test_custom_threshold(self):
        with patch("sys.argv", [
            "sherox.wake", "--mic", "--model", "m.onnx", "--threshold", "0.7"
        ]):
            args = wake_module.parse_args()
        assert args.threshold == 0.7

    def test_default_debounce(self):
        with patch("sys.argv", ["sherox.wake", "--mic", "--model", "m.onnx"]):
            args = wake_module.parse_args()
        assert args.debounce == 2.0

    def test_default_chunk_size(self):
        with patch("sys.argv", ["sherox.wake", "--mic", "--model", "m.onnx"]):
            args = wake_module.parse_args()
        assert args.chunk_size == 2.0

    def test_requires_model(self):
        with patch("sys.argv", ["sherox.wake", "--mic"]):
            with pytest.raises(SystemExit):
                wake_module.parse_args()


# ---------------------------------------------------------------------------
# _validate_model_paths
# ---------------------------------------------------------------------------

class TestValidateModelPaths:
    def test_empty_paths_raises(self, tmp_path):
        with pytest.raises(SherpaError):
            wake_module._validate_model_paths([], tmp_path)

    def test_resolves_existing_file(self, tmp_path):
        m = tmp_path / "m.onnx"
        m.write_bytes(b"")
        result = wake_module._validate_model_paths([str(m)], tmp_path)
        assert result == [str(m)]

    def test_resolves_relative_to_models_dir(self, tmp_path):
        m = tmp_path / "models" / "m.onnx"
        m.parent.mkdir()
        m.write_bytes(b"")
        result = wake_module._validate_model_paths(["m.onnx"], tmp_path)
        assert result == [str(m)]

    def test_resolves_bare_name_relative_to_models_dir(self, tmp_path):
        m = tmp_path / "models" / "m.onnx"
        m.parent.mkdir()
        m.write_bytes(b"")
        result = wake_module._validate_model_paths(["m"], tmp_path)
        assert result == [str(m)]

    def test_existing_non_onnx_file_raises(self, tmp_path):
        m = tmp_path / "not_a_model.txt"
        m.write_bytes(b"")
        with pytest.raises(SherpaError):
            wake_module._validate_model_paths([str(m)], tmp_path)

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(SherpaError):
            wake_module._validate_model_paths([str(tmp_path / "no.onnx")], tmp_path)

    def test_multiple_paths(self, tmp_path):
        a = tmp_path / "a.onnx"
        b = tmp_path / "b.onnx"
        a.write_bytes(b"")
        b.write_bytes(b"")
        result = wake_module._validate_model_paths([str(a), str(b)], tmp_path)
        assert result == [str(a), str(b)]


# ---------------------------------------------------------------------------
# _validate_runtime_args
# ---------------------------------------------------------------------------

class TestValidateRuntimeArgs:
    def _args(self, **kwargs):
        values = {
            "threshold": 0.5,
            "debounce": 2.0,
            "chunk_size": 2.0,
            "wav": None,
        }
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_valid_args(self):
        wake_module._validate_runtime_args(self._args())

    def test_threshold_below_0_raises(self):
        with pytest.raises(SherpaError):
            wake_module._validate_runtime_args(self._args(threshold=-0.1))

    def test_threshold_above_1_raises(self):
        with pytest.raises(SherpaError):
            wake_module._validate_runtime_args(self._args(threshold=1.1))

    def test_negative_debounce_raises(self):
        with pytest.raises(SherpaError):
            wake_module._validate_runtime_args(self._args(debounce=-0.1))

    def test_nonpositive_chunk_size_raises(self):
        with pytest.raises(SherpaError):
            wake_module._validate_runtime_args(self._args(chunk_size=0.0))

    def test_missing_wav_raises(self, tmp_path):
        with pytest.raises(SherpaError):
            wake_module._validate_runtime_args(self._args(wav=str(tmp_path / "missing.wav")))


# ---------------------------------------------------------------------------
# WakeConfig
# ---------------------------------------------------------------------------

class TestWakeConfig:
    def test_defaults(self):
        cfg = WakeConfig()
        assert cfg.model_paths == []
        assert cfg.threshold == 0.5
        assert cfg.debounce == 2.0
        assert cfg.chunk_size == 2.0
        assert cfg.wav == ""

    def test_custom_values(self):
        cfg = WakeConfig(
            model_paths=["a.onnx", "b.onnx"],
            threshold=0.7,
            debounce=1.0,
        )
        assert cfg.model_paths == ["a.onnx", "b.onnx"]
        assert cfg.threshold == 0.7
        assert cfg.debounce == 1.0


# ---------------------------------------------------------------------------
# model loading and WAV scanning
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_load_model_wraps_dependency_error(self):
        WakeWordModel = MagicMock(side_effect=ValueError("bad model"))
        with patch.object(wake_module, "_require_livekit_wakeword", return_value=WakeWordModel):
            with pytest.raises(SherpaError):
                wake_module._load_model(["bad.onnx"])


class TestRunWav:
    def test_empty_wav_is_not_scored(self):
        fake_sf = MagicMock()
        fake_sf.read.return_value = (np.array([], dtype=np.float32), 16000)
        model = MagicMock()
        cfg = WakeConfig(model_paths=["m.onnx"], wav="empty.wav", chunk_size=2.0)

        with patch.object(wake_module, "_require_soundfile", return_value=fake_sf):
            wake_module.run_wav(model, cfg)

        model.predict.assert_not_called()

    def test_short_wav_is_padded_and_scored(self):
        fake_sf = MagicMock()
        fake_sf.read.return_value = (np.ones(100, dtype=np.float32), 16000)
        model = MagicMock()
        model.predict.return_value = {"hey": 0.1}
        cfg = WakeConfig(model_paths=["m.onnx"], wav="short.wav", chunk_size=2.0)

        with patch.object(wake_module, "_require_soundfile", return_value=fake_sf):
            wake_module.run_wav(model, cfg)

        model.predict.assert_called_once()
        scored = model.predict.call_args.args[0]
        assert scored.dtype == np.int16
        assert len(scored) == 32000

    def test_short_resampled_wav_is_padded_and_scored(self):
        fake_sf = MagicMock()
        fake_sf.read.return_value = (np.ones(1, dtype=np.float32), 48000)
        model = MagicMock()
        model.predict.return_value = {"hey": 0.1}
        cfg = WakeConfig(model_paths=["m.onnx"], wav="short.wav", chunk_size=2.0)

        with patch.object(wake_module, "_require_soundfile", return_value=fake_sf):
            wake_module.run_wav(model, cfg)

        model.predict.assert_called_once()
        scored = model.predict.call_args.args[0]
        assert len(scored) == 32000

    def test_invalid_wav_sample_rate_raises(self):
        fake_sf = MagicMock()
        fake_sf.read.return_value = (np.ones(100, dtype=np.float32), 0)
        model = MagicMock()
        cfg = WakeConfig(model_paths=["m.onnx"], wav="bad.wav", chunk_size=2.0)

        with patch.object(wake_module, "_require_soundfile", return_value=fake_sf):
            with pytest.raises(SherpaError):
                wake_module.run_wav(model, cfg)


# ---------------------------------------------------------------------------
# main dispatch
# ---------------------------------------------------------------------------

class TestMain:
    def _mock_model(self, scores: dict[str, float] | None = None):
        m = MagicMock()
        m.predict.return_value = scores or {"hey_livekit": 0.1}
        return m

    def test_main_dispatches_to_run_mic(self, tmp_path):
        m_file = tmp_path / "m.onnx"
        m_file.write_bytes(b"")
        with patch("sys.argv", [
            "sherox.wake", "--mic", "--model", str(m_file)
        ]), \
        patch.object(wake_module, "_load_model", return_value=self._mock_model()), \
        patch.object(wake_module, "run_mic") as mock_mic:
            wake_module.main()
        mock_mic.assert_called_once()

    def test_main_dispatches_to_run_wav(self, tmp_path):
        m_file = tmp_path / "m.onnx"
        m_file.write_bytes(b"")
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"")
        with patch("sys.argv", [
            "sherox.wake", "--wav", str(wav), "--model", str(m_file)
        ]), \
        patch.object(wake_module, "_load_model", return_value=self._mock_model()), \
        patch.object(wake_module, "run_wav") as mock_wav:
            wake_module.main()
        mock_wav.assert_called_once()
