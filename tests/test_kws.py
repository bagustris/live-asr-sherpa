"""Tests for sherox.kws — keyword spotting module."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

import sherox.kws as kws_module
from sherox import ConfigError, SherpaError
from sherox.config import KwsConfig


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_mic_mode(self):
        with patch("sys.argv", ["sherox.kws", "--mic", "--keywords", "hey sherpa"]):
            args = kws_module.parse_args()
        assert args.mic is True
        assert args.wav is None

    def test_wav_mode(self):
        with patch("sys.argv", ["sherox.kws", "--wav", "audio.wav", "--keywords", "hey"]):
            args = kws_module.parse_args()
        assert args.wav == "audio.wav"
        assert args.mic is False

    def test_mic_and_wav_mutually_exclusive(self):
        with patch("sys.argv", [
            "sherox.kws", "--mic", "--wav", "a.wav", "--keywords", "hey"
        ]):
            with pytest.raises(SystemExit):
                kws_module.parse_args()

    def test_requires_mic_or_wav(self):
        with patch("sys.argv", ["sherox.kws", "--keywords", "hey"]):
            with pytest.raises(SystemExit):
                kws_module.parse_args()

    def test_keywords_argument(self):
        with patch("sys.argv", ["sherox.kws", "--mic", "--keywords", "hey sherpa, ok google"]):
            args = kws_module.parse_args()
        assert args.keywords == "hey sherpa, ok google"

    def test_keywords_file_argument(self):
        with patch("sys.argv", ["sherox.kws", "--mic", "--keywords-file", "kws.txt"]):
            args = kws_module.parse_args()
        assert args.keywords_file == "kws.txt"

    def test_keywords_and_keywords_file_mutually_exclusive(self):
        with patch("sys.argv", [
            "sherox.kws", "--mic", "--keywords", "hey", "--keywords-file", "kws.txt"
        ]):
            with pytest.raises(SystemExit):
                kws_module.parse_args()

    def test_requires_keywords_or_keywords_file(self):
        with patch("sys.argv", ["sherox.kws", "--mic"]):
            with pytest.raises(SystemExit):
                kws_module.parse_args()

    def test_default_sample_rate(self):
        with patch("sys.argv", ["sherox.kws", "--mic", "--keywords", "hey"]):
            args = kws_module.parse_args()
        assert args.sample_rate == 16000

    def test_default_threads(self):
        with patch("sys.argv", ["sherox.kws", "--mic", "--keywords", "hey"]):
            args = kws_module.parse_args()
        assert args.threads == 4

    def test_default_chunk_size(self):
        with patch("sys.argv", ["sherox.kws", "--mic", "--keywords", "hey"]):
            args = kws_module.parse_args()
        assert args.chunk_size == 0.1

    def test_default_max_active_paths(self):
        with patch("sys.argv", ["sherox.kws", "--mic", "--keywords", "hey"]):
            args = kws_module.parse_args()
        assert args.max_active_paths == 4

    def test_custom_capture_rate(self):
        with patch("sys.argv", [
            "sherox.kws", "--mic", "--keywords", "hey", "--capture-rate", "48000"
        ]):
            args = kws_module.parse_args()
        assert args.capture_rate == 48000

    def test_custom_model_dir(self):
        with patch("sys.argv", [
            "sherox.kws", "--mic", "--keywords", "hey", "--model-dir", "models/custom"
        ]):
            args = kws_module.parse_args()
        assert args.model_dir == "models/custom"

    def test_default_no_mic_level_is_false(self):
        with patch("sys.argv", ["sherox.kws", "--mic", "--keywords", "hey"]):
            args = kws_module.parse_args()
        assert args.no_mic_level is False

    def test_no_mic_level_flag(self):
        with patch("sys.argv", ["sherox.kws", "--mic", "--keywords", "hey", "--no-mic-level"]):
            args = kws_module.parse_args()
        assert args.no_mic_level is True


# ---------------------------------------------------------------------------
# _validate_model
# ---------------------------------------------------------------------------

class TestValidateModel:
    def test_returns_existing_dir(self, tmp_path):
        model_dir = tmp_path / "models" / kws_module._MODEL_NAME
        model_dir.mkdir(parents=True)
        result = kws_module._validate_model("", tmp_path)
        assert result == model_dir

    def test_returns_custom_dir_when_given(self, tmp_path):
        custom = tmp_path / "my_kws_model"
        custom.mkdir()
        result = kws_module._validate_model(str(custom), tmp_path)
        assert result == custom

    def test_exits_when_custom_dir_not_found(self, tmp_path):
        with pytest.raises(SherpaError):
            kws_module._validate_model(str(tmp_path / "no_such_dir"), tmp_path)

    def test_downloads_and_extracts_when_missing(self, tmp_path):
        target_dir = tmp_path / "models" / kws_module._MODEL_NAME

        def fake_download(url, dest):
            pass

        import tarfile, io  # noqa: PLC0415
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
            info = tarfile.TarInfo(name=kws_module._MODEL_NAME)
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
        buf.seek(0)

        with patch.object(kws_module, "download_file", side_effect=fake_download), \
             patch("tarfile.open", return_value=tarfile.open(fileobj=buf, mode="r:bz2")):
            target_dir.mkdir(parents=True)
            result = kws_module._validate_model("", tmp_path)
        assert result == target_dir


# ---------------------------------------------------------------------------
# _resolve_keywords
# ---------------------------------------------------------------------------

class TestResolveKeywords:
    def test_creates_temp_file_from_keywords_str(self, tmp_path):
        cfg = KwsConfig(keywords_str="hey sherpa, ok google")
        path = kws_module._resolve_keywords(cfg, tmp_path)
        try:
            content = Path(path).read_text()
            assert "hey sherpa" in content
            assert "ok google" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_returns_keywords_file_path(self, tmp_path):
        kw_file = tmp_path / "kws.txt"
        kw_file.write_text("hey sherpa\n")
        cfg = KwsConfig(keywords_file=str(kw_file))
        path = kws_module._resolve_keywords(cfg, tmp_path)
        assert path == str(kw_file)

    def test_exits_when_keywords_file_not_found(self, tmp_path):
        cfg = KwsConfig(keywords_file=str(tmp_path / "no_file.txt"))
        with pytest.raises(SherpaError):
            kws_module._resolve_keywords(cfg, tmp_path)

    def test_exits_when_no_keywords_given(self, tmp_path):
        cfg = KwsConfig()  # both keywords_str and keywords_file empty
        with pytest.raises(SherpaError):
            kws_module._resolve_keywords(cfg, tmp_path)

    def test_keywords_str_takes_priority_over_file(self, tmp_path):
        kw_file = tmp_path / "kws.txt"
        kw_file.write_text("from file\n")
        cfg = KwsConfig(keywords_str="from string", keywords_file=str(kw_file))
        path = kws_module._resolve_keywords(cfg, tmp_path)
        try:
            content = Path(path).read_text()
            assert "from string" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_empty_keywords_str_exits(self, tmp_path):
        cfg = KwsConfig(keywords_str="  ,  , ")  # only whitespace/commas
        with pytest.raises(SherpaError):
            kws_module._resolve_keywords(cfg, tmp_path)


# ---------------------------------------------------------------------------
# KwsConfig
# ---------------------------------------------------------------------------

class TestKwsConfig:
    def test_defaults(self):
        cfg = KwsConfig()
        assert cfg.sample_rate == 16000
        assert cfg.chunk_size == 0.1
        assert cfg.num_threads == 4
        assert cfg.max_active_paths == 4
        assert cfg.capture_rate == 16000
        assert cfg.keywords_str == ""
        assert cfg.keywords_file == ""
        assert cfg.model_dir == ""
        assert cfg.wav == ""

    def test_custom_values(self):
        cfg = KwsConfig(
            keywords_str="hey sherpa",
            sample_rate=8000,
            num_threads=2,
        )
        assert cfg.keywords_str == "hey sherpa"
        assert cfg.sample_rate == 8000
        assert cfg.num_threads == 2


# ---------------------------------------------------------------------------
# main dispatch
# ---------------------------------------------------------------------------

class TestMain:
    def _mock_spotter(self):
        spotter = MagicMock()
        stream = MagicMock()
        spotter.create_stream.return_value = stream
        spotter.is_ready.return_value = False
        result = MagicMock()
        result.keyword = ""
        spotter.get_result.return_value = result
        return spotter

    def test_main_calls_run_mic(self, tmp_path):
        model_dir = tmp_path / kws_module._MODEL_NAME
        model_dir.mkdir()
        kw_file = tmp_path / "kws.txt"
        kw_file.write_text("hey sherpa\n")

        with patch("sys.argv", [
            "sherox.kws", "--mic", "--keywords-file", str(kw_file),
            "--model-dir", str(model_dir),
        ]), \
        patch.object(kws_module, "_build_spotter", return_value=self._mock_spotter()), \
        patch.object(kws_module, "run_mic") as mock_run:
            kws_module.main()
        mock_run.assert_called_once()

    def test_main_calls_run_wav(self, tmp_path):
        model_dir = tmp_path / kws_module._MODEL_NAME
        model_dir.mkdir()
        wav = tmp_path / "audio.wav"
        wav.touch()
        kw_file = tmp_path / "kws.txt"
        kw_file.write_text("hey sherpa\n")

        with patch("sys.argv", [
            "sherox.kws", "--wav", str(wav),
            "--keywords-file", str(kw_file),
            "--model-dir", str(model_dir),
        ]), \
        patch.object(kws_module, "_build_spotter", return_value=self._mock_spotter()), \
        patch.object(kws_module, "run_wav") as mock_run:
            kws_module.main()
        mock_run.assert_called_once()

    def test_main_cleans_up_temp_keywords_file(self, tmp_path):
        """Temp file created from --keywords string must be deleted after main()."""
        model_dir = tmp_path / kws_module._MODEL_NAME
        model_dir.mkdir()

        created_paths: list[str] = []

        original_resolve = kws_module._resolve_keywords

        def recording_resolve(cfg, model_dir):
            path = original_resolve(cfg, model_dir)
            created_paths.append(path)
            return path

        with patch("sys.argv", [
            "sherox.kws", "--mic", "--keywords", "hey sherpa",
            "--model-dir", str(model_dir),
        ]), \
        patch.object(kws_module, "_resolve_keywords", side_effect=recording_resolve), \
        patch.object(kws_module, "_build_spotter", return_value=self._mock_spotter()), \
        patch.object(kws_module, "run_mic"):
            kws_module.main()

        for p in created_paths:
            assert not Path(p).exists(), f"Temp file was not cleaned up: {p}"
