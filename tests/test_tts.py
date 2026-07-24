import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest

import sherox.tts as tts_module
from sherox import AudioError, ConfigError, SherpaError
from sherox.config import TtsConfig


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_text_mode(self):
        with patch("sys.argv", ["sherox.tts", "--text", "Hello"]):
            args = tts_module.parse_args()
        assert args.text == "Hello"
        assert args.file is None

    def test_file_mode(self):
        with patch("sys.argv", ["sherox.tts", "--file", "input.txt"]):
            args = tts_module.parse_args()
        assert args.file == "input.txt"
        assert args.text is None

    def test_no_text_source_allowed(self):
        with patch("sys.argv", ["sherox.tts"]):
            args = tts_module.parse_args()
        assert args.text is None
        assert args.file is None

    def test_text_and_file_mutually_exclusive(self):
        with patch("sys.argv", ["sherox.tts", "--text", "hi", "--file", "f.txt"]):
            with pytest.raises(SystemExit):
                tts_module.parse_args()

    def test_defaults(self):
        with patch("sys.argv", ["sherox.tts"]):
            args = tts_module.parse_args()
        assert args.lang == "ind"
        assert args.model_dir is None
        assert args.speaker_id == 0
        assert args.speed == 1.0
        assert args.output == "output.wav"
        assert args.play is False
        assert args.no_save is False
        assert args.threads == 4

    def test_custom_lang(self):
        with patch("sys.argv", ["sherox.tts", "--lang", "jpn"]):
            args = tts_module.parse_args()
        assert args.lang == "jpn"

    def test_custom_lang_alias(self):
        with patch("sys.argv", ["sherox.tts", "--lang", "jp"]):
            args = tts_module.parse_args()
        assert args.lang == "jp"

    def test_custom_lang_kitten(self):
        with patch("sys.argv", ["sherox.tts", "--lang", "eng-kitten"]):
            args = tts_module.parse_args()
        assert args.lang == "eng-kitten"

    def test_custom_speed(self):
        with patch("sys.argv", ["sherox.tts", "--speed", "0.8"]):
            args = tts_module.parse_args()
        assert args.speed == 0.8

    def test_play_flag(self):
        with patch("sys.argv", ["sherox.tts", "--play"]):
            args = tts_module.parse_args()
        assert args.play is True

    def test_no_save_flag(self):
        with patch("sys.argv", ["sherox.tts", "--play", "--no-save"]):
            args = tts_module.parse_args()
        assert args.no_save is True

    def test_custom_output(self):
        with patch("sys.argv", ["sherox.tts", "--output", "out.wav"]):
            args = tts_module.parse_args()
        assert args.output == "out.wav"

    def test_output_none(self):
        with patch("sys.argv", ["sherox.tts", "--play", "--output", "none"]):
            args = tts_module.parse_args()
        assert args.output == "none"

    def test_custom_threads(self):
        with patch("sys.argv", ["sherox.tts", "--threads", "8"]):
            args = tts_module.parse_args()
        assert args.threads == 8

    def test_custom_model_dir(self):
        with patch("sys.argv", ["sherox.tts", "--model-dir", "models/custom"]):
            args = tts_module.parse_args()
        assert args.model_dir == "models/custom"


# ---------------------------------------------------------------------------
# _validate_runtime_args
# ---------------------------------------------------------------------------

class TestValidateRuntimeArgs:
    def _args(self, **kwargs):
        defaults = dict(
            speaker_id=0,
            speed=1.0,
            threads=4,
            play=False,
            no_save=False,
            output="output.wav",
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_valid_passes(self):
        tts_module._validate_runtime_args(self._args())

    def test_negative_speaker_id_exits(self):
        with pytest.raises(ConfigError):
            tts_module._validate_runtime_args(self._args(speaker_id=-1))

    def test_zero_speed_exits(self):
        with pytest.raises(ConfigError):
            tts_module._validate_runtime_args(self._args(speed=0.0))

    def test_negative_speed_exits(self):
        with pytest.raises(ConfigError):
            tts_module._validate_runtime_args(self._args(speed=-0.5))

    def test_zero_threads_exits(self):
        with pytest.raises(ConfigError):
            tts_module._validate_runtime_args(self._args(threads=0))

    def test_no_save_without_play_exits(self):
        with pytest.raises(ConfigError):
            tts_module._validate_runtime_args(self._args(no_save=True))

    def test_output_none_without_play_exits(self):
        with pytest.raises(ConfigError):
            tts_module._validate_runtime_args(self._args(output="none"))

    def test_no_save_with_play_passes(self):
        tts_module._validate_runtime_args(self._args(play=True, no_save=True))

    def test_output_dash_with_play_passes(self):
        tts_module._validate_runtime_args(self._args(play=True, output="-"))


# ---------------------------------------------------------------------------
# _download_file
# ---------------------------------------------------------------------------

class TestDownloadFile:
    def test_success(self, tmp_path):
        dest = tmp_path / "model.tar.bz2"
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "100"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b""]

        with patch("sherox.utils.urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            tts_module._download_file("http://example.com/model.tar.bz2", dest)

        mock_urlopen.assert_called_once()

    def test_failure_exits(self, tmp_path):
        dest = tmp_path / "model.tar.bz2"
        with patch("sherox.utils.urllib.request.urlopen", side_effect=Exception("net error")):
            with pytest.raises(SherpaError):
                tts_module._download_file("http://example.com/model.tar.bz2", dest)

    def test_progress_with_positive_total(self, tmp_path):
        dest = tmp_path / "model.tar.bz2"
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "4096"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b"x" * 1024, b""]

        with patch("sherox.utils.urllib.request.urlopen", return_value=mock_response):
            tts_module._download_file("http://example.com/model.tar.bz2", dest)

        assert dest.exists()

    def test_progress_skipped_when_total_zero(self, tmp_path):
        dest = tmp_path / "model.tar.bz2"
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "0"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b""]

        with patch("sherox.utils.urllib.request.urlopen", return_value=mock_response):
            tts_module._download_file("http://example.com/model.tar.bz2", dest)


# ---------------------------------------------------------------------------
# _safe_tar_members
# ---------------------------------------------------------------------------

class TestSafeTarMembers:
    def test_yields_safe_regular_file(self, tmp_path):
        member = MagicMock()
        member.isdev.return_value = False
        member.name = "safe_file.txt"
        tf = MagicMock()
        tf.getmembers.return_value = [member]
        result = list(tts_module._safe_tar_members(tf, tmp_path))
        assert member in result

    def test_skips_device_files(self, tmp_path):
        member = MagicMock()
        member.isdev.return_value = True
        tf = MagicMock()
        tf.getmembers.return_value = [member]
        result = list(tts_module._safe_tar_members(tf, tmp_path))
        assert result == []

    def test_skips_path_traversal(self, tmp_path):
        member = MagicMock()
        member.isdev.return_value = False
        member.name = "../etc/passwd"
        tf = MagicMock()
        tf.getmembers.return_value = [member]
        result = list(tts_module._safe_tar_members(tf, tmp_path))
        assert result == []

    def test_yields_nested_safe_file(self, tmp_path):
        member = MagicMock()
        member.isdev.return_value = False
        member.name = "subdir/model.onnx"
        tf = MagicMock()
        tf.getmembers.return_value = [member]
        result = list(tts_module._safe_tar_members(tf, tmp_path))
        assert member in result


# ---------------------------------------------------------------------------
# _ensure_model
# ---------------------------------------------------------------------------

class TestEnsureModel:
    def test_language_alias_is_normalized(self, tmp_path):
        meta = tts_module._TTS_MODELS["ind"]
        target = tmp_path / "models" / meta["extracted"]
        target.mkdir(parents=True)
        result = tts_module._ensure_model("id", None, tmp_path)
        assert result == target

    def test_returns_existing_target_dir(self, tmp_path):
        meta = tts_module._TTS_MODELS["ind"]
        target = tmp_path / "models" / meta["extracted"]
        target.mkdir(parents=True)
        result = tts_module._ensure_model("ind", None, tmp_path)
        assert result == target

    def test_returns_custom_dir_when_given(self, tmp_path):
        custom = tmp_path / "my_model"
        custom.mkdir()
        result = tts_module._ensure_model("ind", custom, tmp_path)
        assert result == custom

    def test_exits_on_unsupported_lang(self, tmp_path):
        with pytest.raises(ConfigError):
            tts_module._ensure_model("xyz", None, tmp_path)

    def test_exits_when_custom_dir_not_found(self, tmp_path):
        with pytest.raises(ConfigError):
            tts_module._ensure_model("ind", tmp_path / "no_such_dir", tmp_path)

    def test_downloads_and_extracts_when_missing(self, tmp_path):
        meta = tts_module._TTS_MODELS["ind"]
        target = tmp_path / "models" / meta["extracted"]

        def fake_download(url, dest):
            pass

        def fake_tar_open(*args, **kwargs):
            # Simulate extraction by creating the target directory
            target.mkdir(parents=True, exist_ok=True)
            ctx = MagicMock()
            tf = MagicMock()
            tf.getmembers.return_value = []
            ctx.__enter__ = MagicMock(return_value=tf)
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx

        # Target does NOT exist before call — fake_tar_open creates it
        with patch.object(tts_module, "_download_file", side_effect=fake_download), \
             patch("tarfile.open", side_effect=fake_tar_open):
            result = tts_module._ensure_model("ind", None, tmp_path)
        assert result == target

    def test_exits_when_extracted_dir_missing_after_extraction(self, tmp_path):
        def fake_download(url, dest):
            pass

        def fake_tar_open(*args, **kwargs):
            ctx = MagicMock()
            tf = MagicMock()
            tf.getmembers.return_value = []
            ctx.__enter__ = MagicMock(return_value=tf)
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx

        with patch.object(tts_module, "_download_file", side_effect=fake_download), \
             patch("tarfile.open", side_effect=fake_tar_open), \
             pytest.raises(ConfigError):
            tts_module._ensure_model("ind", None, tmp_path)

    def test_exits_when_extraction_raises(self, tmp_path):
        def fake_download(url, dest):
            pass

        with patch.object(tts_module, "_download_file", side_effect=fake_download), \
             patch("tarfile.open", side_effect=Exception("corrupt archive")), \
             pytest.raises(ConfigError):
            tts_module._ensure_model("ind", None, tmp_path)

    def test_exits_for_non_sherpa_onnx_backend_with_auto_resolution(self, tmp_path):
        # Japanese uses piper_plus backend, which doesn't support auto-resolution
        with pytest.raises(ConfigError):
            tts_module._ensure_model("jpn", None, tmp_path)


# ---------------------------------------------------------------------------
# _ensure_sarashina_onnx_model
# ---------------------------------------------------------------------------

class TestEnsureSarashinaOnnxModel:
    def _write_complete_model(self, target: Path) -> None:
        (target / "llm").mkdir(parents=True, exist_ok=True)
        for name in tts_module._SARASHINA_ONNX_REQUIRED_FILES:
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            (target / name).write_text("x")

    def test_noop_when_already_present(self, tmp_path):
        target = tmp_path / "sarashina-onnx"
        self._write_complete_model(target)
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            tts_module._ensure_sarashina_onnx_model(target)
        mock_dl.assert_not_called()

    def test_uses_shared_cache_link_when_available(self, tmp_path):
        target = tmp_path / "sarashina-onnx"

        def fake_try_link(project_dir, model_type):
            self._write_complete_model(project_dir)
            return True

        with patch("sherox.model_cache.try_link", side_effect=fake_try_link) as mock_link, \
             patch("huggingface_hub.snapshot_download") as mock_dl:
            tts_module._ensure_sarashina_onnx_model(target)
        mock_link.assert_called_once_with(target, "tts_jpn-sarashina-onnx")
        mock_dl.assert_not_called()

    def test_downloads_from_hf_when_missing(self, tmp_path):
        target = tmp_path / "sarashina-onnx"

        def fake_snapshot_download(repo_id, local_dir):
            assert repo_id == tts_module._SARASHINA_ONNX_HF_REPO
            self._write_complete_model(Path(local_dir))

        with patch("sherox.model_cache.try_link", return_value=False), \
             patch("sherox.model_cache.migrate") as mock_migrate, \
             patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download) as mock_dl:
            tts_module._ensure_sarashina_onnx_model(target)
        mock_dl.assert_called_once_with(tts_module._SARASHINA_ONNX_HF_REPO, local_dir=str(target))
        mock_migrate.assert_called_once_with(target, "tts_jpn-sarashina-onnx")
        assert tts_module._sarashina_onnx_model_complete(target)

    def test_exits_when_huggingface_hub_missing(self, tmp_path):
        target = tmp_path / "sarashina-onnx"
        with patch("sherox.model_cache.try_link", return_value=False), \
             patch.dict("sys.modules", {"huggingface_hub": None}), \
             pytest.raises(ConfigError):
            tts_module._ensure_sarashina_onnx_model(target)

    def test_exits_when_files_missing_after_download(self, tmp_path):
        target = tmp_path / "sarashina-onnx"
        with patch("sherox.model_cache.try_link", return_value=False), \
             patch("huggingface_hub.snapshot_download"), \
             pytest.raises(ConfigError):
            tts_module._ensure_sarashina_onnx_model(target)

    def test_stale_cache_link_is_invalidated_and_redownloaded(self, tmp_path):
        """A cached copy that predates campplus.onnx/s3_tokenizer.onnx being
        added must not be silently reused — it should be invalidated and a
        fresh download triggered instead."""
        target = tmp_path / "sarashina-onnx"

        def fake_try_link(project_dir, model_type):
            # Simulate an old, incomplete cache: only the pre-cloning files exist.
            project_dir.mkdir(parents=True, exist_ok=True)
            (project_dir / "meta.json").write_text("{}")
            return True

        def fake_snapshot_download(repo_id, local_dir):
            self._write_complete_model(Path(local_dir))

        with patch("sherox.model_cache.try_link", side_effect=fake_try_link), \
             patch("sherox.model_cache.invalidate") as mock_invalidate, \
             patch("sherox.model_cache.migrate"), \
             patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download) as mock_dl:
            tts_module._ensure_sarashina_onnx_model(target)

        mock_invalidate.assert_called_once_with(target, "tts_jpn-sarashina-onnx")
        mock_dl.assert_called_once()
        assert tts_module._sarashina_onnx_model_complete(target)

    def test_dangling_symlink_is_cleared_before_fresh_download(self, tmp_path):
        """A broken symlink (its cache target removed independently of sherox,
        so try_link's own cache lookup finds nothing) must not make the
        subsequent mkdir(exist_ok=True) raise FileExistsError."""
        target = tmp_path / "sarashina-onnx"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(tmp_path / "nonexistent-cache-target")
        assert target.is_symlink() and not target.exists()

        def fake_snapshot_download(repo_id, local_dir):
            self._write_complete_model(Path(local_dir))

        with patch("sherox.model_cache.try_link", return_value=False), \
             patch("sherox.model_cache.migrate"), \
             patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download):
            tts_module._ensure_sarashina_onnx_model(target)

        assert not target.is_symlink()
        assert tts_module._sarashina_onnx_model_complete(target)


# ---------------------------------------------------------------------------
# build_tts
# ---------------------------------------------------------------------------

class TestBuildTts:
    def test_builds_successfully(self, tmp_path):
        meta = tts_module._TTS_MODELS["ind"]
        model_dir = tmp_path / meta["extracted"]
        model_dir.mkdir()
        (model_dir / meta["model"]).touch()
        (model_dir / meta["tokens"]).touch()
        (model_dir / meta["data_dir"]).mkdir()

        mock_sherpa = MagicMock()
        mock_config = MagicMock()
        mock_config.validate.return_value = True
        mock_sherpa.OfflineTtsConfig.return_value = mock_config
        mock_tts = MagicMock()
        mock_sherpa.OfflineTts.return_value = mock_tts

        cfg = TtsConfig(language="ind", model_dir=str(model_dir))
        with patch.dict("sys.modules", {"sherpa_onnx": mock_sherpa}):
            result = tts_module.build_tts(cfg, tmp_path)
        assert result.backend == "sherpa_onnx"
        assert result.model is mock_tts

    def test_builds_piper_plus_successfully(self, tmp_path):
        mock_piper = SimpleNamespace(
            get_voices=MagicMock(return_value={"ja_JP-tsukuyomi-chan-medium": {}}),
            ensure_voice_exists=MagicMock(),
            find_voice=MagicMock(
                return_value=(tmp_path / "tsukuyomi.onnx", tmp_path / "config.json")
            ),
            PiperVoice=SimpleNamespace(load=MagicMock()),
        )
        mock_engine = MagicMock()
        mock_piper.PiperVoice.load.return_value = mock_engine
        cfg = TtsConfig(language="jpn")
        with patch.object(tts_module, "_require_piper_plus", return_value=mock_piper):
            result = tts_module.build_tts(cfg, tmp_path)
        assert result.backend == "piper_plus"
        assert result.model is mock_engine
        assert result.language_id == 0
        mock_piper.ensure_voice_exists.assert_called_once()
        mock_piper.PiperVoice.load.assert_called_once()

    def test_exits_on_unsupported_lang(self, tmp_path):
        cfg = TtsConfig(language="xyz")
        mock_sherpa = MagicMock()
        with patch.dict("sys.modules", {"sherpa_onnx": mock_sherpa}), \
             pytest.raises(ConfigError):
            tts_module.build_tts(cfg, tmp_path)

    def test_exits_on_invalid_config(self, tmp_path):
        meta = tts_module._TTS_MODELS["ind"]
        model_dir = tmp_path / meta["extracted"]
        model_dir.mkdir()
        (model_dir / meta["model"]).touch()
        (model_dir / meta["tokens"]).touch()
        (model_dir / meta["data_dir"]).mkdir()

        mock_sherpa = MagicMock()
        mock_config = MagicMock()
        mock_config.validate.return_value = False
        mock_sherpa.OfflineTtsConfig.return_value = mock_config

        cfg = TtsConfig(language="ind", model_dir=str(model_dir))
        with patch.dict("sys.modules", {"sherpa_onnx": mock_sherpa}), \
             pytest.raises(ConfigError):
            tts_module.build_tts(cfg, tmp_path)

    def test_piper_plus_exits_when_model_dir_given(self, tmp_path):
        model_dir = tmp_path / "custom"
        model_dir.mkdir()
        cfg = TtsConfig(language="jpn", model_dir=str(model_dir))
        with pytest.raises(ConfigError):
            tts_module.build_tts(cfg, tmp_path)

    def test_require_piper_plus_returns_cached_runtime(self):
        import sherox.tts as tts_module
        mock_piper = SimpleNamespace(
            get_voices=MagicMock(),
            ensure_voice_exists=MagicMock(),
            find_voice=MagicMock(),
            PiperVoice=MagicMock(),
        )
        original = tts_module.piper_runtime
        try:
            tts_module.piper_runtime = mock_piper
            result = tts_module._require_piper_plus()
            assert result is mock_piper
        finally:
            tts_module.piper_runtime = original

    def test_require_piper_plus_imports_when_not_cached(self):
        import sherox.tts as tts_module
        original = tts_module.piper_runtime
        try:
            tts_module.piper_runtime = None
            with patch.dict("sys.modules", {
                "piper": MagicMock(),
                "piper.download": MagicMock(),
                "piper.voice": MagicMock(),
            }):
                result = tts_module._require_piper_plus()
            assert result is not None
            assert hasattr(result, "ensure_voice_exists")
        finally:
            tts_module.piper_runtime = original


# ---------------------------------------------------------------------------
# synthesise
# ---------------------------------------------------------------------------

class TestSynthesise:
    def test_returns_samples_and_rate(self):
        mock_audio = MagicMock()
        mock_audio.samples = [0.1, 0.2, 0.3]
        mock_audio.sample_rate = 22050
        mock_tts = MagicMock()
        mock_tts.generate.return_value = mock_audio
        cfg = TtsConfig(speaker_id=0, speed=1.0)
        samples, sr = tts_module.synthesise(mock_tts, "Hello", cfg)
        assert sr == 22050
        assert samples.dtype == np.float32
        mock_tts.generate.assert_called_once_with(text="Hello", sid=0, speed=1.0)


# ---------------------------------------------------------------------------
# synthesise_to_file
# ---------------------------------------------------------------------------

class TestSynthesiseToFile:
    def test_writes_sherpa_output(self):
        mock_audio = MagicMock()
        mock_audio.samples = [0.1, 0.2]
        mock_audio.sample_rate = 22050
        mock_tts = MagicMock()
        mock_tts.generate.return_value = mock_audio
        mock_sf = MagicMock()
        cfg = TtsConfig(output="out.wav")
        with patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            result = tts_module.synthesise_to_file(mock_tts, "Hello", cfg)
        assert result[1] == 22050
        mock_sf.write.assert_called_once()

    def test_sherpa_no_save_skips_write(self):
        mock_audio = MagicMock()
        mock_audio.samples = [0.1, 0.2]
        mock_audio.sample_rate = 22050
        mock_tts = MagicMock()
        mock_tts.generate.return_value = mock_audio
        mock_sf = MagicMock()
        cfg = TtsConfig(output="out.wav", play=True, no_save=True)
        with patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            samples, sample_rate = tts_module.synthesise_to_file(mock_tts, "Hello", cfg)
        assert sample_rate == 22050
        assert samples.dtype == np.float32
        mock_sf.write.assert_not_called()

    def test_writes_piper_plus_output(self):
        mock_engine = MagicMock()
        tts = SimpleNamespace(backend="piper_plus", model=mock_engine)
        mock_wav = MagicMock()
        mock_sf = MagicMock()
        mock_sf.read.return_value = (np.array([0.1, 0.2], dtype=np.float32), 22050)
        cfg = TtsConfig(language="jpn", output="out.wav")
        with patch("wave.open") as mock_wave_open, \
             patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            mock_wave_open.return_value.__enter__.return_value = mock_wav
            samples, sample_rate = tts_module.synthesise_to_file(tts, "こんにちは", cfg)
        assert sample_rate == 22050
        assert samples.dtype == np.float32
        mock_engine.synthesize.assert_called_once()
        mock_sf.read.assert_called_once_with("out.wav", dtype="float32")

    def test_reads_piper_plus_output_for_playback(self):
        mock_engine = MagicMock()
        tts = SimpleNamespace(backend="piper_plus", model=mock_engine)
        mock_wav = MagicMock()
        mock_sf = MagicMock()
        mock_sf.read.return_value = (np.array([0.1, 0.2], dtype=np.float32), 22050)
        cfg = TtsConfig(language="jpn", output="out.wav", play=True)
        with patch("wave.open") as mock_wave_open, \
             patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            mock_wave_open.return_value.__enter__.return_value = mock_wav
            samples, sample_rate = tts_module.synthesise_to_file(tts, "こんにちは", cfg)
        assert sample_rate == 22050
        assert samples.dtype == np.float32
        mock_engine.synthesize.assert_called_once()
        mock_sf.read.assert_called_once_with("out.wav", dtype="float32")

    def test_piper_plus_no_save_uses_memory_buffer_for_playback(self):
        mock_engine = MagicMock()
        tts = SimpleNamespace(backend="piper_plus", model=mock_engine)
        mock_wav = MagicMock()
        mock_sf = MagicMock()
        mock_sf.read.return_value = (np.array([0.1, 0.2], dtype=np.float32), 22050)
        cfg = TtsConfig(language="jpn", output="out.wav", play=True, no_save=True)
        with patch("wave.open") as mock_wave_open, \
             patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            mock_wave_open.return_value.__enter__.return_value = mock_wav
            samples, sample_rate = tts_module.synthesise_to_file(tts, "こんにちは", cfg)
        assert sample_rate == 22050
        assert samples.dtype == np.float32
        assert isinstance(mock_wave_open.call_args.args[0], tts_module.io.BytesIO)
        assert isinstance(mock_sf.read.call_args.args[0], tts_module.io.BytesIO)

    def test_exits_for_unsupported_backend(self):
        tts = SimpleNamespace(backend="unsupported_backend")
        cfg = TtsConfig(output="out.wav")
        with pytest.raises(ConfigError):
            tts_module.synthesise_to_file(tts, "Hello", cfg)

    def test_raises_assertion_error_for_unsupported_backend_when_error_mocked(self):
        tts = SimpleNamespace(backend="unsupported_backend")
        cfg = TtsConfig(output="out.wav")
        # Mock _error to not exit, so the code continues to the unreachable line
        with patch.object(tts_module, "_error"):
            with pytest.raises(AssertionError, match="unreachable"):
                tts_module.synthesise_to_file(tts, "Hello", cfg)


# ---------------------------------------------------------------------------
# _play
# ---------------------------------------------------------------------------

class TestPlay:
    def test_plays_audio(self):
        mock_sd = MagicMock()
        samples = np.zeros(1000, dtype=np.float32)
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            tts_module._play(samples, 22050)
        mock_sd.play.assert_called_once_with(samples, samplerate=22050)
        mock_sd.wait.assert_called_once()

    def test_exits_when_sounddevice_missing(self):
        samples = np.zeros(1000, dtype=np.float32)
        with patch.dict("sys.modules", {"sounddevice": None}):
            with pytest.raises(ConfigError):
                tts_module._play(samples, 22050)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def _mock_tts(self):
        mock_tts = MagicMock()
        mock_audio = MagicMock()
        mock_audio.samples = [0.0] * 100
        mock_audio.sample_rate = 22050
        mock_tts.generate.return_value = mock_audio
        return mock_tts

    def test_main_with_text(self, tmp_path):
        out = str(tmp_path / "out.wav")
        mock_sf = MagicMock()
        with patch("sys.argv", ["sherox.tts", "--text", "Hello", "--output", out]), \
             patch.object(tts_module, "build_tts", return_value=self._mock_tts()), \
             patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            tts_module.main()
        mock_sf.write.assert_called_once()

    def test_main_with_file(self, tmp_path):
        txt = tmp_path / "input.txt"
        txt.write_text("Hello world")
        out = str(tmp_path / "out.wav")
        mock_sf = MagicMock()
        with patch("sys.argv", ["sherox.tts", "--file", str(txt), "--output", out]), \
             patch.object(tts_module, "build_tts", return_value=self._mock_tts()), \
             patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            tts_module.main()
        mock_sf.write.assert_called_once()

    def test_main_exits_if_file_not_found(self, tmp_path):
        with patch("sys.argv", ["sherox.tts", "--file", str(tmp_path / "missing.txt")]), \
             pytest.raises(SystemExit):
            tts_module.main()

    def test_main_with_stdin(self, tmp_path):
        out = str(tmp_path / "out.wav")
        mock_sf = MagicMock()
        with patch("sys.argv", ["sherox.tts", "--output", out]), \
             patch("sys.stdin.isatty", return_value=False), \
             patch("sys.stdin.read", return_value="Hello stdin"), \
             patch.object(tts_module, "build_tts", return_value=self._mock_tts()), \
             patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            tts_module.main()
        mock_sf.write.assert_called_once()

    def test_main_stdin_tty_shows_prompt(self, tmp_path, capsys):
        out = str(tmp_path / "out.wav")
        mock_sf = MagicMock()
        with patch("sys.argv", ["sherox.tts", "--output", out]), \
             patch("sys.stdin.isatty", return_value=True), \
             patch("sys.stdin.read", return_value="Hello"), \
             patch.object(tts_module, "build_tts", return_value=self._mock_tts()), \
             patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            tts_module.main()

    def test_main_exits_on_empty_text(self, tmp_path):
        with patch("sys.argv", ["sherox.tts"]), \
             patch("sys.stdin.isatty", return_value=False), \
             patch("sys.stdin.read", return_value="   "), \
             pytest.raises(SystemExit):
            tts_module.main()

    def test_main_with_play(self, tmp_path):
        out = str(tmp_path / "out.wav")
        mock_sf = MagicMock()
        with patch("sys.argv", ["sherox.tts", "--text", "Hi", "--output", out, "--play"]), \
             patch.object(tts_module, "build_tts", return_value=self._mock_tts()), \
             patch.object(tts_module, "_play") as mock_play, \
             patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            tts_module.main()
        mock_play.assert_called_once()

    def test_main_with_play_no_save(self):
        mock_sf = MagicMock()
        with patch("sys.argv", ["sherox.tts", "--text", "Hi", "--play", "--no-save"]), \
             patch.object(tts_module, "build_tts", return_value=self._mock_tts()), \
             patch.object(tts_module, "_play") as mock_play, \
             patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            tts_module.main()
        mock_play.assert_called_once()
        mock_sf.write.assert_not_called()

    def test_main_with_output_none_sets_no_save(self):
        mock_sf = MagicMock()
        with patch("sys.argv", ["sherox.tts", "--text", "Hi", "--play", "--output", "none"]), \
             patch.object(tts_module, "build_tts", return_value=self._mock_tts()) as mock_build, \
             patch.object(tts_module, "_play"), \
             patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            tts_module.main()
        cfg = mock_build.call_args[0][0]
        assert cfg.no_save is True
        mock_sf.write.assert_not_called()

    def test_main_with_output_dash_sets_no_save(self):
        mock_sf = MagicMock()
        with patch("sys.argv", ["sherox.tts", "--text", "Hi", "--play", "--output", "-"]), \
             patch.object(tts_module, "build_tts", return_value=self._mock_tts()) as mock_build, \
             patch.object(tts_module, "_play"), \
             patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            tts_module.main()
        cfg = mock_build.call_args[0][0]
        assert cfg.no_save is True
        mock_sf.write.assert_not_called()

    def test_main_no_save_without_play_exits(self):
        with patch("sys.argv", ["sherox.tts", "--text", "Hi", "--no-save"]), \
             pytest.raises(SystemExit):
            tts_module.main()

    def test_main_exits_when_playback_requested_but_no_samples(self, tmp_path):
        out = str(tmp_path / "out.wav")
        mock_sf = MagicMock()
        # Mock synthesise_to_file to return None (no samples)
        with patch("sys.argv", ["sherox.tts", "--text", "Hi", "--output", out, "--play"]), \
             patch.object(tts_module, "build_tts", return_value=self._mock_tts()), \
             patch.object(tts_module, "synthesise_to_file", return_value=None), \
             patch.object(tts_module, "_require_soundfile", return_value=mock_sf), \
             pytest.raises(SystemExit):
            tts_module.main()

    def test_main_with_custom_model_dir(self, tmp_path):
        model_dir = tmp_path / "custom_model"
        model_dir.mkdir()
        out = str(tmp_path / "out.wav")
        mock_sf = MagicMock()
        with patch("sys.argv", [
            "sherox.tts", "--text", "Hello",
            "--model-dir", str(model_dir),
            "--output", out,
        ]), \
        patch.object(tts_module, "build_tts", return_value=self._mock_tts()), \
        patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            tts_module.main()
        mock_sf.write.assert_called_once()

    def test_main_normalizes_language_alias(self, tmp_path):
        out = str(tmp_path / "out.wav")
        mock_sf = MagicMock()
        with patch("sys.argv", ["sherox.tts", "--text", "Hi", "--lang", "jp", "--output", out]), \
             patch.object(tts_module, "build_tts", return_value=self._mock_tts()) as mock_build, \
             patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            tts_module.main()

        cfg = mock_build.call_args[0][0]
        assert cfg.language == "jpn"


# ---------------------------------------------------------------------------
# _require_soundfile — success path
# ---------------------------------------------------------------------------

class TestRequireSoundfileTts:
    def test_imports_when_sentinel_is_none(self):
        import types
        fake_sf = MagicMock()
        fake_sf.write = MagicMock()
        initial = types.SimpleNamespace(write=None)
        with patch.object(tts_module, "sf", initial):
            with patch.dict("sys.modules", {"soundfile": fake_sf}):
                result = tts_module._require_soundfile()
        assert result is fake_sf

    def test_returns_early_when_already_loaded(self):
        fake_sf = MagicMock()
        fake_sf.write = MagicMock()
        with patch.object(tts_module, "sf", fake_sf):
            result = tts_module._require_soundfile()
        assert result is fake_sf


# ---------------------------------------------------------------------------
# _require_sarashina
# ---------------------------------------------------------------------------

class TestRequireSarashina:
    def test_returns_cached_runtime(self):
        mock_sarashina = SimpleNamespace(SarashinaTTSGenerator=MagicMock())
        original = tts_module.sarashina_runtime
        try:
            tts_module.sarashina_runtime = mock_sarashina
            result = tts_module._require_sarashina()
            assert result is mock_sarashina
        finally:
            tts_module.sarashina_runtime = original

    def test_imports_when_not_cached(self):
        original = tts_module.sarashina_runtime
        try:
            tts_module.sarashina_runtime = None
            mock_gen = MagicMock()
            mock_mod = MagicMock()
            mock_mod.SarashinaTTSGenerator = mock_gen
            with patch.dict("sys.modules", {
                "sarashina_tts": MagicMock(),
                "sarashina_tts.generate": MagicMock(),
                "sarashina_tts.generate.generate": mock_mod,
            }):
                result = tts_module._require_sarashina()
            assert result is not None
            assert hasattr(result, "SarashinaTTSGenerator")
        finally:
            tts_module.sarashina_runtime = original


# ---------------------------------------------------------------------------
# _quantize_sarashina_llm
# ---------------------------------------------------------------------------

class TestQuantizeSarashinaLlm:
    def test_replaces_text_generator_model_with_quantized_version(self):
        torch = pytest.importorskip("torch")
        llm = torch.nn.Linear(4, 4)
        generator = SimpleNamespace(text_generator=SimpleNamespace(model=llm))

        tts_module._quantize_sarashina_llm(generator)

        assert generator.text_generator.model is not llm
        out = generator.text_generator.model(torch.zeros(1, 4))
        assert out.shape == (1, 4)

    def test_noop_when_text_generator_missing(self):
        generator = SimpleNamespace()
        tts_module._quantize_sarashina_llm(generator)  # should not raise


# ---------------------------------------------------------------------------
# build_tts — sarashina backend
# ---------------------------------------------------------------------------

class TestBuildTtsSarashina:
    def test_builds_sarashina_successfully(self, tmp_path):
        mock_generator = MagicMock()
        mock_sarashina = SimpleNamespace(
            SarashinaTTSGenerator=MagicMock(return_value=mock_generator)
        )
        cfg = TtsConfig(language="jpn-sarashina")
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.object(tts_module, "_require_sarashina", return_value=mock_sarashina), \
             patch.dict("sys.modules", {"torch": mock_torch}):
            result = tts_module.build_tts(cfg, tmp_path)
        assert result.backend == "sarashina"
        assert result.model is mock_generator
        assert result.prompt_cache == {}
        call_kwargs = mock_sarashina.SarashinaTTSGenerator.call_args[1]
        assert call_kwargs["decoder_fp16"] is False
        assert call_kwargs["watermark"] is False

    def test_builds_sarashina_passes_watermark_true(self, tmp_path):
        mock_generator = MagicMock()
        mock_sarashina = SimpleNamespace(
            SarashinaTTSGenerator=MagicMock(return_value=mock_generator)
        )
        cfg = TtsConfig(language="jpn-sarashina", watermark=True)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.object(tts_module, "_require_sarashina", return_value=mock_sarashina), \
             patch.dict("sys.modules", {"torch": mock_torch}):
            tts_module.build_tts(cfg, tmp_path)
        call_kwargs = mock_sarashina.SarashinaTTSGenerator.call_args[1]
        assert call_kwargs["watermark"] is True

    def test_builds_sarashina_fp16_enabled_on_cuda(self, tmp_path):
        mock_generator = MagicMock()
        mock_sarashina = SimpleNamespace(
            SarashinaTTSGenerator=MagicMock(return_value=mock_generator)
        )
        cfg = TtsConfig(language="jpn-sarashina")
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.object(tts_module, "_require_sarashina", return_value=mock_sarashina), \
             patch.dict("sys.modules", {"torch": mock_torch}):
            result = tts_module.build_tts(cfg, tmp_path)
        call_kwargs = mock_sarashina.SarashinaTTSGenerator.call_args[1]
        assert call_kwargs["decoder_fp16"] is True

    def test_builds_sarashina_quantizes_llm_on_cpu(self, tmp_path):
        mock_generator = MagicMock()
        mock_sarashina = SimpleNamespace(
            SarashinaTTSGenerator=MagicMock(return_value=mock_generator)
        )
        cfg = TtsConfig(language="jpn-sarashina")
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.object(tts_module, "_require_sarashina", return_value=mock_sarashina), \
             patch.object(tts_module, "_quantize_sarashina_llm") as mock_quantize, \
             patch.dict("sys.modules", {"torch": mock_torch}):
            tts_module.build_tts(cfg, tmp_path)
        mock_quantize.assert_called_once_with(mock_generator)

    def test_builds_sarashina_skips_quantization_on_cuda(self, tmp_path):
        mock_generator = MagicMock()
        mock_sarashina = SimpleNamespace(
            SarashinaTTSGenerator=MagicMock(return_value=mock_generator)
        )
        cfg = TtsConfig(language="jpn-sarashina")
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.object(tts_module, "_require_sarashina", return_value=mock_sarashina), \
             patch.object(tts_module, "_quantize_sarashina_llm") as mock_quantize, \
             patch.dict("sys.modules", {"torch": mock_torch}):
            tts_module.build_tts(cfg, tmp_path)
        mock_quantize.assert_not_called()

    def test_builds_sarashina_with_custom_model_dir(self, tmp_path):
        custom_dir = tmp_path / "my_sarashina"
        custom_dir.mkdir()
        mock_generator = MagicMock()
        mock_sarashina = SimpleNamespace(
            SarashinaTTSGenerator=MagicMock(return_value=mock_generator)
        )
        cfg = TtsConfig(language="jpn-sarashina", model_dir=str(custom_dir))
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.object(tts_module, "_require_sarashina", return_value=mock_sarashina), \
             patch.dict("sys.modules", {"torch": mock_torch}):
            result = tts_module.build_tts(cfg, tmp_path)
        assert result.backend == "sarashina"
        call_kwargs = mock_sarashina.SarashinaTTSGenerator.call_args[1]
        assert call_kwargs["model_dir"] == str(custom_dir)

    def test_sarashina_alias_resolves(self, tmp_path):
        mock_generator = MagicMock()
        mock_sarashina = SimpleNamespace(
            SarashinaTTSGenerator=MagicMock(return_value=mock_generator)
        )
        cfg = TtsConfig(language="sarashina")
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.object(tts_module, "_require_sarashina", return_value=mock_sarashina), \
             patch.dict("sys.modules", {"torch": mock_torch}):
            result = tts_module.build_tts(cfg, tmp_path)
        assert result.backend == "sarashina"

    def test_ensure_model_exits_for_sarashina_backend(self, tmp_path):
        with pytest.raises(ConfigError):
            tts_module._ensure_model("jpn-sarashina", None, tmp_path)


# ---------------------------------------------------------------------------
# synthesise_to_file — sarashina backend
# ---------------------------------------------------------------------------

class TestSynthesiseToFileSarashina:
    def _make_tts(self):
        mock_generator = MagicMock()
        import torch
        mock_wav = torch.zeros(1, 24000)
        mock_generator.generate.return_value = [mock_wav]
        return SimpleNamespace(backend="sarashina", model=mock_generator)

    def test_synthesise_without_audio_prompt(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        tts = self._make_tts()
        mock_sf = MagicMock()
        cfg = TtsConfig(language="jpn-sarashina", output="out.wav")
        with patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            result = tts_module.synthesise_to_file(tts, "こんにちは。", cfg)
        tts.model.generate.assert_called_once_with(["こんにちは。"], flow_embedding=None)
        assert result[1] == 24000
        mock_sf.write.assert_called_once()

    def test_synthesise_with_audio_prompt(self, tmp_path):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        prompt_file = tmp_path / "prompt.wav"
        prompt_file.touch()
        tts = self._make_tts()
        mock_sf = MagicMock()
        cfg = TtsConfig(
            language="jpn-sarashina",
            output="out.wav",
            audio_prompt=str(prompt_file),
            audio_prompt_text="テスト音声です。",
        )
        with patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            result = tts_module.synthesise_to_file(tts, "こんにちは。", cfg)
        tts.model._extract_audio_prompt_tokens.assert_called_once_with(str(prompt_file))
        tts.model._extract_zero_shot_embedding.assert_called_once_with(str(prompt_file))
        tts.model._extract_audio_prompt_feat.assert_called_once_with(str(prompt_file))
        tts.model.generate.assert_called_once()
        assert result[1] == 24000

    def test_audio_prompt_extraction_cached_across_calls(self, tmp_path):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        prompt_file = tmp_path / "prompt.wav"
        prompt_file.touch()
        tts = self._make_tts()
        mock_sf = MagicMock()
        cfg = TtsConfig(
            language="jpn-sarashina",
            output="out.wav",
            audio_prompt=str(prompt_file),
            audio_prompt_text="テスト音声です。",
        )
        with patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            tts_module.synthesise_to_file(tts, "こんにちは。", cfg)
            tts_module.synthesise_to_file(tts, "もう一度。", cfg)
        tts.model._extract_audio_prompt_tokens.assert_called_once_with(str(prompt_file))
        tts.model._extract_zero_shot_embedding.assert_called_once_with(str(prompt_file))
        tts.model._extract_audio_prompt_feat.assert_called_once_with(str(prompt_file))
        assert tts.model.generate.call_count == 2

    def test_audio_prompt_cache_busts_on_file_change(self, tmp_path):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        prompt_file = tmp_path / "prompt.wav"
        prompt_file.write_bytes(b"a")
        tts = self._make_tts()
        mock_sf = MagicMock()
        cfg = TtsConfig(
            language="jpn-sarashina",
            output="out.wav",
            audio_prompt=str(prompt_file),
            audio_prompt_text="テスト音声です。",
        )
        with patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            tts_module.synthesise_to_file(tts, "こんにちは。", cfg)
            prompt_file.write_bytes(b"ab")
            tts_module.synthesise_to_file(tts, "こんにちは。", cfg)
        assert tts.model._extract_audio_prompt_tokens.call_count == 2

    def test_synthesise_returns_float32(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        tts = self._make_tts()
        mock_sf = MagicMock()
        cfg = TtsConfig(language="jpn-sarashina", output="out.wav")
        with patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            samples, sr = tts_module.synthesise_to_file(tts, "テスト", cfg)
        assert samples.dtype == np.float32

    def test_no_save_skips_write(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        tts = self._make_tts()
        mock_sf = MagicMock()
        cfg = TtsConfig(language="jpn-sarashina", output="out.wav", play=True, no_save=True)
        with patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            samples, sr = tts_module.synthesise_to_file(tts, "テスト", cfg)
        assert sr == 24000
        assert samples.dtype == np.float32
        mock_sf.write.assert_not_called()


# ---------------------------------------------------------------------------
# build_tts / synthesise_to_file — sarashina_onnx backend
# ---------------------------------------------------------------------------

class TestSarashinaOnnxBackend:
    def test_language_alias_and_registry(self):
        assert tts_module._normalize_language("sarashina-onnx") == "jpn-sarashina-onnx"
        assert tts_module._normalize_language("jpn_sarashina_onnx") == "jpn-sarashina-onnx"
        assert tts_module._TTS_MODELS["jpn-sarashina-onnx"]["backend"] == "sarashina_onnx"

    def test_build_errors_when_artifacts_missing(self, tmp_path):
        cfg = TtsConfig(language="jpn-sarashina-onnx", model_dir=str(tmp_path))
        with pytest.raises(ConfigError):
            tts_module.build_tts(cfg, tmp_path)

    def test_build_constructs_runtime(self, tmp_path):
        model_dir = tmp_path / "sarashina-onnx"
        model_dir.mkdir()
        (model_dir / "meta.json").write_text("{}")
        mock_runtime = MagicMock()
        mock_mod = SimpleNamespace(SarashinaOnnxRuntime=MagicMock(return_value=mock_runtime))
        cfg = TtsConfig(language="jpn-sarashina-onnx", model_dir=str(model_dir), num_threads=2)
        with patch.dict("sys.modules", {"sherox.sarashina_onnx": mock_mod}):
            result = tts_module.build_tts(cfg, tmp_path)
        assert result.backend == "sarashina_onnx"
        assert result.model is mock_runtime
        assert result.model_dir == str(model_dir)
        mock_mod.SarashinaOnnxRuntime.assert_called_once_with(str(model_dir), num_threads=2)

    def test_build_auto_downloads_when_no_model_dir_given(self, tmp_path):
        """Without an explicit --model-dir, build_tts must try to auto-download
        rather than erroring — this is the whole point of publishing the model
        to Hugging Face."""
        mock_runtime = MagicMock()
        mock_mod = SimpleNamespace(SarashinaOnnxRuntime=MagicMock(return_value=mock_runtime))
        cfg = TtsConfig(language="jpn-sarashina-onnx")

        def fake_ensure(target_dir):
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "meta.json").write_text("{}")

        with patch.dict("sys.modules", {"sherox.sarashina_onnx": mock_mod}), \
             patch.object(tts_module, "_ensure_sarashina_onnx_model", side_effect=fake_ensure) as mock_ensure:
            result = tts_module.build_tts(cfg, tmp_path)

        expected_dir = tmp_path / "models" / "sarashina-onnx"
        mock_ensure.assert_called_once_with(expected_dir)
        assert result.model_dir == str(expected_dir)

    def test_synthesise_default_voice(self, tmp_path):
        runtime = MagicMock()
        runtime.synthesise.return_value = (np.zeros(24000, dtype=np.float32), 24000)
        tts = SimpleNamespace(backend="sarashina_onnx", model=runtime, model_dir=str(tmp_path))
        mock_sf = MagicMock()
        cfg = TtsConfig(language="jpn-sarashina-onnx", output="out.wav")
        with patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            samples, sr = tts_module.synthesise_to_file(tts, "テスト", cfg)
        runtime.synthesise.assert_called_once_with("テスト", seed=0)
        assert sr == 24000
        assert samples.dtype == np.float32
        mock_sf.write.assert_called_once()

    def test_synthesise_with_audio_prompt_extracts_and_caches(self, tmp_path):
        prompt_file = tmp_path / "prompt.wav"
        prompt_file.touch()
        runtime = MagicMock()
        runtime.synthesise.return_value = (np.zeros(100, dtype=np.float32), 24000)
        tts = SimpleNamespace(
            backend="sarashina_onnx", model=runtime, model_dir=str(tmp_path),
            prompt_cache=__import__("collections").OrderedDict(),
        )
        mock_sf = MagicMock()
        mock_extract = MagicMock(return_value=([1, 2, 3], np.zeros(192, dtype=np.float32), np.zeros((1, 5, 80), dtype=np.float32)))
        mock_mod = SimpleNamespace(extract_prompt_features=mock_extract)
        cfg = TtsConfig(
            language="jpn-sarashina-onnx", output="out.wav",
            audio_prompt=str(prompt_file), audio_prompt_text="プロンプト。",
        )
        with patch.dict("sys.modules", {"sherox.sarashina_onnx": mock_mod}), \
             patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            tts_module.synthesise_to_file(tts, "こんにちは。", cfg)
            tts_module.synthesise_to_file(tts, "もう一度。", cfg)
        # Extraction runs once (against the ONNX model dir, which now also holds
        # campplus.onnx / s3_tokenizer.onnx) and is reused from the cache on the
        # second call.
        mock_extract.assert_called_once_with(str(prompt_file), str(tmp_path))
        assert runtime.synthesise.call_count == 2
        _, kwargs = runtime.synthesise.call_args
        assert kwargs["audio_prompt_tokens"] == [1, 2, 3]
        assert kwargs["audio_prompt_text"] == "プロンプト。"


# ---------------------------------------------------------------------------
# parse_args — audio-prompt args
# ---------------------------------------------------------------------------

class TestParseArgsAudioPrompt:
    def test_audio_prompt_default_none(self):
        with patch("sys.argv", ["sherox.tts"]):
            args = tts_module.parse_args()
        assert args.audio_prompt is None
        assert args.audio_prompt_text == ""

    def test_audio_prompt_set(self):
        with patch("sys.argv", ["sherox.tts", "--audio-prompt", "ref.wav"]):
            args = tts_module.parse_args()
        assert args.audio_prompt == "ref.wav"

    def test_audio_prompt_text_set(self):
        with patch("sys.argv", ["sherox.tts", "--audio-prompt-text", "テスト"]):
            args = tts_module.parse_args()
        assert args.audio_prompt_text == "テスト"

    def test_watermark_default_false(self):
        with patch("sys.argv", ["sherox.tts"]):
            args = tts_module.parse_args()
        assert args.watermark is False

    def test_watermark_flag_enables(self):
        with patch("sys.argv", ["sherox.tts", "--watermark"]):
            args = tts_module.parse_args()
        assert args.watermark is True


# ---------------------------------------------------------------------------
# main — sarashina integration
# ---------------------------------------------------------------------------

class TestMainSarashina:
    def _mock_sarashina_tts(self):
        try:
            import torch
        except ImportError:
            return None
        mock_generator = MagicMock()
        mock_wav = torch.zeros(1, 24000)
        mock_generator.generate.return_value = [mock_wav]
        return SimpleNamespace(backend="sarashina", model=mock_generator)

    def test_main_exits_when_audio_prompt_not_found(self, tmp_path):
        out = str(tmp_path / "out.wav")
        with patch("sys.argv", [
            "sherox.tts", "--text", "こんにちは。",
            "--lang", "jpn-sarashina",
            "--audio-prompt", str(tmp_path / "missing.wav"),
            "--output", out,
        ]), pytest.raises(SystemExit):
            tts_module.main()

    def test_main_passes_audio_prompt_to_cfg(self, tmp_path):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        prompt_file = tmp_path / "prompt.wav"
        prompt_file.touch()
        out = str(tmp_path / "out.wav")
        mock_tts = self._mock_sarashina_tts()
        if mock_tts is None:
            pytest.skip("torch not installed")
        mock_sf = MagicMock()
        with patch("sys.argv", [
            "sherox.tts", "--text", "こんにちは。",
            "--lang", "jpn-sarashina",
            "--audio-prompt", str(prompt_file),
            "--audio-prompt-text", "プロンプト。",
            "--output", out,
        ]), \
        patch.object(tts_module, "build_tts", return_value=mock_tts) as mock_build, \
        patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            tts_module.main()
        cfg = mock_build.call_args[0][0]
        assert cfg.audio_prompt == str(prompt_file)
        assert cfg.audio_prompt_text == "プロンプト。"
        assert cfg.language == "jpn-sarashina"


# ---------------------------------------------------------------------------
# Chinese TTS (zho)
# ---------------------------------------------------------------------------

class TestChineseTts:
    """Tests for the Chinese (zho) TTS language entry."""

    def test_zho_in_models_registry(self):
        assert "zho" in tts_module._TTS_MODELS

    def test_zho_model_metadata_has_no_data_dir(self):
        """Chinese model uses lexicon, not espeak data_dir."""
        meta = tts_module._TTS_MODELS["zho"]
        assert meta.get("data_dir") == "" or "data_dir" not in meta

    def test_zho_model_metadata_has_lexicon(self):
        meta = tts_module._TTS_MODELS["zho"]
        assert meta.get("lexicon"), "zho model should have a non-empty lexicon entry"

    def test_zh_alias_normalises_to_zho(self):
        alias = tts_module._LANGUAGE_ALIASES.get("zh")
        assert alias == "zho"

    def test_zh_cn_alias_normalises_to_zho(self):
        assert tts_module._LANGUAGE_ALIASES.get("zh-cn") == "zho"

    def test_zh_tw_alias_normalises_to_zho(self):
        assert tts_module._LANGUAGE_ALIASES.get("zh-tw") == "zho"

    def test_cmn_alias_normalises_to_zho(self):
        assert tts_module._LANGUAGE_ALIASES.get("cmn") == "zho"

    def test_build_tts_zho_passes_lexicon_not_data_dir(self, tmp_path):
        """build_tts must pass lexicon= and data_dir='' for the zho model."""
        meta = tts_module._TTS_MODELS["zho"]
        model_dir = tmp_path / meta["extracted"]
        model_dir.mkdir(parents=True)
        (model_dir / meta["model"]).touch()
        (model_dir / meta["tokens"]).touch()
        (model_dir / meta["lexicon"]).touch()

        mock_sherpa = MagicMock()
        mock_config = MagicMock()
        mock_config.validate.return_value = True
        mock_sherpa.OfflineTtsConfig.return_value = mock_config
        mock_sherpa.OfflineTts.return_value = MagicMock()

        cfg = TtsConfig(language="zho", model_dir=str(model_dir))
        with patch.dict("sys.modules", {"sherpa_onnx": mock_sherpa}):
            tts_module.build_tts(cfg, tmp_path)

        vits_call = mock_sherpa.OfflineTtsVitsModelConfig.call_args
        assert vits_call is not None
        kwargs = vits_call.kwargs if vits_call.kwargs else dict(zip(
            ["model", "lexicon", "data_dir", "tokens"], vits_call.args
        ))
        assert kwargs.get("data_dir", "NOT_SET") == "", (
            "data_dir should be '' for zho (lexicon-based model)"
        )
        assert kwargs.get("lexicon", "") != "", "lexicon should be non-empty for zho"

    def test_build_tts_zho_via_zh_alias(self, tmp_path):
        """Passing --lang zh should resolve to the zho model directory."""
        meta = tts_module._TTS_MODELS["zho"]
        model_dir = tmp_path / meta["extracted"]
        model_dir.mkdir(parents=True)
        (model_dir / meta["model"]).touch()
        (model_dir / meta["tokens"]).touch()
        (model_dir / meta["lexicon"]).touch()

        mock_sherpa = MagicMock()
        mock_config = MagicMock()
        mock_config.validate.return_value = True
        mock_sherpa.OfflineTtsConfig.return_value = mock_config
        mock_sherpa.OfflineTts.return_value = MagicMock()

        cfg = TtsConfig(language="zh", model_dir=str(model_dir))
        with patch.dict("sys.modules", {"sherpa_onnx": mock_sherpa}):
            result = tts_module.build_tts(cfg, tmp_path)
        assert result.backend == "sherpa_onnx"


# ---------------------------------------------------------------------------
# Supertonic-3 TTS
# ---------------------------------------------------------------------------

class TestSupertonicTts:
    """Tests for the Supertonic-3 multi-language TTS model."""

    SUPERTONIC_LANGS = [
        "kor", "ara", "bul", "ces", "dan", "ell", "est", "fin",
        "hin", "hrv", "hun", "ita", "lit", "lav", "nld", "pol",
        "por", "ron", "rus", "slk", "slv", "swe", "tur", "ukr", "vie",
    ]

    def test_supertonic_base_metadata(self):
        meta = tts_module._SUPERTONIC_BASE
        assert meta["backend"] == "supertonic"
        assert meta["sample_rate"] == 24000
        assert "duration_predictor" in meta["files"]
        assert "text_encoder" in meta["files"]
        assert "vocoder" in meta["files"]

    def test_all_supertonic_languages_in_registry(self):
        for code in self.SUPERTONIC_LANGS:
            assert code in tts_module._TTS_MODELS, f"{code} missing from _TTS_MODELS"
            meta = tts_module._TTS_MODELS[code]
            assert meta["backend"] == "supertonic", f"{code} should use supertonic backend"

    def test_supertonic_languages_have_lang_code(self):
        for code in self.SUPERTONIC_LANGS:
            meta = tts_module._TTS_MODELS[code]
            assert "lang_code" in meta, f"{code} missing lang_code"
            assert len(meta["lang_code"]) == 2, f"{code} lang_code should be 2-letter"

    def test_supertonic_languages_share_url(self):
        urls = {tts_module._TTS_MODELS[c]["url"] for c in self.SUPERTONIC_LANGS}
        assert len(urls) == 1, "All supertonic languages should share the same URL"

    def test_supertonic_aliases_resolve(self):
        alias_map = {
            "ko": "kor", "ar": "ara", "bg": "bul", "cs": "ces",
            "da": "dan", "el": "ell", "et": "est", "fi": "fin",
            "hi": "hin", "hr": "hrv", "hu": "hun", "it": "ita",
            "lt": "lit", "lv": "lav", "nl": "nld", "dut": "nld",
            "pl": "pol", "pt": "por", "ro": "ron", "rum": "ron",
            "ru": "rus", "sk": "slk", "sl": "slv", "sv": "swe",
            "tr": "tur", "uk": "ukr", "vi": "vie",
        }
        for alias, expected in alias_map.items():
            assert tts_module._LANGUAGE_ALIASES.get(alias) == expected, (
                f"Alias '{alias}' should resolve to '{expected}'"
            )

    def test_existing_languages_not_overridden(self):
        """Languages with dedicated models should NOT use supertonic."""
        for code in ["eng", "deu", "fra", "spa", "ind", "zho", "jpn"]:
            assert code in tts_module._TTS_MODELS
            meta = tts_module._TTS_MODELS[code]
            assert meta["backend"] != "supertonic", (
                f"{code} should keep its dedicated model, not supertonic"
            )

    def test_indonesian_supertonic_available(self):
        """Indonesian Supertonic-3 (ind-supertonic) should be available."""
        meta = tts_module._TTS_MODELS["ind-supertonic"]
        assert meta["backend"] == "supertonic"
        assert meta["lang_code"] == "id"

    def test_japanese_supertonic_available(self):
        """Japanese Supertonic-3 (jpn-supertonic) should be available."""
        meta = tts_module._TTS_MODELS["jpn-supertonic"]
        assert meta["backend"] == "supertonic"
        assert meta["lang_code"] == "ja"

    def test_japanese_supertonic_alias_resolves(self):
        assert tts_module._normalize_language("jpn-supertonic") == "jpn-supertonic"

    def test_japanese_supertonic_shares_supertonic_url(self):
        meta = tts_module._TTS_MODELS["jpn-supertonic"]
        assert meta["url"] == tts_module._SUPERTONIC_BASE["url"]

    def test_parse_args_supertonic_lang(self):
        with patch("sys.argv", ["sherox.tts", "--lang", "kor"]):
            args = tts_module.parse_args()
        assert args.lang == "kor"

    def test_parse_args_supertonic_lang_alias(self):
        with patch("sys.argv", ["sherox.tts", "--lang", "ko"]):
            args = tts_module.parse_args()
        assert args.lang == "ko"

    def test_build_tts_supertonic(self, tmp_path):
        meta = tts_module._TTS_MODELS["rus"]
        model_dir = tmp_path / "models" / "supertonic" / meta["extracted"]
        model_dir.mkdir(parents=True)
        for f in meta["files"].values():
            (model_dir / f).touch()

        mock_sherpa = MagicMock()
        mock_config = MagicMock()
        mock_config.validate.return_value = True
        mock_sherpa.OfflineTtsConfig.return_value = mock_config
        mock_sherpa.OfflineTts.return_value = MagicMock()

        cfg = TtsConfig(language="rus")
        with patch.dict("sys.modules", {"sherpa_onnx": mock_sherpa}):
            result = tts_module.build_tts(cfg, tmp_path)
        assert result.backend == "supertonic"
        assert result.lang_code == "ru"
        assert result.sample_rate == 24000

    def test_synthesise_to_file_supertonic(self):
        mock_sf = MagicMock()
        mock_tts_instance = MagicMock()
        mock_audio = MagicMock()
        mock_audio.samples = [0.1, 0.2, 0.3]
        mock_tts_instance.generate.return_value = mock_audio

        tts = SimpleNamespace(
            backend="supertonic",
            model=mock_tts_instance,
            lang_code="ko",
            sample_rate=24000,
        )
        cfg = TtsConfig(output="out.wav", language="kor")
        with patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            samples, sr = tts_module.synthesise_to_file(tts, "안녕하세요", cfg)
        assert sr == 24000
        assert samples.dtype == np.float32
        mock_sf.write.assert_called_once()

    def test_synthesise_to_file_supertonic_passes_lang(self):
        mock_sf = MagicMock()
        mock_tts_instance = MagicMock()
        mock_audio = MagicMock()
        mock_audio.samples = [0.1, 0.2]
        mock_tts_instance.generate.return_value = mock_audio

        tts = SimpleNamespace(
            backend="supertonic",
            model=mock_tts_instance,
            lang_code="ja",
            sample_rate=24000,
        )
        cfg = TtsConfig(output="out.wav", language="jpn")
        with patch.object(tts_module, "_require_soundfile", return_value=mock_sf):
            tts_module.synthesise_to_file(tts, "こんにちは", cfg)

        call_args = mock_tts_instance.generate.call_args
        gen_config = call_args[0][1]
        assert gen_config.extra == {"lang": "ja"}
