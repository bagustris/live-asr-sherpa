from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import asr_engine
from config import Config


# ---------------------------------------------------------------------------
# _find
# ---------------------------------------------------------------------------

class TestFind:
    def test_finds_matching_file(self, tmp_path):
        f = tmp_path / "encoder.onnx"
        f.touch()
        assert asr_engine._find(tmp_path, "encoder*.onnx") == str(f)

    def test_raises_file_not_found_when_no_match(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="encoder"):
            asr_engine._find(tmp_path, "encoder*.onnx")

    def test_returns_first_match_when_multiple(self, tmp_path):
        (tmp_path / "encoder1.onnx").touch()
        (tmp_path / "encoder2.onnx").touch()
        result = asr_engine._find(tmp_path, "encoder*.onnx")
        assert "encoder1.onnx" in result

    def test_error_message_includes_pattern(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="joiner\\*.onnx"):
            asr_engine._find(tmp_path, "joiner*.onnx")


# ---------------------------------------------------------------------------
# build_vad
# ---------------------------------------------------------------------------

class TestBuildVad:
    def test_raises_when_vad_model_is_empty(self):
        cfg = Config(vad_model="")
        with pytest.raises(ValueError, match="--vad-model"):
            asr_engine.build_vad(cfg)

    def test_builds_vad_when_model_provided(self, tmp_path):
        vad_path = tmp_path / "silero_vad.onnx"
        vad_path.touch()
        cfg = Config(vad_model=str(vad_path), sample_rate=16000, num_threads=2,
                     vad_threshold=0.5, vad_min_silence_duration=0.5,
                     vad_min_speech_duration=0.25)

        mock_vad = MagicMock()
        with patch("asr_engine.sherpa_onnx.VadModelConfig"), \
             patch("asr_engine.sherpa_onnx.SileroVadModelConfig"), \
             patch("asr_engine.sherpa_onnx.VoiceActivityDetector", return_value=mock_vad):
            result = asr_engine.build_vad(cfg)

        assert result is mock_vad

    def test_vad_error_message_includes_download_hint(self):
        cfg = Config(vad_model="")
        with pytest.raises(ValueError, match="silero_vad.onnx"):
            asr_engine.build_vad(cfg)

    def test_vad_config_receives_correct_sample_rate(self, tmp_path):
        vad_path = tmp_path / "vad.onnx"
        vad_path.touch()
        cfg = Config(vad_model=str(vad_path), sample_rate=48000, num_threads=4,
                     vad_threshold=0.6)

        captured = {}

        def capture_vad_config(silero_vad, sample_rate, num_threads):
            captured["sample_rate"] = sample_rate
            return MagicMock()

        with patch("asr_engine.sherpa_onnx.SileroVadModelConfig"), \
             patch("asr_engine.sherpa_onnx.VadModelConfig",
                   side_effect=capture_vad_config), \
             patch("asr_engine.sherpa_onnx.VoiceActivityDetector"):
            asr_engine.build_vad(cfg)

        assert captured["sample_rate"] == 48000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _touch_files(directory: Path, *names: str) -> None:
    for name in names:
        (directory / name).touch()


# ---------------------------------------------------------------------------
# build_recognizer  (online / streaming)
# ---------------------------------------------------------------------------

class TestBuildRecognizer:
    def test_default_routes_to_transducer(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="")

        mock_rec = MagicMock()
        with patch("asr_engine.sherpa_onnx.OnlineRecognizer") as mock_cls:
            mock_cls.from_transducer.return_value = mock_rec
            result = asr_engine.build_recognizer(cfg)

        mock_cls.from_transducer.assert_called_once()
        assert result is mock_rec

    def test_transducer_passes_model_type(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="zipformer2")

        with patch("asr_engine.sherpa_onnx.OnlineRecognizer") as mock_cls:
            mock_cls.from_transducer.return_value = MagicMock()
            asr_engine.build_recognizer(cfg)

        _, kwargs = mock_cls.from_transducer.call_args
        assert kwargs["model_type"] == "zipformer2"

    def test_paraformer_route(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "encoder.onnx", "decoder.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="paraformer")

        with patch("asr_engine.sherpa_onnx.OnlineRecognizer") as mock_cls:
            mock_cls.from_paraformer.return_value = MagicMock()
            asr_engine.build_recognizer(cfg)

        mock_cls.from_paraformer.assert_called_once()

    def test_wenet_ctc_route(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "model.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="wenet_ctc")

        with patch("asr_engine.sherpa_onnx.OnlineRecognizer") as mock_cls:
            mock_cls.from_wenet_ctc.return_value = MagicMock()
            asr_engine.build_recognizer(cfg)

        mock_cls.from_wenet_ctc.assert_called_once()

    def test_zipformer2_ctc_route(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "model.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="zipformer2_ctc")

        with patch("asr_engine.sherpa_onnx.OnlineRecognizer") as mock_cls:
            mock_cls.from_zipformer2_ctc.return_value = MagicMock()
            asr_engine.build_recognizer(cfg)

        mock_cls.from_zipformer2_ctc.assert_called_once()

    def test_ctc_route(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "model.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="ctc")

        with patch("asr_engine.sherpa_onnx.OnlineRecognizer") as mock_cls:
            mock_cls.from_ctc.return_value = MagicMock()
            asr_engine.build_recognizer(cfg)

        mock_cls.from_ctc.assert_called_once()

    def test_model_type_is_lowercased(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "model.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="CTC")

        with patch("asr_engine.sherpa_onnx.OnlineRecognizer") as mock_cls:
            mock_cls.from_ctc.return_value = MagicMock()
            asr_engine.build_recognizer(cfg)

        mock_cls.from_ctc.assert_called_once()


# ---------------------------------------------------------------------------
# build_offline_recognizer
# ---------------------------------------------------------------------------

class TestBuildOfflineRecognizer:
    def test_default_routes_to_transducer(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="")

        with patch("asr_engine.sherpa_onnx.OfflineRecognizer") as mock_cls:
            mock_cls.from_transducer.return_value = MagicMock()
            asr_engine.build_offline_recognizer(cfg)

        mock_cls.from_transducer.assert_called_once()

    def test_whisper_route(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "encoder.onnx", "decoder.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="whisper", language="en")

        with patch("asr_engine.sherpa_onnx.OfflineRecognizer") as mock_cls:
            mock_cls.from_whisper.return_value = MagicMock()
            asr_engine.build_offline_recognizer(cfg)

        mock_cls.from_whisper.assert_called_once()

    def test_whisper_passes_language(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "encoder.onnx", "decoder.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="whisper", language="zh")

        with patch("asr_engine.sherpa_onnx.OfflineRecognizer") as mock_cls:
            mock_cls.from_whisper.return_value = MagicMock()
            asr_engine.build_offline_recognizer(cfg)

        _, kwargs = mock_cls.from_whisper.call_args
        assert kwargs["language"] == "zh"

    def test_paraformer_route(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "model.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="paraformer")

        with patch("asr_engine.sherpa_onnx.OfflineRecognizer") as mock_cls:
            mock_cls.from_paraformer.return_value = MagicMock()
            asr_engine.build_offline_recognizer(cfg)

        mock_cls.from_paraformer.assert_called_once()

    def test_ctc_route(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "model.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="ctc")

        with patch("asr_engine.sherpa_onnx.OfflineRecognizer") as mock_cls:
            mock_cls.from_ctc.return_value = MagicMock()
            asr_engine.build_offline_recognizer(cfg)

        mock_cls.from_ctc.assert_called_once()

    def test_nemo_ctc_route(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "model.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="nemo_ctc")

        with patch("asr_engine.sherpa_onnx.OfflineRecognizer") as mock_cls:
            mock_cls.from_ctc.return_value = MagicMock()
            asr_engine.build_offline_recognizer(cfg)

        mock_cls.from_ctc.assert_called_once()

    def test_sense_voice_route(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "model.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="sense_voice")

        with patch("asr_engine.sherpa_onnx.OfflineRecognizer") as mock_cls:
            mock_cls.from_sense_voice.return_value = MagicMock()
            asr_engine.build_offline_recognizer(cfg)

        mock_cls.from_sense_voice.assert_called_once()

    def test_moonshine_route(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "preprocess.onnx", "encode.onnx",
                     "uncached_decoder.onnx", "cached_decoder.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="moonshine")

        with patch("asr_engine.sherpa_onnx.OfflineRecognizer") as mock_cls:
            mock_cls.from_moonshine.return_value = MagicMock()
            asr_engine.build_offline_recognizer(cfg)

        mock_cls.from_moonshine.assert_called_once()

    def test_fire_red_asr_route(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "encoder.onnx", "decoder.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="fire_red_asr")

        with patch("asr_engine.sherpa_onnx.OfflineRecognizer") as mock_cls:
            mock_cls.from_fire_red_asr.return_value = MagicMock()
            asr_engine.build_offline_recognizer(cfg)

        mock_cls.from_fire_red_asr.assert_called_once()

    def test_model_type_is_lowercased(self, tmp_path):
        _touch_files(tmp_path, "tokens.txt", "encoder.onnx", "decoder.onnx")
        cfg = Config(model_dir=str(tmp_path), model_type="WHISPER", language="en")

        with patch("asr_engine.sherpa_onnx.OfflineRecognizer") as mock_cls:
            mock_cls.from_whisper.return_value = MagicMock()
            asr_engine.build_offline_recognizer(cfg)

        mock_cls.from_whisper.assert_called_once()
