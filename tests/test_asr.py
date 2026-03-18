from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sherox.asr as main_module


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_mic_mode(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.mic is True
        assert args.wav is None

    def test_wav_mode(self):
        with patch("sys.argv", ["sherox.asr", "--wav", "audio.wav"]):
            args = main_module.parse_args()
        assert args.wav == "audio.wav"
        assert args.mic is False

    def test_mic_and_wav_are_mutually_exclusive(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--wav", "audio.wav"]):
            with pytest.raises(SystemExit):
                main_module.parse_args()

    def test_requires_mic_or_wav(self):
        with patch("sys.argv", ["sherox.asr"]):
            with pytest.raises(SystemExit):
                main_module.parse_args()

    def test_default_sample_rate(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.sample_rate == 16000

    def test_default_chunk_size(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.chunk_size == 0.16

    def test_default_threads(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.threads == 4

    def test_default_model_type_is_empty(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.model_type == ""

    def test_default_offline_is_false(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.offline is False

    def test_default_language(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.language == "en"

    def test_default_listening_is_false(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.listening is False

    def test_custom_model_type(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--model-type", "whisper"]):
            args = main_module.parse_args()
        assert args.model_type == "whisper"

    def test_offline_flag(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--offline"]):
            args = main_module.parse_args()
        assert args.offline is True

    def test_custom_language(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--language", "zh"]):
            args = main_module.parse_args()
        assert args.language == "zh"

    def test_custom_threads(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--threads", "8"]):
            args = main_module.parse_args()
        assert args.threads == 8

    def test_custom_sample_rate(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--sample-rate", "48000"]):
            args = main_module.parse_args()
        assert args.sample_rate == 48000

    def test_listening_flag(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--listening"]):
            args = main_module.parse_args()
        assert args.listening is True

    def test_custom_model_dir(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--model-dir", "models/custom"]):
            args = main_module.parse_args()
        assert args.model_dir == "models/custom"

    def test_default_model_dir_is_none(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.model_dir is None

    def test_default_vad_model_is_silero(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.vad_type == "silero"

    def test_vad_model_ten_vad(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--vad-model", "ten-vad"]):
            args = main_module.parse_args()
        assert args.vad_type == "ten-vad"

    def test_vad_model_invalid_choice_exits(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--vad-model", "invalid"]):
            with pytest.raises(SystemExit):
                main_module.parse_args()

    def test_default_ten_vad_model_is_int8(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.ten_vad_model == "ten-vad.int8.onnx"

    def test_ten_vad_model_full_onnx(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--ten-vad-model", "ten-vad.onnx"]):
            args = main_module.parse_args()
        assert args.ten_vad_model == "ten-vad.onnx"


# ---------------------------------------------------------------------------
# _validate_vad
# ---------------------------------------------------------------------------

class TestValidateVad:
    def test_returns_empty_when_not_offline_silero(self):
        result = main_module._validate_vad("silero", "ten-vad.int8.onnx", False, Path("/proj"))
        assert result == ""

    def test_returns_empty_when_not_offline_ten_vad(self):
        result = main_module._validate_vad("ten-vad", "ten-vad.int8.onnx", False, Path("/proj"))
        assert result == ""

    def test_downloads_silero_when_offline_and_no_file(self, tmp_path):
        with patch.object(main_module, "_download_file") as mock_dl:
            result = main_module._validate_vad("silero", "ten-vad.int8.onnx", True, tmp_path)

        expected_path = str(tmp_path / "models" / "silero_vad.onnx")
        assert result == expected_path
        mock_dl.assert_called_once()

    def test_does_not_download_silero_when_already_exists(self, tmp_path):
        vad_path = tmp_path / "models" / "silero_vad.onnx"
        vad_path.parent.mkdir()
        vad_path.touch()

        with patch.object(main_module, "_download_file") as mock_dl:
            result = main_module._validate_vad("silero", "ten-vad.int8.onnx", True, tmp_path)

        mock_dl.assert_not_called()
        assert result == str(vad_path)

    def test_downloads_ten_vad_int8_when_offline_and_no_file(self, tmp_path):
        with patch.object(main_module, "_download_file") as mock_dl:
            result = main_module._validate_vad("ten-vad", "ten-vad.int8.onnx", True, tmp_path)

        expected_path = str(tmp_path / "models" / "ten-vad.int8.onnx")
        assert result == expected_path
        mock_dl.assert_called_once()

    def test_downloads_ten_vad_onnx_variant_when_specified(self, tmp_path):
        with patch.object(main_module, "_download_file") as mock_dl:
            result = main_module._validate_vad("ten-vad", "ten-vad.onnx", True, tmp_path)

        expected_path = str(tmp_path / "models" / "ten-vad.onnx")
        assert result == expected_path
        mock_dl.assert_called_once()
        url_used = mock_dl.call_args[0][0]
        assert "ten-vad.onnx" in url_used
        assert "int8" not in url_used

    def test_does_not_download_ten_vad_when_already_exists(self, tmp_path):
        vad_path = tmp_path / "models" / "ten-vad.int8.onnx"
        vad_path.parent.mkdir()
        vad_path.touch()

        with patch.object(main_module, "_download_file") as mock_dl:
            main_module._validate_vad("ten-vad", "ten-vad.int8.onnx", True, tmp_path)

        mock_dl.assert_not_called()

    def test_ten_vad_url_differs_between_variants(self):
        int8_url = main_module._TEN_VAD_MODEL_URLS["ten-vad.int8.onnx"]
        full_url = main_module._TEN_VAD_MODEL_URLS["ten-vad.onnx"]
        assert int8_url != full_url
        assert "int8" in int8_url
        assert "int8" not in full_url

    def test_unknown_vad_type_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            main_module._validate_vad("unknown-vad", "ten-vad.int8.onnx", True, tmp_path)

    def test_unknown_ten_vad_model_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            main_module._validate_vad("ten-vad", "unknown-model.onnx", True, tmp_path)


# ---------------------------------------------------------------------------
# _validate_wav
# ---------------------------------------------------------------------------

def _mock_sf(channels: int, samplerate: int):
    mock_f = MagicMock()
    mock_f.__enter__ = MagicMock(return_value=mock_f)
    mock_f.__exit__ = MagicMock(return_value=False)
    mock_f.channels = channels
    mock_f.samplerate = samplerate
    return mock_f


class TestValidateWav:
    def test_exits_when_file_not_found(self, tmp_path):
        with pytest.raises(SystemExit):
            main_module._validate_wav(str(tmp_path / "missing.wav"), 16000)

    def test_exits_on_multichannel_audio(self, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.touch()
        with patch("sherox.asr.sf.SoundFile", return_value=_mock_sf(2, 16000)):
            with pytest.raises(SystemExit):
                main_module._validate_wav(str(wav), 16000)

    def test_exits_on_wrong_sample_rate(self, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.touch()
        with patch("sherox.asr.sf.SoundFile", return_value=_mock_sf(1, 44100)):
            with pytest.raises(SystemExit):
                main_module._validate_wav(str(wav), 16000)

    def test_passes_for_valid_mono_16khz_audio(self, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.touch()
        with patch("sherox.asr.sf.SoundFile", return_value=_mock_sf(1, 16000)):
            main_module._validate_wav(str(wav), 16000)  # should not raise

    def test_exits_when_soundfile_raises(self, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.touch()
        with patch("sherox.asr.sf.SoundFile", side_effect=Exception("corrupt")):
            with pytest.raises(SystemExit):
                main_module._validate_wav(str(wav), 16000)


# ---------------------------------------------------------------------------
# _validate_model
# ---------------------------------------------------------------------------

class TestValidateModel:
    def test_does_nothing_when_dir_exists(self, tmp_path):
        with patch.object(main_module, "_download_model") as mock_dl:
            main_module._validate_model(str(tmp_path), "")
        mock_dl.assert_not_called()

    def test_downloads_when_dir_missing(self, tmp_path):
        missing = str(tmp_path / "no_such_dir")
        with patch.object(main_module, "_download_model") as mock_dl:
            main_module._validate_model(missing, "zipformer2")
        mock_dl.assert_called_once_with(missing, "zipformer2")


# ---------------------------------------------------------------------------
# _download_model  (URL-selection logic)
# ---------------------------------------------------------------------------

def _run_download_model(tmp_path, model_dir_name: str, model_type: str):
    """Helper: run _download_model with mocked I/O, return the URL passed to _download_file."""
    model_dir = tmp_path / model_dir_name
    # Create the extracted directory so the rename step succeeds
    extracted_name = _extracted_name_for(model_dir_name, model_type)
    extracted_dir = tmp_path / extracted_name
    extracted_dir.mkdir()

    captured_url = {}

    def fake_download(url, dest):
        captured_url["url"] = url

    with patch.object(main_module, "_download_file", side_effect=fake_download), \
         patch("tarfile.open") as mock_tar:
        mock_tar.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_tar.return_value.__exit__ = MagicMock(return_value=False)
        main_module._download_model(str(model_dir), model_type)

    return captured_url["url"]


def _extracted_name_for(model_dir_name: str, model_type: str) -> str:
    if model_type == "nemo_transducer" or model_dir_name in (
        main_module._PARAKEET_FP16_TARGET,
        main_module._PARAKEET_INT8_TARGET,
    ):
        if "int8" in model_dir_name:
            return main_module._PARAKEET_INT8_EXTRACTED
        return main_module._PARAKEET_FP16_EXTRACTED
    return main_module._MODEL_EXTRACTED


class TestDownloadModel:
    def test_uses_parakeet_fp16_url_for_nemo_transducer(self, tmp_path):
        url = _run_download_model(tmp_path, main_module._PARAKEET_FP16_TARGET, "nemo_transducer")
        assert "parakeet" in url
        assert "int8" not in url

    def test_uses_parakeet_int8_url_for_int8_dir(self, tmp_path):
        url = _run_download_model(tmp_path, main_module._PARAKEET_INT8_TARGET, "nemo_transducer")
        assert "int8" in url

    def test_uses_zipformer_url_for_default_model_type(self, tmp_path):
        url = _run_download_model(tmp_path, main_module._MODEL_TARGET, "")
        assert "zipformer" in url

    def test_archive_deleted_after_extraction(self, tmp_path):
        model_dir = tmp_path / main_module._MODEL_TARGET
        extracted_dir = tmp_path / main_module._MODEL_EXTRACTED
        extracted_dir.mkdir()
        archive = tmp_path / main_module._MODEL_ARCHIVE

        with patch.object(main_module, "_download_file"), \
             patch("tarfile.open") as mock_tar:
            mock_tar.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_tar.return_value.__exit__ = MagicMock(return_value=False)
            main_module._download_model(str(model_dir), "")

        # archive was never created (download mocked), so missing_ok=True handled it
        assert not archive.exists()

    def test_exits_when_extracted_dir_not_found(self, tmp_path, capsys):
        model_dir = tmp_path / main_module._MODEL_TARGET

        with patch.object(main_module, "_download_file"), \
             patch("tarfile.open") as mock_tar, \
             pytest.raises(SystemExit):
            mock_tar.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_tar.return_value.__exit__ = MagicMock(return_value=False)
            main_module._download_model(str(model_dir), "")
