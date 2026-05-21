import argparse
import urllib.request
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
        assert args.wav == ["audio.wav"]
        assert args.mic is False

    def test_wav_multiple_files(self):
        with patch("sys.argv", ["sherox.asr", "--wav", "a.wav", "b.wav"]):
            args = main_module.parse_args()
        assert args.wav == ["a.wav", "b.wav"]

    def test_mic_and_wav_are_mutually_exclusive(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--wav", "audio.wav"]):
            with pytest.raises(SystemExit):
                main_module.parse_args()

    def test_pipe_mode(self):
        with patch("sys.argv", ["sherox.asr", "--pipe"]):
            args = main_module.parse_args()
        assert args.pipe is True
        assert args.mic is False
        assert args.wav is None

    def test_pipe_and_mic_are_mutually_exclusive(self):
        with patch("sys.argv", ["sherox.asr", "--pipe", "--mic"]):
            with pytest.raises(SystemExit):
                main_module.parse_args()

    def test_pipe_and_wav_are_mutually_exclusive(self):
        with patch("sys.argv", ["sherox.asr", "--pipe", "--wav", "audio.wav"]):
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

    def test_lang_alias(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--lang", "jp"]):
            args = main_module.parse_args()
        assert args.language == "jp"

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

    def test_final_only_flag(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--final-only"]):
            args = main_module.parse_args()
        assert args.final_only is True

    def test_final_only_default_false(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.final_only is False

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

    def test_default_device_is_cpu(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.device == "cpu"

    def test_device_cuda(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--device", "cuda"]):
            args = main_module.parse_args()
        assert args.device == "cuda"

    def test_default_denoise_is_false(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.denoise is False

    def test_denoise_flag(self):
        with patch("sys.argv", ["sherox.asr", "--wav", "audio.wav", "--denoise"]):
            args = main_module.parse_args()
        assert args.denoise is True

    def test_default_word_timestamps_is_false(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.word_timestamps is False

    def test_word_timestamps_flag(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--word-timestamps"]):
            args = main_module.parse_args()
        assert args.word_timestamps is True

    def test_default_punctuation_is_false(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.punctuation is False

    def test_punctuation_flag(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--punctuation"]):
            args = main_module.parse_args()
        assert args.punctuation is True

    def test_default_output_is_empty(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.output == ""

    def test_output_format_choices(self):
        for fmt in ("srt", "vtt", "txt"):
            with patch("sys.argv", ["sherox.asr", "--mic", "--output-format", fmt]):
                args = main_module.parse_args()
            assert args.output_format == fmt

    def test_output_format_invalid_exits(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--output-format", "pdf"]):
            with pytest.raises(SystemExit):
                main_module.parse_args()

    def test_default_translate_is_false(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.translate is False

    def test_translate_flag(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--translate"]):
            args = main_module.parse_args()
        assert args.translate is True

    def test_default_no_color_is_false(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.no_color is False

    def test_no_color_flag(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--no-color"]):
            args = main_module.parse_args()
        assert args.no_color is True

    def test_default_json_output_is_false(self):
        with patch("sys.argv", ["sherox.asr", "--mic"]):
            args = main_module.parse_args()
        assert args.json_output is False

    def test_json_flag(self):
        with patch("sys.argv", ["sherox.asr", "--mic", "--json"]):
            args = main_module.parse_args()
        assert args.json_output is True

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

    def test_warns_on_wrong_sample_rate(self, tmp_path, capsys):
        wav = tmp_path / "audio.wav"
        wav.touch()
        with patch("sherox.asr.sf.SoundFile", return_value=_mock_sf(1, 44100)):
            main_module._validate_wav(str(wav), 16000)  # should NOT raise
        assert "resampling" in capsys.readouterr().out

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
# _validate_runtime_args
# ---------------------------------------------------------------------------

class TestValidateRuntimeArgs:
    def _base_args(self, **overrides):
        defaults = dict(
            sample_rate=16000,
            capture_rate=16000,
            chunk_size=0.16,
            threads=4,
            speaker_tag=False,
            diarization=False,
            num_speakers=-1,
            denoise=False,
            wav=None,
            pipe=False,
            output="",
            output_dir="",
            translate=False,
            offline=False,
            model_type="",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_speaker_tag_requires_diarization(self):
        args = self._base_args(speaker_tag=True, diarization=False)
        with pytest.raises(SystemExit):
            main_module._validate_runtime_args(args)

    def test_num_speakers_zero_is_rejected(self):
        args = self._base_args(diarization=True, num_speakers=0)
        with pytest.raises(SystemExit):
            main_module._validate_runtime_args(args)

    def test_denoise_requires_wav(self):
        args = self._base_args(denoise=True, wav=None)
        with pytest.raises(SystemExit):
            main_module._validate_runtime_args(args)

    def test_denoise_blocked_with_pipe(self):
        args = self._base_args(denoise=True, pipe=True)
        with pytest.raises(SystemExit):
            main_module._validate_runtime_args(args)

    def test_output_with_multiple_wav_exits(self):
        args = self._base_args(wav=["a.wav", "b.wav"], output="/out.srt")
        with pytest.raises(SystemExit):
            main_module._validate_runtime_args(args)

    def test_output_with_single_wav_ok(self):
        args = self._base_args(wav=["a.wav"], output="/out.srt")
        main_module._validate_runtime_args(args)  # must not raise

    def test_output_dir_without_wav_exits(self):
        args = self._base_args(wav=None, output_dir="/some/dir")
        with pytest.raises(SystemExit):
            main_module._validate_runtime_args(args)

    def test_output_dir_with_pipe_exits(self):
        args = self._base_args(pipe=True, output_dir="/some/dir")
        with pytest.raises(SystemExit):
            main_module._validate_runtime_args(args)

    def test_translate_requires_offline(self):
        args = self._base_args(translate=True, offline=False, model_type="whisper")
        with pytest.raises(SystemExit):
            main_module._validate_runtime_args(args)

    def test_translate_requires_whisper_model_type(self):
        args = self._base_args(translate=True, offline=True, model_type="")
        with pytest.raises(SystemExit):
            main_module._validate_runtime_args(args)

    def test_translate_with_sense_voice_exits(self):
        args = self._base_args(translate=True, offline=True, model_type="sense_voice")
        with pytest.raises(SystemExit):
            main_module._validate_runtime_args(args)

    def test_translate_with_offline_whisper_ok(self):
        args = self._base_args(translate=True, offline=True, model_type="whisper")
        main_module._validate_runtime_args(args)  # must not raise

    def test_translate_false_does_not_validate(self):
        # When translate=False the model_type check must not be enforced.
        args = self._base_args(translate=False, offline=False, model_type="")
        main_module._validate_runtime_args(args)  # must not raise


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
    if model_type == "ja" or model_dir_name == main_module._REAZON_JA_TARGET:
        return main_module._REAZON_JA_EXTRACTED
    if model_type in ("ja-en", "ja-en-mls-5k") or model_dir_name in (
        main_module._REAZON_JA_EN_TARGET,
        main_module._REAZON_JA_EN_MLS_TARGET,
    ):
        return main_module._REAZON_JA_EN_EXTRACTED
    if model_type == "parakeet-ctc-ja" or model_dir_name == main_module._PARAKEET_CTC_JA_INT8_TARGET:
        return main_module._PARAKEET_CTC_JA_INT8_EXTRACTED
    if model_type == "cohere_transcribe" or model_dir_name == main_module._COHERE_TRANSCRIBE_TARGET:
        return main_module._COHERE_TRANSCRIBE_EXTRACTED
    if model_type == "multilingual_streaming" or model_dir_name == main_module._MULTILINGUAL_STREAMING_TARGET:
        return main_module._MULTILINGUAL_STREAMING_EXTRACTED
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

    def test_uses_reazon_ja_url_for_ja_model_type(self, tmp_path):
        url = _run_download_model(tmp_path, main_module._REAZON_JA_TARGET, "ja")
        assert "reazonspeech" in url
        assert "ja-en" not in url

    def test_uses_reazon_ja_url_for_ja_dir_name(self, tmp_path):
        url = _run_download_model(tmp_path, main_module._REAZON_JA_TARGET, "")
        assert "reazonspeech" in url
        assert "ja-en" not in url

    def test_uses_reazon_ja_en_url_for_ja_en_model_type(self, tmp_path):
        url = _run_download_model(tmp_path, main_module._REAZON_JA_EN_TARGET, "ja-en")
        assert "reazonspeech" in url
        assert "ja-en" in url

    def test_uses_reazon_ja_en_url_for_ja_en_mls_5k_model_type(self, tmp_path):
        url = _run_download_model(tmp_path, main_module._REAZON_JA_EN_MLS_TARGET, "ja-en-mls-5k")
        assert "reazonspeech" in url
        assert "ja-en" in url

    def test_uses_cohere_transcribe_url_for_cohere_transcribe_model_type(self, tmp_path):
        url = _run_download_model(tmp_path, main_module._COHERE_TRANSCRIBE_TARGET, "cohere_transcribe")
        assert "cohere" in url

    def test_uses_cohere_transcribe_url_for_cohere_transcribe_dir_name(self, tmp_path):
        url = _run_download_model(tmp_path, main_module._COHERE_TRANSCRIBE_TARGET, "")
        assert "cohere" in url

    def test_uses_multilingual_streaming_url_for_multilingual_streaming_model_type(self, tmp_path):
        url = _run_download_model(tmp_path, main_module._MULTILINGUAL_STREAMING_TARGET, "multilingual_streaming")
        assert "multilingual" in url or "ar_en_id" in url

    def test_uses_multilingual_streaming_url_for_multilingual_streaming_dir_name(self, tmp_path):
        url = _run_download_model(tmp_path, main_module._MULTILINGUAL_STREAMING_TARGET, "")
        assert "ar_en_id" in url

    def test_reazon_ja_and_ja_en_use_different_urls(self, tmp_path):
        ja_url = _run_download_model(tmp_path, main_module._REAZON_JA_TARGET, "ja")
        ja_en_url = _run_download_model(tmp_path, main_module._REAZON_JA_EN_TARGET, "ja-en")
        assert ja_url != ja_en_url

    def test_uses_parakeet_ctc_ja_url_for_parakeet_ctc_ja_model_type(self, tmp_path):
        url = _run_download_model(tmp_path, main_module._PARAKEET_CTC_JA_INT8_TARGET, "parakeet-ctc-ja")
        assert "parakeet" in url
        assert "ja" in url
        assert "int8" in url

    def test_uses_parakeet_ctc_ja_url_for_parakeet_ctc_ja_dir_name(self, tmp_path):
        url = _run_download_model(tmp_path, main_module._PARAKEET_CTC_JA_INT8_TARGET, "")
        assert "parakeet" in url
        assert "ja" in url


# ---------------------------------------------------------------------------
# _require_soundfile — success path
# ---------------------------------------------------------------------------

class TestRequireSoundfile:
    def test_imports_when_sentinel_is_none(self):
        import types
        fake_sf = MagicMock()
        fake_sf.SoundFile = MagicMock()
        initial = types.SimpleNamespace(SoundFile=None)
        with patch.object(main_module, "sf", initial):
            with patch.dict("sys.modules", {"soundfile": fake_sf}):
                result = main_module._require_soundfile()
        assert result is fake_sf

    def test_returns_early_when_already_loaded(self):
        fake_sf = MagicMock()
        fake_sf.SoundFile = MagicMock()
        with patch.object(main_module, "sf", fake_sf):
            result = main_module._require_soundfile()
        assert result is fake_sf


# ---------------------------------------------------------------------------
# _validate_runtime_args — numeric checks
# ---------------------------------------------------------------------------

class TestValidateRuntimeArgsNumeric:
    def _base_args(self, **overrides):
        import argparse
        args = argparse.Namespace(
            sample_rate=16000,
            capture_rate=16000,
            chunk_size=0.16,
            threads=4,
            speaker_tag=False,
            diarization=False,
            num_speakers=-1,
        )
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_zero_sample_rate_exits(self):
        with pytest.raises(SystemExit):
            main_module._validate_runtime_args(self._base_args(sample_rate=0))

    def test_negative_sample_rate_exits(self):
        with pytest.raises(SystemExit):
            main_module._validate_runtime_args(self._base_args(sample_rate=-1))

    def test_zero_capture_rate_exits(self):
        with pytest.raises(SystemExit):
            main_module._validate_runtime_args(self._base_args(capture_rate=0))

    def test_zero_chunk_size_exits(self):
        with pytest.raises(SystemExit):
            main_module._validate_runtime_args(self._base_args(chunk_size=0))

    def test_zero_threads_exits(self):
        with pytest.raises(SystemExit):
            main_module._validate_runtime_args(self._base_args(threads=0))


# ---------------------------------------------------------------------------
# _download_file
# ---------------------------------------------------------------------------

class TestDownloadFile:
    def test_calls_urlopen_with_correct_url(self, tmp_path):
        dest = tmp_path / "file.tar.bz2"
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "100"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b""]

        with patch("sherox.utils.urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            main_module._download_file("http://example.com/file.tar.bz2", dest)

        mock_urlopen.assert_called_once()
        args = mock_urlopen.call_args[0]
        assert isinstance(args[0], urllib.request.Request)
        assert args[0].full_url == "http://example.com/file.tar.bz2"

    def test_exits_when_download_fails(self, tmp_path):
        dest = tmp_path / "file.tar.bz2"
        with patch("sherox.utils.urllib.request.urlopen", side_effect=Exception("network error")):
            with pytest.raises(SystemExit):
                main_module._download_file("http://example.com/file.tar.bz2", dest)

    def test_progress_writes_percentage(self, tmp_path, capsys):
        dest = tmp_path / "file.tar.bz2"
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "10240"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        # Simulate downloading 5120 bytes (50%)
        mock_response.read.side_effect = [b"x" * 5120, b""]

        with patch("sherox.utils.urllib.request.urlopen", return_value=mock_response):
            main_module._download_file("http://example.com/file.tar.bz2", dest)

        out = capsys.readouterr().out
        assert "50" in out

    def test_progress_skips_when_total_zero(self, tmp_path, capsys):
        dest = tmp_path / "file.tar.bz2"
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "0"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b""]

        with patch("sherox.utils.urllib.request.urlopen", return_value=mock_response):
            main_module._download_file("http://example.com/file.tar.bz2", dest)

        # Drain info messages printed by _download_file itself
        capsys.readouterr()
        # No percentage should be printed since total is 0
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# _safe_tar_members
# ---------------------------------------------------------------------------

class TestSafeTarMembers:
    def test_yields_safe_member(self, tmp_path):
        member = MagicMock()
        member.name = "safe_file.txt"
        member.isdev.return_value = False
        tf = MagicMock()
        tf.getmembers.return_value = [member]
        result = list(main_module._safe_tar_members(tf, tmp_path))
        assert member in result

    def test_skips_device_file(self, tmp_path):
        member = MagicMock()
        member.name = "safe_file.txt"
        member.isdev.return_value = True
        tf = MagicMock()
        tf.getmembers.return_value = [member]
        result = list(main_module._safe_tar_members(tf, tmp_path))
        assert result == []

    def test_skips_path_traversal(self, tmp_path):
        member = MagicMock()
        member.name = "../etc/passwd"
        member.isdev.return_value = False
        tf = MagicMock()
        tf.getmembers.return_value = [member]
        result = list(main_module._safe_tar_members(tf, tmp_path))
        assert result == []

    def test_yields_nested_safe_member(self, tmp_path):
        member = MagicMock()
        member.name = "subdir/file.txt"
        member.isdev.return_value = False
        tf = MagicMock()
        tf.getmembers.return_value = [member]
        result = list(main_module._safe_tar_members(tf, tmp_path))
        assert member in result


# ---------------------------------------------------------------------------
# _safe_extract_tar
# ---------------------------------------------------------------------------

class TestSafeExtractTar:
    def test_extracts_safe_member(self, tmp_path):
        member = MagicMock()
        member.name = "safe_file.txt"
        member.issym.return_value = False
        member.islnk.return_value = False
        tar = MagicMock()
        tar.getmembers.return_value = [member]
        main_module._safe_extract_tar(tar, tmp_path)
        tar.extract.assert_called_once_with(member, path=tmp_path.resolve())

    def test_skips_symlinks(self, tmp_path):
        member = MagicMock()
        member.name = "link"
        member.issym.return_value = True
        member.islnk.return_value = False
        tar = MagicMock()
        tar.getmembers.return_value = [member]
        main_module._safe_extract_tar(tar, tmp_path)
        tar.extract.assert_not_called()

    def test_skips_hard_links(self, tmp_path):
        member = MagicMock()
        member.name = "hardlink"
        member.issym.return_value = False
        member.islnk.return_value = True
        tar = MagicMock()
        tar.getmembers.return_value = [member]
        main_module._safe_extract_tar(tar, tmp_path)
        tar.extract.assert_not_called()

    def test_skips_path_traversal(self, tmp_path):
        member = MagicMock()
        member.name = "../outside.txt"
        member.issym.return_value = False
        member.islnk.return_value = False
        tar = MagicMock()
        tar.getmembers.return_value = [member]
        main_module._safe_extract_tar(tar, tmp_path)
        tar.extract.assert_not_called()


# ---------------------------------------------------------------------------
# _validate_diarization_models
# ---------------------------------------------------------------------------

class TestValidateDiarizationModels:
    def test_returns_existing_custom_paths(self, tmp_path):
        seg = tmp_path / "seg.onnx"
        emb = tmp_path / "emb.onnx"
        seg.touch()
        emb.touch()
        result_seg, result_emb = main_module._validate_diarization_models(
            str(seg), str(emb), tmp_path
        )
        assert result_seg == str(seg)
        assert result_emb == str(emb)

    def test_exits_when_custom_seg_missing(self, tmp_path):
        with pytest.raises(SystemExit):
            main_module._validate_diarization_models(
                str(tmp_path / "missing.onnx"), "", tmp_path
            )

    def test_exits_when_custom_emb_missing(self, tmp_path):
        seg = tmp_path / "seg.onnx"
        seg.touch()
        with pytest.raises(SystemExit):
            main_module._validate_diarization_models(
                str(seg), str(tmp_path / "missing_emb.onnx"), tmp_path
            )

    def test_downloads_seg_when_missing(self, tmp_path):
        emb = tmp_path / "models" / main_module._DIAR_EMB_FILE
        emb.parent.mkdir(parents=True)
        emb.touch()

        # Create the seg model path after fake extraction
        seg_dir = tmp_path / "models" / main_module._DIAR_SEG_EXTRACTED
        seg_file = seg_dir / main_module._DIAR_SEG_MODEL_FILE

        def fake_download(url, dest):
            # simulate archive download
            pass

        def fake_tar_open(*args, **kwargs):
            # On enter, create the seg file to simulate extraction
            seg_dir.mkdir(parents=True, exist_ok=True)
            seg_file.parent.mkdir(parents=True, exist_ok=True)
            seg_file.touch()
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=MagicMock())
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx

        with patch.object(main_module, "_download_file", side_effect=fake_download), \
             patch("sherox.asr._safe_extract_tar"), \
             patch("tarfile.open", side_effect=fake_tar_open):
            # The seg model is created by fake_tar_open, so it exists after extraction
            result_seg, result_emb = main_module._validate_diarization_models(
                "", str(emb), tmp_path
            )
        assert main_module._DIAR_SEG_MODEL_FILE in result_seg

    def test_downloads_emb_when_missing(self, tmp_path):
        # Create the seg path to skip seg download
        seg_dir = tmp_path / "models" / main_module._DIAR_SEG_EXTRACTED
        seg_dir.mkdir(parents=True)
        seg_file = seg_dir / main_module._DIAR_SEG_MODEL_FILE
        seg_file.touch()

        def fake_download(url, dest):
            # simulate emb download - create the file
            (tmp_path / "models" / main_module._DIAR_EMB_FILE).touch()

        with patch.object(main_module, "_download_file", side_effect=fake_download):
            result_seg, result_emb = main_module._validate_diarization_models(
                "", "", tmp_path
            )
        assert main_module._DIAR_EMB_FILE in result_emb


# ---------------------------------------------------------------------------
# _validate_mic
# ---------------------------------------------------------------------------

class TestValidateMic:
    def test_passes_with_input_devices(self):
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"max_input_channels": 2, "name": "Microphone"}
        ]
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            main_module._validate_mic()  # should not raise

    def test_exits_when_no_input_devices(self):
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"max_input_channels": 0, "name": "Speaker"}
        ]
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            with pytest.raises(SystemExit):
                main_module._validate_mic()

    def test_exits_on_exception(self):
        with patch.dict("sys.modules", {"sounddevice": None}):
            with pytest.raises(SystemExit):
                main_module._validate_mic()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def _common_patches(self, tmp_path, offline=False, wav=None, model_type="",
                        diarization=False, capture_rate=16000, model_dir=None):
        """Return a dict of patches for testing main()."""
        import argparse
        # Wrap single WAV path in a list to match nargs='+' behaviour.
        wav_list = [wav] if isinstance(wav, str) else wav
        args = argparse.Namespace(
            mic=wav_list is None,
            wav=wav_list,
            pipe=False,
            model_dir=model_dir,
            model_type=model_type,
            offline=offline,
            language="en",
            sample_rate=16000,
            capture_rate=capture_rate,
            chunk_size=0.16,
            threads=4,
            listening=False,
            vad_type="silero",
            ten_vad_model="ten-vad.int8.onnx",
            diarization=diarization,
            diarization_seg_model="",
            diarization_emb_model="",
            num_speakers=-1,
            speaker_tag=False,
            device="cpu",
            denoise=False,
            word_timestamps=False,
            punctuation=False,
            punct_model="",
            output="",
            output_dir="",
            output_format="txt",
            final_only=False,
            translate=False,
            no_color=False,
            json_output=False,
        )
        return args

    def test_online_mic_mode(self):
        args = self._common_patches(None, model_dir="models/zipformer-en-2023")
        mock_rec = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model"), \
             patch.object(main_module, "_validate_vad", return_value=""), \
             patch.object(main_module, "_validate_mic"), \
             patch("sherox.asr.build_recognizer", return_value=mock_rec), \
             patch("sherox.asr.mic_stream", return_value=iter([])), \
             patch("sherox.asr.run_streaming"):
            main_module.main()

    def test_pipe_mode_calls_run_streaming(self):
        args = self._common_patches(None, model_dir="models/zipformer-en-2023")
        args.mic = False
        args.pipe = True
        mock_rec = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model"), \
             patch.object(main_module, "_validate_vad", return_value=""), \
             patch("sherox.asr.build_recognizer", return_value=mock_rec), \
             patch("sherox.asr.pipe_stream", return_value=iter([])) as mock_pipe, \
             patch("sherox.asr.run_streaming") as mock_run:
            main_module.main()

        mock_pipe.assert_called_once()
        mock_run.assert_called_once()

    def test_online_wav_mode(self, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.touch()
        args = self._common_patches(
            tmp_path, wav=str(wav), model_dir="models/zipformer-en-2023"
        )
        mock_rec = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model"), \
             patch.object(main_module, "_validate_vad", return_value=""), \
             patch.object(main_module, "_validate_wav"), \
             patch("sherox.asr.build_recognizer", return_value=mock_rec), \
             patch("sherox.asr.read_wav", return_value=iter([])), \
             patch("sherox.asr.run_streaming"):
            main_module.main()

    def test_online_wav_final_only_skips_progress(self, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.touch()
        args = self._common_patches(
            tmp_path, wav=str(wav), model_dir="models/zipformer-en-2023"
        )
        args.final_only = True
        mock_rec = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model"), \
             patch.object(main_module, "_validate_vad", return_value=""), \
             patch.object(main_module, "_validate_wav"), \
             patch("sherox.asr.build_recognizer", return_value=mock_rec), \
             patch("sherox.asr.read_wav", return_value=iter([])), \
             patch("rich.progress.Progress") as mock_progress, \
             patch("sherox.asr.run_streaming") as mock_run:
            main_module.main()

        mock_progress.assert_not_called()
        assert mock_run.call_args.kwargs["final_only"] is True

    def test_offline_mic_mode(self):
        args = self._common_patches(None, offline=True)
        mock_rec = MagicMock()
        mock_vad = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model"), \
             patch.object(main_module, "_validate_vad", return_value="models/silero_vad.onnx"), \
             patch.object(main_module, "_validate_mic"), \
             patch("sherox.asr.build_offline_recognizer", return_value=mock_rec), \
             patch("sherox.asr.build_vad", return_value=mock_vad), \
             patch("sherox.asr.mic_stream", return_value=iter([])), \
             patch("sherox.asr.run_offline_vad_streaming"):
            main_module.main()

    def test_offline_wav_mode(self, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.touch()
        args = self._common_patches(tmp_path, offline=True, wav=str(wav))
        mock_rec = MagicMock()
        mock_vad = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model"), \
             patch.object(main_module, "_validate_vad", return_value="models/silero_vad.onnx"), \
             patch.object(main_module, "_validate_wav"), \
             patch("sherox.asr.build_offline_recognizer", return_value=mock_rec), \
             patch("sherox.asr.build_vad", return_value=mock_vad), \
             patch("sherox.asr.read_wav", return_value=iter([])), \
             patch("sherox.asr.run_offline_vad_streaming"):
            main_module.main()

    def test_ja_model_type_sets_reazon_dir(self):
        args = self._common_patches(None, model_type="ja")
        mock_rec = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model") as mock_vm, \
             patch.object(main_module, "_validate_vad", return_value=""), \
             patch.object(main_module, "_validate_mic"), \
             patch("sherox.asr.build_offline_recognizer", return_value=mock_rec), \
             patch("sherox.asr.build_vad", return_value=MagicMock()), \
             patch("sherox.asr.mic_stream", return_value=iter([])), \
             patch("sherox.asr.run_offline_vad_streaming"):
            main_module.main()

        # _validate_model should have been called with a reazon path
        called_dir = mock_vm.call_args[0][0]
        assert main_module._REAZON_JA_TARGET in called_dir

    def test_ja_en_model_type_sets_reazon_en_dir(self):
        args = self._common_patches(None, model_type="ja-en")
        mock_rec = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model") as mock_vm, \
             patch.object(main_module, "_validate_vad", return_value=""), \
             patch.object(main_module, "_validate_mic"), \
             patch("sherox.asr.build_offline_recognizer", return_value=mock_rec), \
             patch("sherox.asr.build_vad", return_value=MagicMock()), \
             patch("sherox.asr.mic_stream", return_value=iter([])), \
             patch("sherox.asr.run_offline_vad_streaming"):
            main_module.main()

        called_dir = mock_vm.call_args[0][0]
        assert main_module._REAZON_JA_EN_TARGET in called_dir

    def test_cohere_transcribe_model_type_sets_cohere_dir(self):
        args = self._common_patches(None, model_type="cohere_transcribe")
        mock_rec = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model") as mock_vm, \
             patch.object(main_module, "_validate_vad", return_value=""), \
             patch.object(main_module, "_validate_mic"), \
             patch("sherox.asr.build_offline_recognizer", return_value=mock_rec), \
             patch("sherox.asr.build_vad", return_value=MagicMock()), \
             patch("sherox.asr.mic_stream", return_value=iter([])), \
             patch("sherox.asr.run_offline_vad_streaming"):
            main_module.main()

        called_dir = mock_vm.call_args[0][0]
        assert main_module._COHERE_TRANSCRIBE_TARGET in called_dir

    def test_multilingual_streaming_model_type_sets_multilingual_dir(self):
        args = self._common_patches(None, model_type="multilingual_streaming")
        mock_rec = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model") as mock_vm, \
             patch.object(main_module, "_validate_vad", return_value=""), \
             patch.object(main_module, "_validate_mic"), \
             patch("sherox.asr.build_recognizer", return_value=mock_rec), \
             patch("sherox.asr.mic_stream", return_value=iter([])), \
             patch("sherox.asr.run_streaming"):
            main_module.main()

        called_dir = mock_vm.call_args[0][0]
        assert main_module._MULTILINGUAL_STREAMING_TARGET in called_dir

    def test_ja_en_mls_5k_model_type(self):
        args = self._common_patches(None, model_type="ja-en-mls-5k")
        mock_rec = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model") as mock_vm, \
             patch.object(main_module, "_validate_vad", return_value=""), \
             patch.object(main_module, "_validate_mic"), \
             patch("sherox.asr.build_offline_recognizer", return_value=mock_rec), \
             patch("sherox.asr.build_vad", return_value=MagicMock()), \
             patch("sherox.asr.mic_stream", return_value=iter([])), \
             patch("sherox.asr.run_offline_vad_streaming"):
            main_module.main()

        called_dir = mock_vm.call_args[0][0]
        assert main_module._REAZON_JA_EN_MLS_TARGET in called_dir

    def test_auto_offline_for_whisper_model_type(self):
        args = self._common_patches(None, model_type="whisper", offline=False)
        mock_rec = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model"), \
             patch.object(main_module, "_validate_vad", return_value="models/silero.onnx"), \
             patch.object(main_module, "_validate_mic"), \
             patch("sherox.asr.build_offline_recognizer", return_value=mock_rec), \
             patch("sherox.asr.build_vad", return_value=MagicMock()), \
             patch("sherox.asr.mic_stream", return_value=iter([])), \
             patch("sherox.asr.run_offline_vad_streaming"):
            main_module.main()
        # Should have switched to offline automatically — no assertion needed,
        # just verify no exception raised

    def test_offline_with_diarization(self):
        args = self._common_patches(None, offline=True, diarization=True)
        mock_rec = MagicMock()
        mock_vad = MagicMock()
        mock_diarizer = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model"), \
             patch.object(main_module, "_validate_vad", return_value="models/silero_vad.onnx"), \
             patch.object(main_module, "_validate_mic"), \
             patch.object(main_module, "_validate_diarization_models",
                          return_value=("seg.onnx", "emb.onnx")), \
             patch("sherox.asr.build_offline_recognizer", return_value=mock_rec), \
             patch("sherox.asr.build_vad", return_value=mock_vad), \
             patch("sherox.asr.build_diarization", return_value=mock_diarizer), \
             patch("sherox.asr.mic_stream", return_value=iter([])), \
             patch("sherox.asr.run_offline_vad_streaming") as mock_run:
            main_module.main()

        # run_offline_vad_streaming should have been called with the diarizer
        kwargs = mock_run.call_args[1]
        assert kwargs.get("diarization") is mock_diarizer

    def test_custom_model_dir_used(self, tmp_path):
        custom_dir = str(tmp_path)
        args = self._common_patches(None, model_dir=custom_dir)
        mock_rec = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model") as mock_vm, \
             patch.object(main_module, "_validate_vad", return_value=""), \
             patch.object(main_module, "_validate_mic"), \
             patch("sherox.asr.build_recognizer", return_value=mock_rec), \
             patch("sherox.asr.mic_stream", return_value=iter([])), \
             patch("sherox.asr.run_streaming"):
            main_module.main()

        called_dir = mock_vm.call_args[0][0]
        assert called_dir == custom_dir

    def test_japanese_language_uses_japanese_default_model(self):
        args = self._common_patches(None)
        args.language = "jp"
        mock_rec = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model") as mock_vm, \
             patch.object(main_module, "_validate_vad", return_value="models/silero_vad.onnx"), \
             patch.object(main_module, "_validate_mic"), \
             patch("sherox.asr.build_offline_recognizer", return_value=mock_rec) as mock_build, \
             patch("sherox.asr.build_vad", return_value=MagicMock()), \
             patch("sherox.asr.mic_stream", return_value=iter([])), \
             patch("sherox.asr.run_offline_vad_streaming"):
            main_module.main()

        called_dir, called_type = mock_vm.call_args[0]
        assert Path(called_dir).name == main_module._PARAKEET_CTC_JA_INT8_TARGET
        assert called_type == "parakeet-ctc-ja"
        cfg = mock_build.call_args[0][0]
        assert cfg.language == "ja"
        assert cfg.offline is True

    def test_german_language_uses_streaming_zipformer_online(self):
        args = self._common_patches(None)
        args.language = "de"
        mock_rec = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model") as mock_vm, \
             patch.object(main_module, "_validate_vad", return_value=""), \
             patch.object(main_module, "_validate_mic"), \
             patch("sherox.asr.build_recognizer", return_value=mock_rec), \
             patch("sherox.asr.mic_stream", return_value=iter([])), \
             patch("sherox.asr.run_streaming"):
            main_module.main()

        called_dir, called_type = mock_vm.call_args[0]
        assert Path(called_dir).name == main_module._GERMAN_STREAMING_TARGET
        assert called_type == ""

    def test_german_language_offline_uses_nemo_ctc(self):
        args = self._common_patches(None, offline=True)
        args.language = "de"
        mock_rec = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model") as mock_vm, \
             patch.object(main_module, "_validate_vad", return_value="models/silero_vad.onnx"), \
             patch.object(main_module, "_validate_mic"), \
             patch("sherox.asr.build_offline_recognizer", return_value=mock_rec), \
             patch("sherox.asr.build_vad", return_value=MagicMock()), \
             patch("sherox.asr.mic_stream", return_value=iter([])), \
             patch("sherox.asr.run_offline_vad_streaming"):
            main_module.main()

        called_dir, called_type = mock_vm.call_args[0]
        assert Path(called_dir).name == main_module._GERMAN_NEMO_TARGET
        assert called_type == "nemo_ctc"

    def test_german_lang_aliases_resolve_to_de(self):
        for alias in ("deu", "ger", "deutsch", "german", "de-DE", "de_DE", "de-AT", "de-CH"):
            assert main_module._normalize_language(alias) == "de", (
                f"alias {alias!r} did not resolve to 'de'"
            )

    def test_english_language_uses_parakeet_int8_default_model(self):
        args = self._common_patches(None)
        mock_rec = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model") as mock_vm, \
             patch.object(main_module, "_validate_vad", return_value="models/silero_vad.onnx"), \
             patch.object(main_module, "_validate_mic"), \
             patch("sherox.asr.build_offline_recognizer", return_value=mock_rec) as mock_build, \
             patch("sherox.asr.build_vad", return_value=MagicMock()), \
             patch("sherox.asr.mic_stream", return_value=iter([])), \
             patch("sherox.asr.run_offline_vad_streaming"):
            main_module.main()

        called_dir, called_type = mock_vm.call_args[0]
        assert Path(called_dir).name == main_module._PARAKEET_INT8_TARGET
        assert called_type == ""
        cfg = mock_build.call_args[0][0]
        assert cfg.language == "en"
        assert cfg.offline is True

    def test_nemo_transducer_uses_parakeet_int8_default_model(self):
        args = self._common_patches(None, model_type="nemo_transducer")
        mock_rec = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model") as mock_vm, \
             patch.object(main_module, "_validate_vad", return_value="models/silero_vad.onnx"), \
             patch.object(main_module, "_validate_mic"), \
             patch("sherox.asr.build_offline_recognizer", return_value=mock_rec), \
             patch("sherox.asr.build_vad", return_value=MagicMock()), \
             patch("sherox.asr.mic_stream", return_value=iter([])), \
             patch("sherox.asr.run_offline_vad_streaming"):
            main_module.main()

        called_dir, called_type = mock_vm.call_args[0]
        assert Path(called_dir).name == main_module._PARAKEET_INT8_TARGET
        assert called_type == "nemo_transducer"

    def test_online_with_diarization(self):
        args = self._common_patches(
            None, diarization=True, model_dir="models/zipformer-en-2023"
        )
        mock_rec = MagicMock()
        mock_diarizer = MagicMock()

        with patch.object(main_module, "parse_args", return_value=args), \
             patch.object(main_module, "_validate_runtime_args"), \
             patch.object(main_module, "_validate_model"), \
             patch.object(main_module, "_validate_vad", return_value=""), \
             patch.object(main_module, "_validate_mic"), \
             patch.object(main_module, "_validate_diarization_models",
                          return_value=("seg.onnx", "emb.onnx")), \
             patch("sherox.asr.build_recognizer", return_value=mock_rec), \
             patch("sherox.asr.build_diarization", return_value=mock_diarizer), \
             patch("sherox.asr.mic_stream", return_value=iter([])), \
             patch("sherox.asr.run_streaming") as mock_run:
            main_module.main()

        kwargs = mock_run.call_args[1]
        assert kwargs.get("diarization") is mock_diarizer


# ---------------------------------------------------------------------------
# _download_model — extraction failure
# ---------------------------------------------------------------------------

class TestDownloadModelExtractionFailure:
    def test_exits_when_tarfile_raises_on_open(self, tmp_path):
        model_dir = tmp_path / main_module._MODEL_TARGET

        with patch.object(main_module, "_download_file"), \
             patch("tarfile.open", side_effect=Exception("corrupt tar")), \
             pytest.raises(SystemExit):
            main_module._download_model(str(model_dir), "")


# ---------------------------------------------------------------------------
# _validate_diarization_models — extraction failure and seg file missing
# ---------------------------------------------------------------------------

class TestValidateDiarizationModelsErrors:
    def test_exits_when_seg_extraction_raises(self, tmp_path):
        emb = tmp_path / "models" / main_module._DIAR_EMB_FILE
        emb.parent.mkdir(parents=True)
        emb.touch()

        with patch.object(main_module, "_download_file"), \
             patch("tarfile.open", side_effect=Exception("corrupt")), \
             pytest.raises(SystemExit):
            main_module._validate_diarization_models("", str(emb), tmp_path)

    def test_exits_when_seg_file_missing_after_extraction(self, tmp_path):
        emb = tmp_path / "models" / main_module._DIAR_EMB_FILE
        emb.parent.mkdir(parents=True)
        emb.touch()

        def fake_tar_open(*args, **kwargs):
            # Does not create the seg model file → seg_path.exists() is False
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=MagicMock())
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx

        with patch.object(main_module, "_download_file"), \
             patch("tarfile.open", side_effect=fake_tar_open), \
             patch.object(main_module, "_safe_extract_tar"), \
             pytest.raises(SystemExit):
            main_module._validate_diarization_models("", str(emb), tmp_path)
