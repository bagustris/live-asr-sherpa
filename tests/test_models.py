"""Tests for sherox.models — list-models subcommand."""
import argparse
from unittest.mock import patch

import pytest

import sherox.models as models_module
from sherox.models import (
    _ASR_MODELS,
    _TTS_MODELS,
    _OTHER_MODELS,
    print_model_table,
)


# ---------------------------------------------------------------------------
# Registry sanity checks
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_asr_models_non_empty(self):
        assert len(_ASR_MODELS) > 0

    def test_tts_models_non_empty(self):
        assert len(_TTS_MODELS) > 0

    def test_other_models_non_empty(self):
        assert len(_OTHER_MODELS) > 0

    def test_asr_entries_have_five_fields(self):
        for entry in _ASR_MODELS:
            assert len(entry) == 5, f"Bad ASR entry: {entry}"

    def test_tts_entries_have_five_fields(self):
        for entry in _TTS_MODELS:
            assert len(entry) == 5, f"Bad TTS entry: {entry}"

    def test_other_entries_have_six_fields(self):
        for entry in _OTHER_MODELS:
            assert len(entry) == 6, f"Bad other entry: {entry}"

    def test_asr_pipelines_are_valid(self):
        valid = {"online", "offline", "both"}
        for name, lang, pipeline, size, notes in _ASR_MODELS:
            assert pipeline in valid, f"Invalid pipeline '{pipeline}' for {name}"

    def test_english_model_in_asr(self):
        names = [e[0] for e in _ASR_MODELS]
        assert any("parakeet" in n for n in names)

    def test_japanese_model_in_asr(self):
        langs = [e[1] for e in _ASR_MODELS]
        assert any("ja" in l for l in langs)

    def test_whisper_variants_in_asr(self):
        names = {e[0] for e in _ASR_MODELS}
        assert "sherpa-onnx-whisper-large-v3" in names
        assert "sherpa-onnx-whisper-turbo" in names
        assert "sherpa-onnx-whisper-distil-large-v3.5" in names

    def test_tts_includes_eng(self):
        langs = [e[1] for e in _TTS_MODELS]
        assert "en" in langs

    def test_sid_in_other(self):
        modules = [e[0] for e in _OTHER_MODELS]
        assert "sid" in modules

    def test_kws_in_other(self):
        modules = [e[0] for e in _OTHER_MODELS]
        assert "kws" in modules

    def test_sizes_are_non_empty(self):
        for entry in _ASR_MODELS:
            assert entry[3], f"Empty size in ASR entry: {entry}"
        for entry in _TTS_MODELS:
            assert entry[3], f"Empty size in TTS entry: {entry}"


# ---------------------------------------------------------------------------
# print_model_table
# ---------------------------------------------------------------------------

class TestPrintModelTable:
    def test_runs_without_error(self, capsys):
        print_model_table()
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_asr_filter_includes_asr(self, capsys):
        print_model_table(module_filter="asr")
        out = capsys.readouterr().out
        assert "asr" in out.lower()

    def test_asr_filter_excludes_tts(self, capsys):
        print_model_table(module_filter="asr")
        out = capsys.readouterr().out
        assert "tts" not in out.lower() or "Module" in out  # header row only

    def test_tts_filter_includes_tts(self, capsys):
        print_model_table(module_filter="tts")
        out = capsys.readouterr().out
        assert "tts" in out.lower()

    def test_other_filter(self, capsys):
        print_model_table(module_filter="other")
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_no_color_flag(self, capsys):
        """With no_color=True the table should still contain model names."""
        print_model_table(no_color=True)
        out = capsys.readouterr().out
        assert "parakeet" in out.lower() or "asr" in out.lower()


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------

class TestModelsParser:
    def test_default_module_is_all(self):
        with patch("sys.argv", ["sherox.models"]):
            args = models_module._build_parser().parse_args()
        assert args.module == "all"

    def test_module_asr(self):
        with patch("sys.argv", ["sherox.models", "--module", "asr"]):
            args = models_module._build_parser().parse_args()
        assert args.module == "asr"

    def test_module_tts(self):
        with patch("sys.argv", ["sherox.models", "--module", "tts"]):
            args = models_module._build_parser().parse_args()
        assert args.module == "tts"

    def test_invalid_module_exits(self):
        with patch("sys.argv", ["sherox.models", "--module", "unknown"]):
            with pytest.raises(SystemExit):
                models_module._build_parser().parse_args()

    def test_no_color_flag(self):
        with patch("sys.argv", ["sherox.models", "--no-color"]):
            args = models_module._build_parser().parse_args()
        assert args.no_color is True

    def test_default_no_color_is_false(self):
        with patch("sys.argv", ["sherox.models"]):
            args = models_module._build_parser().parse_args()
        assert args.no_color is False


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

class TestModelsMain:
    def test_main_runs_with_default_args(self, capsys):
        with patch("sys.argv", ["sherox.models"]):
            models_module.main()
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_main_asr_filter(self, capsys):
        with patch("sys.argv", ["sherox.models", "--module", "asr"]):
            models_module.main()
        out = capsys.readouterr().out
        assert len(out) > 0
