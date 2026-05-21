"""Tests for the top-level `sherox` CLI entry point (sherox/main.py)."""

import pytest
from unittest.mock import patch


import sherox.main as main_module


class TestMainCLI:
    def test_version_flag_short(self, capsys):
        with patch("sys.argv", ["sherox", "-v"]):
            with pytest.raises(SystemExit) as exc:
                main_module.main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "sherox" in captured.out
        import sherox
        assert sherox.__version__ in captured.out

    def test_version_flag_long(self, capsys):
        with patch("sys.argv", ["sherox", "--version"]):
            with pytest.raises(SystemExit) as exc:
                main_module.main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "sherox" in captured.out

    def test_no_args_prints_help(self, capsys):
        with patch("sys.argv", ["sherox"]):
            main_module.main()
        captured = capsys.readouterr()
        assert "sherox" in captured.out
        assert "sherox.asr" in captured.out

    def test_unknown_arg_exits_nonzero(self):
        with patch("sys.argv", ["sherox", "--unknown-flag"]):
            with pytest.raises(SystemExit) as exc:
                main_module.main()
        assert exc.value.code != 0
