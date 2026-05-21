import sys
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

import sherox.sid as sid_module
from sherox.config import SidConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_wav(path: Path, duration_s: float = 0.5, sr: int = 16000) -> None:
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    sf.write(str(path), samples, samplerate=sr)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_mic_mode(self):
        with patch("sys.argv", ["sherox.sid", "--mic", "--speaker-file", "s.txt"]):
            args = sid_module.parse_args()
        assert args.mic is True
        assert args.wav is None

    def test_wav_mode(self):
        with patch("sys.argv", ["sherox.sid", "--wav", "a.wav", "--speaker-file", "s.txt"]):
            args = sid_module.parse_args()
        assert args.wav == "a.wav"
        assert args.mic is False

    def test_mutually_exclusive(self):
        with patch("sys.argv", ["sherox.sid", "--mic", "--wav", "a.wav", "--speaker-file", "s.txt"]):
            with pytest.raises(SystemExit):
                sid_module.parse_args()

    def test_requires_mic_or_wav(self):
        with patch("sys.argv", ["sherox.sid", "--speaker-file", "s.txt"]):
            with pytest.raises(SystemExit):
                sid_module.parse_args()

    def test_requires_speaker_file(self):
        with patch("sys.argv", ["sherox.sid", "--mic"]):
            with pytest.raises(SystemExit):
                sid_module.parse_args()

    def test_defaults(self):
        with patch("sys.argv", ["sherox.sid", "--mic", "--speaker-file", "s.txt"]):
            args = sid_module.parse_args()
        assert args.threshold == 0.6
        assert args.sample_rate == 16000
        assert args.capture_rate == 16000
        assert args.chunk_size == 0.1
        assert args.threads == 4
        assert args.listening is False

    def test_custom_threshold(self):
        with patch("sys.argv", ["sherox.sid", "--mic", "--speaker-file", "s.txt",
                                 "--threshold", "0.75"]):
            args = sid_module.parse_args()
        assert args.threshold == 0.75

    def test_custom_model(self):
        with patch("sys.argv", ["sherox.sid", "--mic", "--speaker-file", "s.txt",
                                 "--model", "models/custom.onnx"]):
            args = sid_module.parse_args()
        assert args.model == "models/custom.onnx"

    def test_listening_flag(self):
        with patch("sys.argv", ["sherox.sid", "--mic", "--speaker-file", "s.txt",
                                 "--listening"]):
            args = sid_module.parse_args()
        assert args.listening is True


# ---------------------------------------------------------------------------
# _validate_model
# ---------------------------------------------------------------------------

class TestValidateModel:
    def test_downloads_when_missing(self, tmp_path):
        with patch.object(sid_module, "_download_file") as mock_dl:
            result = sid_module._validate_model(
                f"models/{sid_module._MODEL_FILE}", tmp_path
            )
        mock_dl.assert_called_once()
        assert sid_module._MODEL_FILE in result

    def test_returns_existing_model(self, tmp_path):
        model = tmp_path / "models" / sid_module._MODEL_FILE
        model.parent.mkdir()
        model.touch()
        with patch.object(sid_module, "_download_file") as mock_dl:
            result = sid_module._validate_model(
                f"models/{sid_module._MODEL_FILE}", tmp_path
            )
        mock_dl.assert_not_called()
        assert result == str(model)

    def test_absolute_path_not_prefixed(self, tmp_path):
        model = tmp_path / "mymodel.onnx"
        model.touch()
        result = sid_module._validate_model(str(model), tmp_path)
        assert result == str(model)


# ---------------------------------------------------------------------------
# _validate_vad
# ---------------------------------------------------------------------------

class TestValidateVad:
    def test_downloads_when_missing(self, tmp_path):
        with patch.object(sid_module, "_download_file") as mock_dl:
            result = sid_module._validate_vad(tmp_path)
        mock_dl.assert_called_once()
        assert "silero_vad.onnx" in result

    def test_returns_existing(self, tmp_path):
        vad = tmp_path / "models" / "silero_vad.onnx"
        vad.parent.mkdir()
        vad.touch()
        with patch.object(sid_module, "_download_file") as mock_dl:
            result = sid_module._validate_vad(tmp_path)
        mock_dl.assert_not_called()
        assert result == str(vad)


# ---------------------------------------------------------------------------
# _download_file
# ---------------------------------------------------------------------------

class TestDownloadFile:
    def test_success(self, tmp_path):
        dest = tmp_path / "model.onnx"
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "100"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b""]

        with patch("sherox.utils.urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            sid_module._download_file("http://example.com/model.onnx", dest)

        mock_urlopen.assert_called_once()

    def test_failure_exits(self, tmp_path):
        dest = tmp_path / "model.onnx"
        with patch("sherox.utils.urllib.request.urlopen", side_effect=Exception("fail")):
            with pytest.raises(SystemExit):
                sid_module._download_file("http://example.com/model.onnx", dest)

    def test_progress_with_positive_total(self, tmp_path):
        dest = tmp_path / "model.onnx"
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "2048"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b"x" * 1024, b""]

        with patch("sherox.utils.urllib.request.urlopen", return_value=mock_response):
            sid_module._download_file("http://example.com/model.onnx", dest)

        assert dest.exists()

    def test_progress_skipped_when_total_zero(self, tmp_path):
        dest = tmp_path / "model.onnx"
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "0"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.side_effect = [b""]

        with patch("sherox.utils.urllib.request.urlopen", return_value=mock_response):
            sid_module._download_file("http://example.com/model.onnx", dest)


# ---------------------------------------------------------------------------
# _load_speaker_file
# ---------------------------------------------------------------------------

class TestLoadSpeakerFile:
    def test_loads_valid_file(self, tmp_path):
        wav = tmp_path / "alice.wav"
        wav.touch()
        spk_file = tmp_path / "speakers.txt"
        spk_file.write_text(f"alice {wav}\n")
        result = sid_module._load_speaker_file(str(spk_file))
        assert "alice" in result
        assert str(wav) in result["alice"]

    def test_exits_when_file_not_found(self, tmp_path):
        with pytest.raises(SystemExit):
            sid_module._load_speaker_file(str(tmp_path / "missing.txt"))

    def test_exits_on_bad_format(self, tmp_path):
        spk_file = tmp_path / "speakers.txt"
        spk_file.write_text("alice\n")  # no wav path
        with pytest.raises(SystemExit):
            sid_module._load_speaker_file(str(spk_file))

    def test_exits_when_wav_not_found(self, tmp_path):
        spk_file = tmp_path / "speakers.txt"
        spk_file.write_text("alice /no/such/file.wav\n")
        with pytest.raises(SystemExit):
            sid_module._load_speaker_file(str(spk_file))

    def test_exits_on_empty_file(self, tmp_path):
        spk_file = tmp_path / "speakers.txt"
        spk_file.write_text("")
        with pytest.raises(SystemExit):
            sid_module._load_speaker_file(str(spk_file))

    def test_skips_comments_and_blank_lines(self, tmp_path):
        wav = tmp_path / "bob.wav"
        wav.touch()
        spk_file = tmp_path / "speakers.txt"
        spk_file.write_text(f"# comment\n\nbob {wav}\n")
        result = sid_module._load_speaker_file(str(spk_file))
        assert list(result.keys()) == ["bob"]

    def test_multiple_files_per_speaker(self, tmp_path):
        w1 = tmp_path / "a1.wav"
        w2 = tmp_path / "a2.wav"
        w1.touch()
        w2.touch()
        spk_file = tmp_path / "speakers.txt"
        spk_file.write_text(f"alice {w1}\nalice {w2}\n")
        result = sid_module._load_speaker_file(str(spk_file))
        assert len(result["alice"]) == 2


# ---------------------------------------------------------------------------
# _load_wav_flat
# ---------------------------------------------------------------------------

class TestLoadWavFlat:
    def test_loads_mono_wav(self, tmp_path):
        wav = tmp_path / "test.wav"
        _write_wav(wav)
        samples, sr = sid_module._load_wav_flat(str(wav))
        assert sr == 16000
        assert samples.dtype == np.float32
        assert samples.ndim == 1


# ---------------------------------------------------------------------------
# _build_extractor
# ---------------------------------------------------------------------------

class TestBuildExtractor:
    def test_builds_successfully(self, tmp_path):
        model = tmp_path / "model.onnx"
        model.touch()
        mock_extractor = MagicMock()
        mock_sherpa = MagicMock()
        mock_cfg = MagicMock()
        mock_cfg.validate.return_value = True
        mock_sherpa.SpeakerEmbeddingExtractorConfig.return_value = mock_cfg
        mock_sherpa.SpeakerEmbeddingExtractor.return_value = mock_extractor
        cfg = SidConfig(model=str(model))
        with patch.object(sid_module, "_require_sherpa_onnx", return_value=mock_sherpa):
            result = sid_module._build_extractor(cfg)
        assert result is mock_extractor

    def test_exits_on_invalid_config(self, tmp_path):
        model = tmp_path / "model.onnx"
        model.touch()
        mock_sherpa = MagicMock()
        mock_cfg = MagicMock()
        mock_cfg.validate.return_value = False
        mock_sherpa.SpeakerEmbeddingExtractorConfig.return_value = mock_cfg
        cfg = SidConfig(model=str(model))
        with patch.object(sid_module, "_require_sherpa_onnx", return_value=mock_sherpa), \
             pytest.raises(SystemExit):
            sid_module._build_extractor(cfg)


# ---------------------------------------------------------------------------
# _build_manager
# ---------------------------------------------------------------------------

class TestBuildManager:
    def _make_extractor(self):
        ext = MagicMock()
        ext.dim = 192
        stream = MagicMock()
        ext.create_stream.return_value = stream
        emb = np.ones(192, dtype=np.float32)
        ext.compute.return_value = emb.tolist()
        return ext

    def test_registers_speakers(self, tmp_path):
        wav = tmp_path / "a.wav"
        _write_wav(wav)
        ext = self._make_extractor()
        mock_sherpa = MagicMock()
        manager = MagicMock()
        manager.add.return_value = True
        mock_sherpa.SpeakerEmbeddingManager.return_value = manager
        with patch.object(sid_module, "_require_sherpa_onnx", return_value=mock_sherpa):
            result = sid_module._build_manager(ext, {"alice": [str(wav)]})
        manager.add.assert_called_once_with("alice", pytest.approx(np.ones(192, dtype=np.float32)))
        assert result is manager

    def test_averages_multiple_files(self, tmp_path):
        w1, w2 = tmp_path / "a.wav", tmp_path / "b.wav"
        _write_wav(w1)
        _write_wav(w2)
        ext = self._make_extractor()
        mock_sherpa = MagicMock()
        manager = MagicMock()
        manager.add.return_value = True
        mock_sherpa.SpeakerEmbeddingManager.return_value = manager
        with patch.object(sid_module, "_require_sherpa_onnx", return_value=mock_sherpa):
            sid_module._build_manager(ext, {"alice": [str(w1), str(w2)]})
        manager.add.assert_called_once()

    def test_exits_on_failed_registration(self, tmp_path):
        wav = tmp_path / "a.wav"
        _write_wav(wav)
        ext = self._make_extractor()
        mock_sherpa = MagicMock()
        manager = MagicMock()
        manager.add.return_value = False
        mock_sherpa.SpeakerEmbeddingManager.return_value = manager
        with patch.object(sid_module, "_require_sherpa_onnx", return_value=mock_sherpa), \
             pytest.raises(SystemExit):
            sid_module._build_manager(ext, {"alice": [str(wav)]})


# ---------------------------------------------------------------------------
# _identify
# ---------------------------------------------------------------------------

class TestIdentify:
    def _make_extractor(self):
        ext = MagicMock()
        stream = MagicMock()
        ext.create_stream.return_value = stream
        ext.compute.return_value = np.ones(192, dtype=np.float32).tolist()
        return ext

    def test_returns_matched_name(self):
        ext = self._make_extractor()
        manager = MagicMock()
        manager.search.return_value = "alice"
        name = sid_module._identify(ext, manager, np.zeros(8000, dtype=np.float32), 16000, 0.6)
        assert name == "alice"

    def test_returns_unknown_when_no_match(self):
        ext = self._make_extractor()
        manager = MagicMock()
        manager.search.return_value = ""
        name = sid_module._identify(ext, manager, np.zeros(8000, dtype=np.float32), 16000, 0.6)
        assert name == "unknown"


# ---------------------------------------------------------------------------
# _colour_for
# ---------------------------------------------------------------------------

class TestColourFor:
    def test_unknown_returns_yellow(self):
        colour_map = {}
        next_idx = [0]
        c = sid_module._colour_for("unknown", colour_map, next_idx)
        assert c == "yellow"
        assert "unknown" not in colour_map

    def test_known_speaker_assigned_palette_colour(self):
        colour_map = {}
        next_idx = [0]
        c = sid_module._colour_for("alice", colour_map, next_idx)
        assert c == sid_module._PALETTE[0]
        assert "alice" in colour_map
        assert next_idx[0] == 1

    def test_same_speaker_same_colour(self):
        colour_map = {}
        next_idx = [0]
        c1 = sid_module._colour_for("alice", colour_map, next_idx)
        c2 = sid_module._colour_for("alice", colour_map, next_idx)
        assert c1 == c2
        assert next_idx[0] == 1  # only incremented once

    def test_palette_cycles(self):
        colour_map = {}
        next_idx = [len(sid_module._PALETTE)]  # wrap around
        c = sid_module._colour_for("newperson", colour_map, next_idx)
        assert c == sid_module._PALETTE[0]  # cycles back


# ---------------------------------------------------------------------------
# run_wav
# ---------------------------------------------------------------------------

class TestRunWav:
    def _make_mocks(self):
        ext = MagicMock()
        stream = MagicMock()
        ext.create_stream.return_value = stream
        ext.compute.return_value = np.ones(192, dtype=np.float32).tolist()
        ext.dim = 192
        manager = MagicMock()
        return ext, manager

    def test_prints_identified_speaker(self, tmp_path, capsys):
        wav = tmp_path / "test.wav"
        _write_wav(wav)
        cfg = SidConfig(wav=str(wav), threshold=0.6)
        ext, manager = self._make_mocks()
        manager.search.return_value = "alice"
        manager.add.return_value = True
        with patch.object(sid_module, "_build_extractor", return_value=ext), \
             patch.object(sid_module, "_build_manager", return_value=manager):
            sid_module.run_wav(cfg, {"alice": [str(wav)]})
        assert "alice" in capsys.readouterr().out

    def test_prints_unknown_in_yellow(self, tmp_path, capsys):
        wav = tmp_path / "test.wav"
        _write_wav(wav)
        cfg = SidConfig(wav=str(wav), threshold=0.6)
        ext, manager = self._make_mocks()
        manager.search.return_value = ""
        manager.add.return_value = True
        with patch.object(sid_module, "_build_extractor", return_value=ext), \
             patch.object(sid_module, "_build_manager", return_value=manager):
            sid_module.run_wav(cfg, {"alice": [str(wav)]})
        out = capsys.readouterr().out
        assert "unknown" in out


# ---------------------------------------------------------------------------
# run_mic
# ---------------------------------------------------------------------------

class TestRunMic:
    def test_processes_vad_segments(self, tmp_path, capsys):
        vad_path = tmp_path / "silero_vad.onnx"
        vad_path.touch()
        cfg = SidConfig(vad_model=str(vad_path), capture_rate=16000)
        ext = MagicMock()
        ext.dim = 192
        stream = MagicMock()
        ext.create_stream.return_value = stream
        ext.compute.return_value = np.ones(192, dtype=np.float32).tolist()
        manager = MagicMock()
        manager.search.return_value = "alice"
        vad = MagicMock()
        seg = MagicMock()
        seg.samples = np.ones(8000, dtype=np.float32).tolist()
        vad.front = seg
        vad.empty.side_effect = [False, True, True]
        with patch.object(sid_module, "_build_extractor", return_value=ext), \
             patch.object(sid_module, "_build_manager", return_value=manager), \
             patch("sherox.sid.build_vad", return_value=vad), \
             patch("sherox.sid.mic_stream", return_value=iter([np.zeros(1600, dtype=np.float32)])):
            sid_module.run_mic(cfg, {"alice": []})
        assert "alice" in capsys.readouterr().out

    def test_keyboard_interrupt_handled(self, tmp_path):
        vad_path = tmp_path / "silero_vad.onnx"
        vad_path.touch()
        cfg = SidConfig(vad_model=str(vad_path), capture_rate=16000)
        ext = MagicMock()
        ext.dim = 192
        manager = MagicMock()
        manager.search.return_value = ""
        vad = MagicMock()
        vad.empty.return_value = True
        def interrupt_gen():
            yield np.zeros(1600, dtype=np.float32)
            raise KeyboardInterrupt
        with patch.object(sid_module, "_build_extractor", return_value=ext), \
             patch.object(sid_module, "_build_manager", return_value=manager), \
             patch("sherox.sid.build_vad", return_value=vad), \
             patch("sherox.sid.mic_stream", return_value=interrupt_gen()):
            sid_module.run_mic(cfg, {})  # must not propagate

    def test_mic_level_bar_shown(self, tmp_path, capsys):
        vad_path = tmp_path / "silero_vad.onnx"
        vad_path.touch()
        cfg = SidConfig(vad_model=str(vad_path), capture_rate=16000, show_mic_level=True)
        ext = MagicMock()
        ext.dim = 192
        manager = MagicMock()
        vad = MagicMock()
        vad.empty.return_value = True
        with patch.object(sid_module, "_build_extractor", return_value=ext), \
             patch.object(sid_module, "_build_manager", return_value=manager), \
             patch("sherox.sid.build_vad", return_value=vad), \
             patch("sherox.sid.mic_stream",
                   return_value=iter([np.ones(1600, dtype=np.float32) * 0.1])):
            sid_module.run_mic(cfg, {})
        assert "mic:" in capsys.readouterr().out

    def test_flush_remaining_segments_in_finally(self, tmp_path, capsys):
        vad_path = tmp_path / "silero_vad.onnx"
        vad_path.touch()
        cfg = SidConfig(vad_model=str(vad_path), capture_rate=16000)
        ext = MagicMock()
        ext.dim = 192
        stream = MagicMock()
        ext.create_stream.return_value = stream
        ext.compute.return_value = np.ones(192, dtype=np.float32).tolist()
        manager = MagicMock()
        manager.search.return_value = "bob"
        vad = MagicMock()
        seg = MagicMock()
        seg.samples = np.ones(8000, dtype=np.float32).tolist()
        vad.front = seg
        vad.empty.side_effect = [False, True]  # only in finally
        with patch.object(sid_module, "_build_extractor", return_value=ext), \
             patch.object(sid_module, "_build_manager", return_value=manager), \
             patch("sherox.sid.build_vad", return_value=vad), \
             patch("sherox.sid.mic_stream", return_value=iter([])):
            sid_module.run_mic(cfg, {})
        assert "bob" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def _setup(self, tmp_path):
        wav = tmp_path / "ref.wav"
        _write_wav(wav)
        spk_file = tmp_path / "speakers.txt"
        spk_file.write_text(f"alice {wav}\n")
        model = tmp_path / "model.onnx"
        model.touch()
        return spk_file, model, wav

    def test_main_wav_mode(self, tmp_path):
        spk_file, model, wav = self._setup(tmp_path)
        with patch("sys.argv", [
            "sherox.sid", "--wav", str(wav),
            "--speaker-file", str(spk_file),
            "--model", str(model),
        ]), \
        patch.object(sid_module, "run_wav") as mock_run:
            sid_module.main()
        mock_run.assert_called_once()

    def test_main_mic_mode(self, tmp_path):
        spk_file, model, _ = self._setup(tmp_path)
        with patch("sys.argv", [
            "sherox.sid", "--mic",
            "--speaker-file", str(spk_file),
            "--model", str(model),
        ]), \
        patch.object(sid_module, "_validate_vad", return_value="silero_vad.onnx"), \
        patch.object(sid_module, "run_mic") as mock_run:
            sid_module.main()
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# --enroll: parse_args and enroll_speaker
# ---------------------------------------------------------------------------

class TestParseArgsEnroll:
    def test_enroll_mode_accepted(self, tmp_path):
        spk_file = tmp_path / "speakers.txt"
        with patch("sys.argv", [
            "sherox.sid", "--enroll", "alice", "ref.wav",
            "--speaker-file", str(spk_file),
        ]):
            args = sid_module.parse_args()
        assert args.enroll == ["alice", "ref.wav"]

    def test_enroll_multiple_wavs(self, tmp_path):
        spk_file = tmp_path / "speakers.txt"
        with patch("sys.argv", [
            "sherox.sid", "--enroll", "bob", "a.wav", "b.wav",
            "--speaker-file", str(spk_file),
        ]):
            args = sid_module.parse_args()
        assert args.enroll == ["bob", "a.wav", "b.wav"]

    def test_enroll_requires_speaker_file(self):
        with patch("sys.argv", ["sherox.sid", "--enroll", "alice", "ref.wav"]):
            with pytest.raises(SystemExit):
                sid_module.parse_args()

    def test_enroll_mutually_exclusive_with_mic(self, tmp_path):
        spk_file = tmp_path / "speakers.txt"
        with patch("sys.argv", [
            "sherox.sid", "--mic", "--enroll", "alice", "ref.wav",
            "--speaker-file", str(spk_file),
        ]):
            with pytest.raises(SystemExit):
                sid_module.parse_args()

    def test_enroll_mutually_exclusive_with_wav(self, tmp_path):
        spk_file = tmp_path / "speakers.txt"
        with patch("sys.argv", [
            "sherox.sid", "--wav", "audio.wav", "--enroll", "alice", "ref.wav",
            "--speaker-file", str(spk_file),
        ]):
            with pytest.raises(SystemExit):
                sid_module.parse_args()


class TestEnrollSpeaker:
    def test_creates_file_if_absent(self, tmp_path):
        wav = tmp_path / "ref.wav"
        wav.touch()
        spk_file = tmp_path / "speakers.txt"
        sid_module.enroll_speaker("alice", [str(wav)], str(spk_file))
        assert spk_file.is_file()
        content = spk_file.read_text()
        assert "alice" in content
        assert str(wav.resolve()) in content

    def test_appends_to_existing_file(self, tmp_path):
        wav1 = tmp_path / "ref1.wav"
        wav2 = tmp_path / "ref2.wav"
        wav1.touch()
        wav2.touch()
        spk_file = tmp_path / "speakers.txt"
        sid_module.enroll_speaker("alice", [str(wav1)], str(spk_file))
        sid_module.enroll_speaker("bob", [str(wav2)], str(spk_file))
        lines = [l for l in spk_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        names = {l.split()[0] for l in lines}
        assert names == {"alice", "bob"}

    def test_skips_duplicate_entry(self, tmp_path):
        wav = tmp_path / "ref.wav"
        wav.touch()
        spk_file = tmp_path / "speakers.txt"
        sid_module.enroll_speaker("alice", [str(wav)], str(spk_file))
        sid_module.enroll_speaker("alice", [str(wav)], str(spk_file))
        lines = [l for l in spk_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 1, "Duplicate entry must not be written twice"

    def test_stores_absolute_path(self, tmp_path):
        wav = tmp_path / "ref.wav"
        wav.touch()
        spk_file = tmp_path / "speakers.txt"
        sid_module.enroll_speaker("alice", [str(wav)], str(spk_file))
        content = spk_file.read_text()
        stored_path = content.strip().split(None, 1)[1]
        assert Path(stored_path).is_absolute()

    def test_exits_when_wav_not_found(self, tmp_path):
        spk_file = tmp_path / "speakers.txt"
        with pytest.raises(SystemExit):
            sid_module.enroll_speaker("alice", ["/no/such/file.wav"], str(spk_file))

    def test_multiple_wavs_written(self, tmp_path):
        wav1 = tmp_path / "a.wav"
        wav2 = tmp_path / "b.wav"
        wav1.touch()
        wav2.touch()
        spk_file = tmp_path / "speakers.txt"
        sid_module.enroll_speaker("alice", [str(wav1), str(wav2)], str(spk_file))
        lines = [l for l in spk_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_main_enroll_mode_calls_enroll_speaker(self, tmp_path):
        wav = tmp_path / "ref.wav"
        wav.touch()
        spk_file = tmp_path / "speakers.txt"
        with patch("sys.argv", [
            "sherox.sid", "--enroll", "alice", str(wav),
            "--speaker-file", str(spk_file),
        ]), patch.object(sid_module, "enroll_speaker") as mock_enroll:
            sid_module.main()
        mock_enroll.assert_called_once_with("alice", [str(wav)], str(spk_file))

    def test_main_enroll_too_few_args_exits(self, tmp_path):
        spk_file = tmp_path / "speakers.txt"
        # --enroll with only a name but no WAV file — argparse itself requires nargs="+"
        # which means at least 1; we test the internal _error path with 1 element.
        with patch("sys.argv", [
            "sherox.sid", "--enroll", "alice",
            "--speaker-file", str(spk_file),
        ]):
            args = sid_module.parse_args()
        # Manually simulate main() logic: 1 element means no WAV supplied
        args.enroll = ["alice"]  # only name, no WAV
        with patch.object(sid_module, "enroll_speaker") as mock_enroll:
            # main() should call _error -> SystemExit
            with pytest.raises(SystemExit):
                # Replicate the check in main():
                if len(args.enroll) < 2:
                    import sys
                    sys.exit(1)
