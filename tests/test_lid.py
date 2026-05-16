import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

import sherox.lid as lid_module
from sherox.config import LidConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_wav(path: Path, duration_s: float = 0.5, sr: int = 16000) -> None:
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    sf.write(str(path), samples, samplerate=sr)


def _make_tar_with_models(archive: Path, size: str) -> None:
    """Build a fake tar.bz2 mirroring the sherpa-onnx Whisper release layout."""
    target = archive.parent / f"sherpa-onnx-whisper-{size}"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{size}-encoder.int8.onnx").write_bytes(b"")
    (target / f"{size}-decoder.int8.onnx").write_bytes(b"")
    with tarfile.open(archive, "w:bz2") as tf:
        tf.add(target, arcname=target.name)
    # Remove the staged dir so _resolve_model has to extract.
    for f in target.iterdir():
        f.unlink()
    target.rmdir()


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_mic_mode(self):
        with patch("sys.argv", ["sherox.lid", "--mic"]):
            args = lid_module.parse_args()
        assert args.mic is True
        assert args.wav is None

    def test_wav_mode(self):
        with patch("sys.argv", ["sherox.lid", "--wav", "a.wav"]):
            args = lid_module.parse_args()
        assert args.wav == "a.wav"
        assert args.mic is False

    def test_mutually_exclusive(self):
        with patch("sys.argv", ["sherox.lid", "--mic", "--wav", "a.wav"]):
            with pytest.raises(SystemExit):
                lid_module.parse_args()

    def test_requires_mic_or_wav(self):
        with patch("sys.argv", ["sherox.lid"]):
            with pytest.raises(SystemExit):
                lid_module.parse_args()

    def test_defaults(self):
        with patch("sys.argv", ["sherox.lid", "--mic"]):
            args = lid_module.parse_args()
        assert args.size == "tiny"
        assert args.encoder == ""
        assert args.decoder == ""
        assert args.sample_rate == 16000
        assert args.capture_rate == 16000
        assert args.chunk_size == 0.1
        assert args.threads == 4
        assert args.provider == "cpu"
        assert args.listening is False

    def test_custom_size(self):
        with patch("sys.argv", ["sherox.lid", "--wav", "a.wav", "--size", "base"]):
            args = lid_module.parse_args()
        assert args.size == "base"

    def test_rejects_unsupported_size(self):
        with patch("sys.argv", ["sherox.lid", "--mic", "--size", "huge"]):
            with pytest.raises(SystemExit):
                lid_module.parse_args()


# ---------------------------------------------------------------------------
# _resolve_model
# ---------------------------------------------------------------------------

class TestResolveModel:
    def test_returns_existing_pair(self, tmp_path):
        models = tmp_path / "models" / "sherpa-onnx-whisper-tiny"
        models.mkdir(parents=True)
        enc = models / "tiny-encoder.int8.onnx"
        dec = models / "tiny-decoder.int8.onnx"
        enc.touch()
        dec.touch()

        cfg = LidConfig(size="tiny")
        with patch.object(lid_module, "_download_file") as mock_dl:
            e, d = lid_module._resolve_model(cfg, tmp_path)
        mock_dl.assert_not_called()
        assert e == str(enc)
        assert d == str(dec)

    def test_uses_explicit_paths(self, tmp_path):
        enc = tmp_path / "enc.onnx"
        dec = tmp_path / "dec.onnx"
        enc.touch()
        dec.touch()
        cfg = LidConfig(encoder=str(enc), decoder=str(dec))
        e, d = lid_module._resolve_model(cfg, tmp_path)
        assert e == str(enc)
        assert d == str(dec)

    def test_exits_when_explicit_missing(self, tmp_path):
        cfg = LidConfig(encoder=str(tmp_path / "no.onnx"), decoder=str(tmp_path / "no2.onnx"))
        with pytest.raises(SystemExit):
            lid_module._resolve_model(cfg, tmp_path)

    def test_exits_on_unsupported_size(self, tmp_path):
        cfg = LidConfig(size="huge")
        with pytest.raises(SystemExit):
            lid_module._resolve_model(cfg, tmp_path)

    def test_downloads_and_extracts(self, tmp_path):
        cfg = LidConfig(size="tiny")

        def fake_download(url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            _make_tar_with_models(dest, "tiny")

        with patch.object(lid_module, "_download_file", side_effect=fake_download):
            e, d = lid_module._resolve_model(cfg, tmp_path)

        assert Path(e).name == "tiny-encoder.int8.onnx"
        assert Path(d).name == "tiny-decoder.int8.onnx"


# ---------------------------------------------------------------------------
# _validate_vad
# ---------------------------------------------------------------------------

class TestValidateVad:
    def test_downloads_when_missing(self, tmp_path):
        with patch.object(lid_module, "_download_file") as mock_dl:
            result = lid_module._validate_vad(tmp_path)
        mock_dl.assert_called_once()
        assert "silero_vad.onnx" in result

    def test_returns_existing(self, tmp_path):
        vad = tmp_path / "models" / "silero_vad.onnx"
        vad.parent.mkdir()
        vad.touch()
        with patch.object(lid_module, "_download_file") as mock_dl:
            result = lid_module._validate_vad(tmp_path)
        mock_dl.assert_not_called()
        assert result == str(vad)


# ---------------------------------------------------------------------------
# _load_wav_flat
# ---------------------------------------------------------------------------

class TestLoadWavFlat:
    def test_loads_mono_wav(self, tmp_path):
        wav = tmp_path / "test.wav"
        _write_wav(wav)
        samples, sr = lid_module._load_wav_flat(str(wav))
        assert sr == 16000
        assert samples.dtype == np.float32
        assert samples.ndim == 1


# ---------------------------------------------------------------------------
# _build_slid
# ---------------------------------------------------------------------------

class TestBuildSlid:
    def test_passes_encoder_decoder_to_config(self, tmp_path):
        cfg = LidConfig(encoder="/tmp/enc.onnx", decoder="/tmp/dec.onnx",
                        num_threads=2, provider="cpu")
        mock_sherpa = MagicMock()
        whisper_cfg = MagicMock()
        slid_cfg = MagicMock()
        slid_instance = MagicMock()
        mock_sherpa.SpokenLanguageIdentificationWhisperConfig.return_value = whisper_cfg
        mock_sherpa.SpokenLanguageIdentificationConfig.return_value = slid_cfg
        mock_sherpa.SpokenLanguageIdentification.return_value = slid_instance
        with patch.object(lid_module, "_require_sherpa_onnx", return_value=mock_sherpa):
            result = lid_module._build_slid(cfg)
        mock_sherpa.SpokenLanguageIdentificationWhisperConfig.assert_called_once_with(
            encoder="/tmp/enc.onnx", decoder="/tmp/dec.onnx",
        )
        mock_sherpa.SpokenLanguageIdentificationConfig.assert_called_once()
        mock_sherpa.SpokenLanguageIdentification.assert_called_once_with(slid_cfg)
        assert result is slid_instance


# ---------------------------------------------------------------------------
# _identify
# ---------------------------------------------------------------------------

class TestIdentify:
    def test_returns_language(self):
        slid = MagicMock()
        stream = MagicMock()
        slid.create_stream.return_value = stream
        slid.compute.return_value = "de"
        lang = lid_module._identify(slid, np.zeros(8000, dtype=np.float32), 16000)
        assert lang == "de"
        stream.accept_waveform.assert_called_once()

    def test_returns_unknown_on_empty(self):
        slid = MagicMock()
        slid.create_stream.return_value = MagicMock()
        slid.compute.return_value = ""
        lang = lid_module._identify(slid, np.zeros(8000, dtype=np.float32), 16000)
        assert lang == "unknown"


# ---------------------------------------------------------------------------
# run_wav
# ---------------------------------------------------------------------------

class TestRunWav:
    def test_prints_detected_language(self, tmp_path, capsys):
        wav = tmp_path / "test.wav"
        _write_wav(wav)
        cfg = LidConfig(wav=str(wav))
        slid = MagicMock()
        slid.create_stream.return_value = MagicMock()
        slid.compute.return_value = "en"
        with patch.object(lid_module, "_build_slid", return_value=slid):
            lid_module.run_wav(cfg)
        assert "en" in capsys.readouterr().out

    def test_prints_unknown_on_empty(self, tmp_path, capsys):
        wav = tmp_path / "test.wav"
        _write_wav(wav)
        cfg = LidConfig(wav=str(wav))
        slid = MagicMock()
        slid.create_stream.return_value = MagicMock()
        slid.compute.return_value = ""
        with patch.object(lid_module, "_build_slid", return_value=slid):
            lid_module.run_wav(cfg)
        assert "unknown" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run_mic
# ---------------------------------------------------------------------------

class TestRunMic:
    def test_processes_vad_segments(self, tmp_path, capsys):
        vad_path = tmp_path / "silero_vad.onnx"
        vad_path.touch()
        cfg = LidConfig(vad_model=str(vad_path), capture_rate=16000)
        slid = MagicMock()
        slid.create_stream.return_value = MagicMock()
        slid.compute.return_value = "ja"
        vad = MagicMock()
        seg = MagicMock()
        seg.samples = np.ones(8000, dtype=np.float32).tolist()
        vad.front = seg
        vad.empty.side_effect = [False, True, True]
        with patch.object(lid_module, "_build_slid", return_value=slid), \
             patch("sherox.lid.build_vad", return_value=vad), \
             patch("sherox.lid.mic_stream",
                   return_value=iter([np.zeros(1600, dtype=np.float32)])):
            lid_module.run_mic(cfg)
        assert "ja" in capsys.readouterr().out

    def test_keyboard_interrupt_handled(self, tmp_path):
        vad_path = tmp_path / "silero_vad.onnx"
        vad_path.touch()
        cfg = LidConfig(vad_model=str(vad_path), capture_rate=16000)
        slid = MagicMock()
        vad = MagicMock()
        vad.empty.return_value = True

        def interrupt_gen():
            yield np.zeros(1600, dtype=np.float32)
            raise KeyboardInterrupt

        with patch.object(lid_module, "_build_slid", return_value=slid), \
             patch("sherox.lid.build_vad", return_value=vad), \
             patch("sherox.lid.mic_stream", return_value=interrupt_gen()):
            lid_module.run_mic(cfg)  # must not propagate

    def test_mic_level_bar_shown(self, tmp_path, capsys):
        vad_path = tmp_path / "silero_vad.onnx"
        vad_path.touch()
        cfg = LidConfig(vad_model=str(vad_path), capture_rate=16000, show_mic_level=True)
        slid = MagicMock()
        vad = MagicMock()
        vad.empty.return_value = True
        with patch.object(lid_module, "_build_slid", return_value=slid), \
             patch("sherox.lid.build_vad", return_value=vad), \
             patch("sherox.lid.mic_stream",
                   return_value=iter([np.ones(1600, dtype=np.float32) * 0.1])):
            lid_module.run_mic(cfg)
        assert "mic:" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_wav_mode(self, tmp_path):
        wav = tmp_path / "test.wav"
        _write_wav(wav)
        enc = tmp_path / "enc.onnx"
        dec = tmp_path / "dec.onnx"
        enc.touch()
        dec.touch()
        with patch("sys.argv", [
            "sherox.lid", "--wav", str(wav),
            "--encoder", str(enc), "--decoder", str(dec),
        ]), patch.object(lid_module, "run_wav") as mock_run:
            lid_module.main()
        mock_run.assert_called_once()

    def test_main_mic_mode(self, tmp_path):
        enc = tmp_path / "enc.onnx"
        dec = tmp_path / "dec.onnx"
        enc.touch()
        dec.touch()
        with patch("sys.argv", [
            "sherox.lid", "--mic",
            "--encoder", str(enc), "--decoder", str(dec),
        ]), \
        patch.object(lid_module, "_validate_vad", return_value="silero_vad.onnx"), \
        patch.object(lid_module, "run_mic") as mock_run:
            lid_module.main()
        mock_run.assert_called_once()

    def test_main_requires_both_encoder_and_decoder(self, tmp_path):
        enc = tmp_path / "enc.onnx"
        enc.touch()
        with patch("sys.argv", [
            "sherox.lid", "--wav", "a.wav", "--encoder", str(enc),
        ]):
            with pytest.raises(SystemExit):
                lid_module.main()
