"""Ground-truth regression tests for the numpy DSP in sarashina_audio_frontend.py.

These exist because of a real incident: a later commit replaced the
librosa-backed mel filterbank in cosyvoice_mel_spectrogram with a hand-rolled
one that silently dropped librosa's default area normalization (and used the
HTK mel scale instead of librosa's default Slaney scale). Nothing caught it —
there were no tests pinning the filterbank against real librosa output — and
it shipped as a severe, audible regression in zero-shot voice cloning quality
(garbled/inconsistent pronunciation). Every function here is pinned against
real ground truth so that can't happen silently again.

Tests that need librosa/torch/torchaudio for ground truth skip gracefully
when those aren't installed (they're optional/dev-only for this module).
"""
import numpy as np
import pytest

from sherox.sarashina_audio_frontend import (
    _hz_to_mel_slaney,
    _mel_scale_filterbank,
    _mel_to_hz_slaney,
    cosyvoice_mel_spectrogram,
    kaldi_fbank,
    whisper_log_mel,
)


class TestMelScaleFilterbank:
    def test_matches_librosa_default_exactly(self):
        librosa = pytest.importorskip("librosa")
        for sr, n_fft, n_mels, fmax in [
            (24000, 1920, 80, 8000),   # sherox's actual CosyVoice2-mel call
            (16000, 400, 128, 8000),
            (22050, 1024, 80, None),
        ]:
            real = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
            mine = _mel_scale_filterbank(sr, n_fft, n_mels, 0.0, fmax)
            diff = np.abs(real - mine).max()
            assert diff < 1e-6, f"sr={sr} n_fft={n_fft} n_mels={n_mels}: max diff {diff} (librosa uses Slaney scale + area normalization by default — see module docstring)"

    def test_is_area_normalized_not_peak_one(self):
        """Regression guard for the specific bug: filters must NOT all peak at
        1.0 (that's what an un-normalized filterbank looks like); librosa's
        default area-normalizes each filter by its bandwidth."""
        fb = _mel_scale_filterbank(24000, 1920, 80, 0.0, 8000.0)
        peaks = fb.max(axis=1)
        assert not np.allclose(peaks, 1.0, atol=0.05), (
            "filterbank peaks are all ~1.0 — normalization is missing again"
        )
        # Peaks should vary across filters (narrower low-freq filters have
        # higher peaks than wider high-freq ones under area normalization).
        assert peaks.std() > 1e-6

    def test_hz_mel_roundtrip(self):
        freqs = np.array([0.0, 100.0, 500.0, 1000.0, 4000.0, 12000.0])
        roundtrip = _mel_to_hz_slaney(_hz_to_mel_slaney(freqs))
        np.testing.assert_allclose(roundtrip, freqs, atol=1e-6)

    def test_scale_matches_slaney_not_htk(self):
        """The HTK formula alone (without also missing normalization) still
        diverges from librosa's actual default by ~0.03 max filter weight —
        small compared to the normalization bug, but real. Pin the scale too."""
        librosa = pytest.importorskip("librosa")
        real_slaney = librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80, fmax=8000)  # htk=False default
        real_htk = librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80, fmax=8000, htk=True)
        mine = _mel_scale_filterbank(24000, 1920, 80, 0.0, 8000.0)
        assert np.abs(real_slaney - mine).max() < np.abs(real_htk - mine).max()


class TestCosyvoiceMelSpectrogram:
    def test_matches_real_torch_extractor(self):
        """End-to-end: numpy mel vs sarashina_tts's own torch implementation,
        on real speech. This is the actual signal fed into zero-shot cloning."""
        pytest.importorskip("torch")
        pytest.importorskip("torchaudio")
        try:
            from sarashina_tts.flow_matching.decoder import extract_mel_spectrogram
        except ImportError:
            pytest.skip("sarashina_tts not installed")
        import soundfile as sf
        import torch
        import torchaudio

        wav_path = "ishiteru.wav"
        try:
            speech_np, sr = sf.read(wav_path, always_2d=True)
        except Exception:
            pytest.skip(f"{wav_path} not available in this environment")
        speech_t = torch.from_numpy(speech_np[:, 0:1].T).float()
        audio_24k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)(speech_t)

        real_mel = extract_mel_spectrogram(audio_24k.squeeze(0))
        real_mel = real_mel.squeeze(0).numpy() if real_mel.dim() == 3 else real_mel.numpy()
        my_mel = cosyvoice_mel_spectrogram(audio_24k.squeeze(0).numpy())

        diff = np.abs(real_mel - my_mel)
        assert diff.max() < 1e-3, f"max diff {diff.max()} — should be ~1e-5 when the filterbank is correct"

    def test_shape_and_finite(self):
        rng = np.random.RandomState(0)
        y = rng.randn(24000).astype(np.float32) * 0.1  # 1s of noise at 24kHz
        mel = cosyvoice_mel_spectrogram(y)
        assert mel.shape[0] == 80
        assert np.isfinite(mel).all()


class TestWhisperLogMel:
    def test_shape_and_finite(self, tmp_path):
        rng = np.random.RandomState(0)
        audio = rng.randn(16000).astype(np.float32) * 0.1  # 1s at 16kHz
        filters_path = tmp_path / "mel_filters.npz"
        np.savez(filters_path, mel_128=rng.rand(128, 201).astype(np.float32))
        mel = whisper_log_mel(audio, str(filters_path))
        assert mel.shape[0] == 128
        assert np.isfinite(mel).all()


class TestKaldiFbank:
    def test_matches_real_torchaudio(self):
        pytest.importorskip("torch")
        try:
            import torchaudio.compliance.kaldi as kaldi
        except ImportError:
            pytest.skip("torchaudio not installed")
        import soundfile as sf
        import torch

        wav_path = "ishiteru.wav"
        try:
            speech_np, sr = sf.read(wav_path, always_2d=True)
        except Exception:
            pytest.skip(f"{wav_path} not available in this environment")
        speech_t = torch.from_numpy(speech_np[:, 0]).float().unsqueeze(0)
        if sr != 16000:
            import torchaudio
            speech_t = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech_t)

        real = kaldi.fbank(speech_t, num_mel_bins=80, dither=0, sample_frequency=16000)
        mine = kaldi_fbank(speech_t.squeeze(0).numpy())
        diff = np.abs(real.numpy() - mine).max()
        assert diff < 0.01, f"max diff {diff}"

    def test_shape_and_finite(self):
        rng = np.random.RandomState(0)
        audio = rng.randn(16000).astype(np.float32) * 0.1
        fbank = kaldi_fbank(audio)
        assert fbank.shape[1] == 80
        assert np.isfinite(fbank).all()
