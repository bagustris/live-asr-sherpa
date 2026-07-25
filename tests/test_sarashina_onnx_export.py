"""Ground-truth regression tests for the manual STFT/ISTFT reimplementation in
sarashina_onnx_export.py.

Context: after fixing a severe zero-shot cloning regression caused by the mel
filterbank (see test_sarashina_audio_frontend.py), the HiFT vocoder's manual
STFT/ISTFT (used to make torch.stft/torch.istft ONNX-exportable) was suspected
as a second source of audible noise. It wasn't — bisection proved _ManualSTFT
and _ManualISTFT match torch.stft/torch.istft to ~1e-5 on real signals, and the
full HiFT wrapper matches the reference decode() to the same precision once
the NSF source's intentional torch.randn_like noise is neutralized in both
(comparing two independent random draws had been the whole problem with the
earlier diagnosis). These tests pin that finding so a real regression here
can't hide behind "well the vocoder sounds noisy anyway" again.

Uses the exact n_fft/hop/window config HiFT actually ships with (16, 4, hann)
so the tests are representative without needing the multi-GB model checkpoint.
"""
import numpy as np
import pytest


class TestManualSTFT:
    def test_matches_torch_stft(self):
        torch = pytest.importorskip("torch")
        from scipy.signal import get_window

        from sherox.sarashina_onnx_export import _build_manual_stft

        n_fft, hop = 16, 4
        window = torch.from_numpy(get_window("hann", n_fft, fftbins=True).astype(np.float32))

        rng = np.random.RandomState(0)
        x = torch.from_numpy(rng.randn(1, 4800).astype(np.float32))  # 0.2s @ 24kHz

        ManualSTFT = _build_manual_stft(torch, torch.nn, torch.nn.functional)
        manual = ManualSTFT(n_fft, hop, window)

        with torch.no_grad():
            real_mine, imag_mine = manual(x)
            ref = torch.stft(x, n_fft, hop, n_fft, window=window, center=True, return_complex=True)
            ref = torch.view_as_real(ref)
            real_ref, imag_ref = ref[..., 0], ref[..., 1]

        assert real_mine.shape == real_ref.shape
        assert torch.abs(real_mine - real_ref).max().item() < 1e-4
        assert torch.abs(imag_mine - imag_ref).max().item() < 1e-4


class TestManualISTFT:
    def test_matches_torch_istft(self):
        torch = pytest.importorskip("torch")
        from scipy.signal import get_window

        from sherox.sarashina_onnx_export import _build_manual_istft

        n_fft, hop = 16, 4
        window = torch.from_numpy(get_window("hann", n_fft, fftbins=True).astype(np.float32))
        n_freq = n_fft // 2 + 1
        n_frames = 300

        rng = np.random.RandomState(0)
        real = torch.from_numpy(rng.randn(1, n_freq, n_frames).astype(np.float32))
        imag = torch.from_numpy(rng.randn(1, n_freq, n_frames).astype(np.float32))

        ManualISTFT = _build_manual_istft(torch, torch.nn)
        manual = ManualISTFT(n_fft, hop, window)

        with torch.no_grad():
            wav_mine = manual(real, imag)
            wav_ref = torch.istft(
                torch.complex(real, imag), n_fft, hop, n_fft, window=window, center=True,
            )

        assert wav_mine.shape == wav_ref.shape
        assert torch.abs(wav_mine - wav_ref).max().item() < 1e-4


class TestHiftWrapperAgainstReference:
    def test_matches_reference_decode_with_noise_neutralized(self):
        """Compares the full _build_hift_wrapper output against a real
        HiFTGenerator.decode() on the same (randomly-initialized) HiFT module
        and the same mel input, with torch.randn_like forced to zero in both
        so the NSF source's intentional dithering can't mask a real bug (or,
        as happened during the original diagnosis, be mistaken for one)."""
        torch = pytest.importorskip("torch")
        try:
            from sarashina_tts.flow_matching.hifigan import HiFTGenerator
        except ImportError:
            pytest.skip("sarashina_tts not installed")

        from sherox.sarashina_onnx_export import (
            _build_hift_wrapper,
            _build_manual_istft,
            _build_manual_stft,
        )

        torch.manual_seed(0)
        hift = HiFTGenerator().eval()
        hift.remove_weight_norm()

        ManualSTFT = _build_manual_stft(torch, torch.nn, torch.nn.functional)
        ManualISTFT = _build_manual_istft(torch, torch.nn)
        wrapper = _build_hift_wrapper(torch, torch.nn, torch.nn.functional, hift, ManualSTFT, ManualISTFT).eval()

        rng = np.random.RandomState(0)
        mel = torch.from_numpy(rng.randn(1, 80, 50).astype(np.float32))

        def zero_randn_like(x, *a, **kw):
            return torch.zeros_like(x)

        orig_randn_like = torch.randn_like
        try:
            with torch.no_grad():
                torch.randn_like = zero_randn_like
                ref_wav, _ = hift.forward(mel)
                torch.randn_like = orig_randn_like

            with torch.no_grad():
                torch.randn_like = zero_randn_like
                my_wav = wrapper(mel)
                torch.randn_like = orig_randn_like
        finally:
            torch.randn_like = orig_randn_like

        diff = torch.abs(ref_wav.squeeze() - my_wav.squeeze())
        assert diff.max().item() < 1e-3, (
            f"manual HiFT wrapper diverges from the reference decode() by {diff.max().item()} "
            "with NSF source noise neutralized in both — this points to a real bug in the "
            "manual STFT/ISTFT or the decode-loop wiring, not expected random-noise variance."
        )
