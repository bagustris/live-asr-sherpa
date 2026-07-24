"""Torch-free audio feature extraction for Sarashina zero-shot voice cloning.

Reimplements, in pure numpy, the three DSP front-ends the original PyTorch
extractors use — each validated against the real torch/torchaudio output on
real speech (see the commit history for the numbers):

    whisper_log_mel        — s3tokenizer.log_mel_spectrogram   (max diff ~2e-5)
    kaldi_fbank             — torchaudio.compliance.kaldi.fbank (max diff ~5e-4)
    cosyvoice_mel_spectrogram — sarashina_tts's CosyVoice2-style mel (max diff ~1e-5)

Resampling uses ``audiokit.resample`` (scipy polyphase, already a sherox
dependency) rather than torchaudio's sinc resampler; the two differ by ~0.006
on a real reference clip, which is small enough to very occasionally flip a
single semantic token to an acoustically adjacent codebook entry (observed:
1 token out of 48) without a perceptible effect on the cloned voice.

Combined with the ONNX-exported CAMPPlus speaker encoder and the S3 semantic
tokenizer (itself already distributed as ONNX upstream), this lets zero-shot
voice cloning run without torch — see :func:`sherox.sarashina_onnx.extract_prompt_features`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_S3_MEL_FILTERS_ASSET = "s3_mel_filters.npz"  # bundled alongside the ONNX artifacts


def whisper_log_mel(audio: np.ndarray, mel_filters_path: str, n_mels: int = 128) -> np.ndarray:
    """Whisper-style log-mel spectrogram (16 kHz), matching s3tokenizer.log_mel_spectrogram.

    Parameters
    ----------
    audio : (T,) float32 waveform at 16 kHz
    mel_filters_path : path to the mel_filters.npz asset (contains "mel_128")

    Returns
    -------
    (n_mels, n_frames) float32
    """
    n_fft, hop = 400, 160
    window = np.hanning(n_fft + 1)[:-1].astype(np.float64)  # periodic Hann
    pad = n_fft // 2
    padded = np.pad(audio.astype(np.float64), (pad, pad), mode="reflect")
    n_frames = 1 + (len(padded) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = padded[idx] * window
    stft = np.fft.rfft(frames, n=n_fft, axis=-1).T  # (n_fft//2+1, n_frames)
    magnitudes = np.abs(stft[:, :-1]) ** 2  # drop the final time frame, matching torch.stft(...)[..., :-1]

    with np.load(mel_filters_path) as f:
        filters = f[f"mel_{n_mels}"]
    mel_spec = filters @ magnitudes
    log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def kaldi_fbank(waveform: np.ndarray, sample_rate: int = 16000, num_mel_bins: int = 80) -> np.ndarray:
    """Kaldi-style mel filterbank, matching torchaudio.compliance.kaldi.fbank with
    this project's exact call defaults: dither=0, window_type='povey',
    frame_length=25ms, frame_shift=10ms, preemphasis_coefficient=0.97,
    snip_edges=True, remove_dc_offset=True, round_to_power_of_two=True,
    low_freq=20, high_freq=0 (-> Nyquist), use_log_fbank=True, use_power=True.

    Parameters
    ----------
    waveform : (T,) float32 waveform at `sample_rate`

    Returns
    -------
    (n_frames, num_mel_bins) float32 log-mel-energies
    """
    waveform = waveform.astype(np.float64)
    frame_length_samples = int(round(sample_rate * 25.0 / 1000.0))
    frame_shift_samples = int(round(sample_rate * 10.0 / 1000.0))
    num_samples = waveform.shape[0]
    if num_samples < frame_length_samples:
        return np.zeros((0, num_mel_bins), dtype=np.float32)
    num_frames = 1 + (num_samples - frame_length_samples) // frame_shift_samples

    idx = np.arange(frame_length_samples)[None, :] + frame_shift_samples * np.arange(num_frames)[:, None]
    frames = waveform[idx].copy()

    # remove DC offset
    frames -= frames.mean(axis=1, keepdims=True)

    # preemphasis: frame[i] -= coeff*frame[i-1] for i>0; frame[0] -= coeff*frame[0]
    coeff = 0.97
    preemph = frames.copy()
    preemph[:, 1:] = frames[:, 1:] - coeff * frames[:, :-1]
    preemph[:, 0] = frames[:, 0] - coeff * frames[:, 0]
    frames = preemph

    # povey window: (0.5 - 0.5*cos(2*pi*i/(N-1)))^0.85
    n = frame_length_samples
    i = np.arange(n)
    povey = (0.5 - 0.5 * np.cos(2 * np.pi * i / (n - 1))) ** 0.85
    frames = frames * povey[None, :]

    # round_to_power_of_two
    fft_size = 1
    while fft_size < frame_length_samples:
        fft_size *= 2

    spec = np.fft.rfft(frames, n=fft_size, axis=-1)
    power = spec.real ** 2 + spec.imag ** 2  # use_power=True

    # Kaldi triangular mel filterbank (HTK mel scale: 1127*ln(1+f/700))
    def mel_scale(freq):
        return 1127.0 * np.log(1.0 + freq / 700.0)

    low_freq = 20.0
    high_freq = 0.5 * sample_rate  # high_freq param is 0 -> Nyquist
    mel_low = mel_scale(low_freq)
    mel_high = mel_scale(high_freq)
    mel_delta = (mel_high - mel_low) / (num_mel_bins + 1)

    num_fft_bins = fft_size // 2 + 1
    fft_freqs = np.arange(num_fft_bins) * sample_rate / fft_size
    mel_of_bin = mel_scale(fft_freqs)

    bins = np.arange(num_mel_bins)
    left_mel = mel_low + bins * mel_delta
    center_mel = mel_low + (bins + 1) * mel_delta
    right_mel = mel_low + (bins + 2) * mel_delta

    m, l, c, r = mel_of_bin[None, :], left_mel[:, None], center_mel[:, None], right_mel[:, None]
    weights = np.where(m <= c, (m - l) / (c - l), (r - m) / (r - c))
    weights = np.clip(weights, 0.0, None)
    weights = np.where((m > l) & (m < r), weights, 0.0)

    mel_energies = power @ weights.T
    # Kaldi's BaseFloat is float32; flooring at float64 eps (vs float32 eps) makes
    # near-silent frames' log-energy far too negative relative to the reference.
    mel_energies = np.clip(mel_energies, np.finfo(np.float32).eps, None)
    return np.log(mel_energies).astype(np.float32)


def _mel_scale_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None
) -> np.ndarray:
    """Create a mel-scale filterbank (HTK formula, librosa-compatible).

    Returns
    -------
    (n_mels, n_fft//2 + 1) float64 matrix of triangular mel filters.
    """
    if fmax is None:
        fmax = sr / 2.0

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_f_min = hz_to_mel(fmin)
    mel_f_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_f_min, mel_f_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_hz = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    fb = np.zeros((n_mels, len(bin_hz)), dtype=np.float64)

    for m in range(n_mels):
        left, center, right = hz_points[m : m + 3]
        lslope = 1.0 / (center - left) if center > left else 0.0
        rslope = 1.0 / (right - center) if right > center else 0.0
        fb[m, :] = np.maximum(0.0, np.minimum(lslope * (bin_hz - left), rslope * (right - bin_hz)))

    return fb


def cosyvoice_mel_spectrogram(
    y: np.ndarray,
    *,
    n_fft: int = 1920,
    num_mels: int = 80,
    sampling_rate: int = 24000,
    hop_size: int = 480,
    win_size: int = 1920,
    fmin: int = 0,
    fmax: int = 8000,
) -> np.ndarray:
    """CosyVoice2-style log-mel spectrogram (24 kHz), matching
    sarashina_tts.flow_matching.decoder.extract_mel_spectrogram (center=False).

    Parameters
    ----------
    y : (T,) float32 waveform at `sampling_rate`

    Returns
    -------
    (num_mels, n_frames) float32
    """
    mel_basis = _mel_scale_filterbank(sampling_rate, n_fft, num_mels, fmin, fmax).astype(np.float64)
    window = np.hanning(win_size + 1)[:-1].astype(np.float64)
    pad = (n_fft - hop_size) // 2
    padded = np.pad(y.astype(np.float64), (pad, pad), mode="reflect")
    n_frames = 1 + (len(padded) - n_fft) // hop_size
    idx = np.arange(n_fft)[None, :] + hop_size * np.arange(n_frames)[:, None]
    frames = padded[idx] * window
    spec = np.fft.rfft(frames, n=n_fft, axis=-1).T  # (n_fft//2+1, n_frames)
    magnitude = np.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9)
    mel = mel_basis @ magnitude
    mel = np.log(np.clip(mel, 1e-5, None))
    return mel.astype(np.float32)
