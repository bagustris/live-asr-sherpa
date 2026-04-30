"""
Resampler quality & speed benchmark.

Metrics (all relevant for speech/ASR quality)
----------------------------------------------
SNR_band  Round-trip SNR using only frequencies BELOW the 16 kHz target Nyquist
          (8 000 Hz).  Higher = less distortion in the audible pass-band.

Edge_SNR  SNR measured only on the first and last 100 ms (edge artifact region).
          Polyphase resampling can ring at edges; FFT resampling wraps around.
          Higher = cleaner edges.

PR_dB     Pass-band ripple: max gain deviation (dB) from 0 dB for a 1 kHz tone
          after a down-then-up round-trip.  Closer to 0 = flatter response.

Speed     Median wall-clock ms for 10 × 30-second 44 100 → 16 000 Hz down-samples.

Candidates
----------
1. scipy.signal.resample      – FFT-based (periodic assumption)
2. scipy.signal.resample_poly – polyphase FIR, default window (β=5)
3. scipy.signal.resample_poly – polyphase FIR, Kaiser β=14 + padtype=line  (current)
4. resampy                    – high-quality sinc (third-party)
5. soxr "HQ"                  – SoX resampler, HQ quality (third-party)
6. soxr "VHQ"                 – SoX resampler, VHQ quality (third-party)
"""

import time
from math import gcd

import numpy as np
from scipy.signal import chirp, resample as sp_resample, resample_poly

# ── test parameters ───────────────────────────────────────────────────────────

ORIG_SR   = 44_100
TARGET_SR = 16_000
DURATION  = 30          # seconds
N_REPEAT  = 10          # timing repetitions
EDGE_MS   = 100         # ms to examine at each edge

g    = gcd(TARGET_SR, ORIG_SR)
UP   = TARGET_SR // g          # 160
DOWN = ORIG_SR   // g          # 441

# Band-limited chirp: only 20 Hz → 7 900 Hz (safely below 8 000 Hz Nyquist).
# Round-trip SNR should be high because no information is discarded.
t = np.linspace(0, DURATION, ORIG_SR * DURATION, endpoint=False, dtype=np.float32)
signal_44k = chirp(t, f0=20, f1=7_900, t1=DURATION, method="logarithmic").astype(np.float32)

# Low-pass mask: keep only ≤ 7 900 Hz content in the round-trip reference
# (the up-sampler may add a tiny amount of content in 7 900–8 000 Hz range)
def _lowpass_ref(x: np.ndarray, sr: int, cutoff: float = 7_900.0) -> np.ndarray:
    """Brick-wall FFT low-pass filter for reference alignment."""
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    X[freqs > cutoff] = 0
    return np.fft.irfft(X, n=len(x)).astype(np.float32)

ref = _lowpass_ref(signal_44k, ORIG_SR)          # ideal reference after round-trip


# ── metric helpers ────────────────────────────────────────────────────────────

def snr_full(original: np.ndarray, reconstructed: np.ndarray) -> float:
    n = min(len(original), len(reconstructed))
    sig, rec = original[:n], reconstructed[:n]
    noise = sig - rec
    p_sig   = np.mean(sig   ** 2)
    p_noise = np.mean(noise ** 2)
    return 10 * np.log10(p_sig / (p_noise + 1e-30))


def snr_edge(original: np.ndarray, reconstructed: np.ndarray, sr: int) -> float:
    """SNR restricted to the first + last EDGE_MS ms (edge artifact zone)."""
    edge_n = int(sr * EDGE_MS / 1000)
    n = min(len(original), len(reconstructed))
    head_o, head_r = original[:edge_n],     reconstructed[:edge_n]
    tail_o, tail_r = original[n - edge_n:n], reconstructed[n - edge_n:n]
    sig   = np.concatenate([head_o, tail_o])
    rec   = np.concatenate([head_r, tail_r])
    noise = sig - rec
    return 10 * np.log10(np.mean(sig**2) / (np.mean(noise**2) + 1e-30))


def passband_ripple_db(down_fn, up_fn) -> float:
    """Gain deviation at 1 kHz after round-trip (closer to 0 dB is better)."""
    tone_sr   = ORIG_SR
    tone_freq = 1_000.0
    n_samples = tone_sr * 5   # 5 seconds
    t_tone = np.arange(n_samples, dtype=np.float32) / tone_sr
    tone = np.sin(2 * np.pi * tone_freq * t_tone).astype(np.float32)
    down = down_fn(tone)
    up   = up_fn(down, len(tone))
    n = min(len(tone), len(up))
    rms_in  = np.sqrt(np.mean(tone[:n] ** 2))
    rms_out = np.sqrt(np.mean(up[:n]   ** 2))
    return 20 * np.log10(rms_out / (rms_in + 1e-30))


def time_median_ms(fn, n=N_REPEAT) -> float:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


# ── candidate definitions ─────────────────────────────────────────────────────

def down_fft(x):
    n_new = int(len(x) * TARGET_SR / ORIG_SR)
    return sp_resample(x, n_new).astype(np.float32)

def up_fft(x, n_target):
    return sp_resample(x, n_target).astype(np.float32)


def down_poly5(x):
    return resample_poly(x, UP, DOWN).astype(np.float32)

def up_poly5(x, n_target):
    return resample_poly(x, DOWN, UP).astype(np.float32)


def down_poly14(x):
    return resample_poly(x, UP, DOWN, window=("kaiser", 14.0), padtype="line").astype(np.float32)

def up_poly14(x, n_target):
    return resample_poly(x, DOWN, UP, window=("kaiser", 14.0), padtype="line").astype(np.float32)


candidates: dict = {
    "scipy.resample (FFT)":         (down_fft,     up_fft),
    "resample_poly (β=5, default)": (down_poly5,   up_poly5),
    "resample_poly (β=14+line)":    (down_poly14,  up_poly14),
}

try:
    import resampy
    def down_resampy(x):
        return resampy.resample(x, ORIG_SR, TARGET_SR).astype(np.float32)
    def up_resampy(x, n_target):
        return resampy.resample(x, TARGET_SR, ORIG_SR).astype(np.float32)
    candidates["resampy"] = (down_resampy, up_resampy)
except ImportError:
    print("resampy not installed — skipping")

try:
    import soxr
    for quality in ("HQ", "VHQ"):
        q = quality  # capture for closure
        def _mk_down(q):
            def _down(x): return soxr.resample(x, ORIG_SR, TARGET_SR, quality=q).astype(np.float32)
            return _down
        def _mk_up(q):
            def _up(x, n): return soxr.resample(x, TARGET_SR, ORIG_SR, quality=q).astype(np.float32)
            return _up
        candidates[f"soxr ({quality})"] = (_mk_down(quality), _mk_up(quality))
except ImportError:
    print("soxr not installed — skipping")


# ── run ───────────────────────────────────────────────────────────────────────

print(f"\n{'Method':<38} {'SNR_band':>10} {'Edge_SNR':>10} {'PR (dB)':>9} {'ms/30s':>9}")
print("-" * 82)

for name, (down_fn, up_fn) in candidates.items():
    # Quality metrics
    down16k = down_fn(signal_44k)
    up44k   = up_fn(down16k, len(signal_44k))

    band_snr  = snr_full(ref, up44k)
    edge_snr  = snr_edge(ref, up44k, ORIG_SR)
    pr        = passband_ripple_db(down_fn, up_fn)

    # Speed
    speed_ms = time_median_ms(lambda f=down_fn: f(signal_44k))

    flag = " ← current" if name == "resample_poly (β=14+line)" else ""
    print(f"{name:<38} {band_snr:>10.1f} {edge_snr:>10.1f} {pr:>9.2f} {speed_ms:>9.1f}{flag}")

print()
print("SNR_band : round-trip SNR for band-limited signal (dB) — higher is better")
print("Edge_SNR : SNR at first+last 100 ms only            (dB) — higher is better")
print("PR (dB)  : 1 kHz gain after round-trip              (dB) — 0.0 is perfect")
print("ms/30s   : median wall-clock time per 30 s clip      (ms) — lower is faster")
