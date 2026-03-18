"""Unified configuration dataclasses for all sherox modules.

Each module owns its own dataclass; CLI arguments override defaults at runtime.
"""
from dataclasses import dataclass


# ── ASR ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Configuration for the ASR module (sherox.asr)."""

    model_dir: str = "models/zipformer-en-2023"
    sample_rate: int = 16000
    chunk_size: float = 0.16    # seconds (~2560 samples at 16 kHz)
    num_threads: int = 4

    # Passed directly to sherpa-onnx as the model_type hint.
    # Online:  "" (auto), transducer, zipformer, zipformer2, conformer, lstm,
    #          paraformer, ctc, wenet_ctc, zipformer2_ctc
    # Offline: "" (auto), transducer, nemo_transducer, paraformer, whisper,
    #          ctc, nemo_ctc, sense_voice, moonshine, fire_red_asr
    model_type: str = ""

    offline: bool = False

    # VAD — used when offline=True
    vad_type: str = "silero"            # "silero" | "ten-vad"
    vad_model: str = ""                 # resolved path, set by asr._validate_vad()
    ten_vad_model: str = "ten-vad.int8.onnx"
    vad_threshold: float = 0.5
    vad_min_silence_duration: float = 0.5
    vad_min_speech_duration: float = 0.25

    language: str = "en"
    show_mic_level: bool = False

    # Speaker diarization
    diarization: bool = False
    diarization_seg_model: str = ""
    diarization_emb_model: str = ""
    diarization_num_speakers: int = -1
    diarization_cluster_threshold: float = 0.5


# ── Segment ───────────────────────────────────────────────────────────────────

@dataclass
class SegmentConfig:
    """Configuration for the VAD segmentation module (sherox.segment)."""

    vad_type: str = "silero"            # "silero" (default) | "ten-vad"
    vad_model: str = ""                 # resolved path, set at runtime
    ten_vad_model: str = "ten-vad.int8.onnx"
    vad_threshold: float = 0.5
    vad_min_silence_duration: float = 0.5
    vad_min_speech_duration: float = 0.25

    sample_rate: int = 16000
    capture_rate: int = 16000           # mic capture rate; model resamples internally
    num_threads: int = 4
    chunk_size: float = 0.1             # seconds

    show_timestamps: bool = True        # print [start – end] per segment
    show_mic_level: bool = False
    output_dir: str = ""                # save segment wav clips here if set


# ── TTS ───────────────────────────────────────────────────────────────────────

@dataclass
class TtsConfig:
    """Configuration for the TTS module (sherox.tts)."""

    # Directory where the TTS model lives (auto-resolved from language if empty).
    model_dir: str = ""

    # ISO 639-3 language code.  Currently supported: "ind" (Indonesian).
    language: str = "ind"

    speaker_id: int = 0
    speed: float = 1.0
    num_threads: int = 4

    # Output file path.  Ignored when play=True and no explicit output given.
    output: str = "output.wav"

    # Play audio through the default output device instead of (or in addition to) saving.
    play: bool = False
