"""Unified configuration dataclasses for all sherox modules.

Each module owns its own dataclass; CLI arguments override defaults at runtime.
"""
from dataclasses import dataclass, field


# ── ASR ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Configuration for the ASR module (sherox.asr)."""

    model_dir: str = "models/parakeet-tdt-0.6b-v2-int8"
    sample_rate: int = 16000
    chunk_size: float = 0.16    # seconds (~2560 samples at 16 kHz)
    num_threads: int = 4

    # Passed directly to sherpa-onnx as the model_type hint.
    # Online:  "" (auto), transducer, zipformer, zipformer2, conformer, lstm,
    #          paraformer, ctc, wenet_ctc, zipformer2_ctc, multilingual_streaming
    # Offline: "" (auto), transducer, nemo_transducer, paraformer, whisper,
    #          ctc, nemo_ctc, sense_voice, moonshine, fire_red_asr, cohere_transcribe
    # ReazonSpeech (offline): ja, ja-en, ja-en-mls-5k
    model_type: str = ""

    offline: bool = False

    # VAD — used when offline=True
    vad_type: str = "silero"            # "silero" | "ten-vad"
    vad_model: str = ""                 # resolved path, set by asr._validate_vad()
    ten_vad_model: str = "ten-vad.int8.onnx"
    vad_threshold: float = 0.1
    vad_min_silence_duration: float = 0.5
    vad_min_speech_duration: float = 0.25
    vad_max_speech_duration: float = 20.0  # force-cut a segment past this length

    language: str = "en"
    show_mic_level: bool = True

    # Speaker diarization
    diarization: bool = False
    diarization_seg_model: str = ""
    diarization_emb_model: str = ""
    diarization_num_speakers: int = -1
    diarization_cluster_threshold: float = 0.5

    # Hardware / inference
    device: str = "cpu"              # ONNX Runtime provider: "cpu", "cuda", "coreml"

    # Audio pre-processing
    denoise: bool = False            # noise reduction pre-processing (offline WAV only)

    # Output / post-processing
    word_timestamps: bool = False    # display per-token timing after each segment
    punctuation: bool = False        # punctuation restoration post-processing
    punct_model: str = ""            # path to OfflinePunctuation model, resolved by asr.py
    translate: bool = False          # request English translation (Whisper multilingual only)

    # Terminal output options (see --no-color / --json in sherox.asr)
    no_color: bool = False           # disable ANSI colour codes in transcript output
    json_output: bool = False        # emit each segment as a JSON line instead of styled text

    # Latency diagnostics
    debug_latency: bool = False      # print per-segment endpoint→text timing to stderr


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
    show_mic_level: bool = True
    output_dir: str = ""                # save segment wav clips here if set


# ── Speaker Identification ───────────────────────────────────────────────────

@dataclass
class SidConfig:
    """Configuration for the Speaker Identification module (sherox.sid)."""

    model: str = "models/nemo_en_titanet_large.onnx"
    threshold: float = 0.6          # cosine similarity cutoff; below → "unknown"
    sample_rate: int = 16000        # expected rate for WAV input
    capture_rate: int = 16000       # mic capture rate (resampled internally)
    chunk_size: float = 0.1         # seconds per mic chunk
    num_threads: int = 4
    vad_model: str = ""             # resolved path to silero_vad.onnx
    vad_threshold: float = 0.3      # VAD sensitivity (higher = less sensitive)
    vad_min_silence_duration: float = 1.0   # seconds of silence to end segment
    vad_min_speech_duration: float = 1.0    # minimum speech for a valid segment
    wav: str = ""                   # path to input WAV file (--wav mode)
    show_mic_level: bool = True


# ── Spoken Language Identification ───────────────────────────────────────────

@dataclass
class LidConfig:
    """Configuration for the Spoken Language Identification module (sherox.lid).

    Uses multilingual Whisper encoder/decoder ONNX pairs published by sherpa-onnx.
    """

    # Resolved paths set by lid._resolve_model()
    encoder: str = ""
    decoder: str = ""

    # Whisper variant — controls which archive is auto-downloaded:
    #   tiny | base | small | medium
    size: str = "tiny"

    num_threads: int = 4
    provider: str = "cpu"           # "cpu" | "cuda" | "coreml"
    sample_rate: int = 16000        # expected rate for WAV input
    capture_rate: int = 16000       # mic capture rate (resampled internally)
    chunk_size: float = 0.1         # seconds per mic chunk
    vad_model: str = ""             # resolved path to silero_vad.onnx (mic mode)
    wav: str = ""                   # path to input WAV file (--wav mode)
    show_mic_level: bool = True


# ── TTS ───────────────────────────────────────────────────────────────────────

@dataclass
class TtsConfig:
    """Configuration for the TTS module (sherox.tts)."""

    # Directory where the TTS model lives (auto-resolved from language if empty).
    model_dir: str = ""

    # ISO 639-3 language code.
    # Supported: "eng", "deu", "fra", "spa", "ind", "zho", "jpn".
    # Short aliases are also accepted (e.g., "en", "de", "zh", "ja").
    language: str = "eng"

    speaker_id: int = 0
    speed: float = 1.0
    num_threads: int = 4

    # Output file path.  Use "none" or "-" with play=True to disable saving.
    output: str = "output.wav"

    # Play audio through the default output device instead of (or in addition to) saving.
    play: bool = False

    # Do not write a WAV file. Requires play=True.
    no_save: bool = False


# ── KWS ──────────────────────────────────────────────────────────────────────

@dataclass
class KwsConfig:
    """Configuration for the keyword spotting module (sherox.kws).

    The default model is the 3.3 M-parameter Zipformer trained on GigaSpeech
    (English).  Supply ``model_dir`` to use a custom model directory.

    Keywords are specified either as a comma-separated string
    (``keywords_str``) or as a path to a plain-text file with one keyword per
    line (``keywords_file``).  ``keywords_str`` takes priority when both are
    set.

    Example::

        cfg = KwsConfig(keywords_str="hey sherpa, ok google")
    """

    model_dir: str = ""

    # Comma-separated keywords (e.g. "hey sherpa, ok google").
    # The string is converted to a temporary file before being passed to
    # sherpa-onnx which expects a file path.
    keywords_str: str = ""

    # Path to a plain-text file; one keyword per line.
    keywords_file: str = ""

    sample_rate: int = 16000
    chunk_size: float = 0.1     # seconds of audio per decode call
    num_threads: int = 4

    # Microphone capture rate (may differ from model sample_rate).
    capture_rate: int = 16000

    max_active_paths: int = 4    # beam width for the keyword spotter
    keywords_score: float = 1.0  # boost score for each keyword token
    keywords_threshold: float = 0.25  # higher = harder to trigger
    num_trailing_blanks: int = 1  # blank tokens required after keyword before firing

    # WAV input path (empty means microphone mode).
    wav: str = ""

    show_mic_level: bool = True
    verbose: bool = False


# ── Wake ──────────────────────────────────────────────────────────────────────

@dataclass
class WakeConfig:
    """Configuration for the wake-word module (sherox.wake).

    Built on top of ``livekit-wakeword``.  Supply one or more ONNX model
    paths in ``model_paths`` — each path becomes an independent wake-word
    detector.  Models are typically produced via the ``livekit-wakeword``
    training pipeline (see https://github.com/livekit/livekit-wakeword).

    Example::

        cfg = WakeConfig(
            model_paths=["models/hey_livekit.onnx"],
            threshold=0.5,
        )
    """

    # Paths to one or more ONNX wake-word models.
    model_paths: list[str] = field(default_factory=list)

    # Detection threshold (0.0 - 1.0; higher = fewer false positives).
    threshold: float = 0.5

    # Minimum seconds between detections of the same wake word.
    debounce: float = 2.0

    # Audio chunk duration per inference call (seconds).
    chunk_size: float = 2.0

    # WAV input path (empty means microphone mode).
    wav: str = ""
