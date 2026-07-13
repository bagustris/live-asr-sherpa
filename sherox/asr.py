"""Streaming ASR — entry point.

Usage:
    # Parakeet English int8 (default, offline, auto-detects model type):
    sherox.asr --mic
    sherox.asr --wav path/to/audio.wav

    # Hint the architecture explicitly (speeds up model loading):
    sherox.asr --mic --model-type zipformer2

    # Parakeet TDT fp16 (offline, better accuracy):
    sherox.asr --mic --offline --model-type nemo_transducer

    # Parakeet TDT int8 (offline, smaller & faster, slightly less accurate):
    sherox.asr --mic --offline --model-type nemo_transducer --model-dir models/parakeet-tdt-0.6b-v2-int8

    # Whisper (offline):
    sherox.asr --mic --offline --model-type whisper --language en

    # SenseVoice (offline):
    sherox.asr --mic --offline --model-type sense_voice

    # German streaming (online, auto-downloaded):
    sherox.asr --mic --lang de
    sherox.asr --wav audio.wav --lang de

    # German NeMo CTC (offline, auto-downloaded):
    sherox.asr --mic --lang de --offline
    sherox.asr --wav audio.wav --lang de --offline

    # ReazonSpeech Japanese (offline):
    sherox.asr --mic --model-type reazonspeech-ja
    sherox.asr --wav path/to/audio.wav --model-type reazonspeech-ja

    # ReazonSpeech bilingual Japanese-English (offline):
    sherox.asr --mic --model-type reazonspeech-ja-en
    sherox.asr --wav path/to/audio.wav --model-type reazonspeech-ja-en

    # ReazonSpeech bilingual trained on ReazonSpeech + MLS English 5k (offline):
    sherox.asr --mic --model-type reazonspeech-ja-en-mls-5k
    sherox.asr --wav path/to/audio.wav --model-type reazonspeech-ja-en-mls-5k

    # NeMo Parakeet CTC Japanese (offline, 0.6B int8, 35k vocab):
    sherox.asr --mic --model-type parakeet-ctc-ja
    sherox.asr --wav path/to/audio.wav --model-type parakeet-ctc-ja

    # Cohere Transcribe multilingual (offline, 14 languages):
    sherox.asr --mic --offline --model-type cohere_transcribe --language en
    sherox.asr --wav path/to/audio.wav --offline --model-type cohere_transcribe --language zh

    # Multilingual streaming zipformer (online, 9 languages including Indonesian):
    sherox.asr --mic --model-type multilingual_streaming
    sherox.asr --wav path/to/audio.wav --model-type multilingual_streaming
    sherox.asr --wav data/saya_suka_id.wav --model-type multilingual_streaming

    # Custom model directory:
    sherox.asr --mic --model-dir models/my-model --offline --model-type nemo_transducer

    # Speaker diarization (offline, auto-downloads lightweight models):
    sherox.asr --mic --offline --diarization

    # Speaker diarization with known speaker count:
    sherox.asr --mic --offline --diarization --num-speakers 2

    # Speaker diarization with [Speaker N] tag prefix:
    sherox.asr --mic --offline --diarization --speaker-tag

    Models are stored under  models/<model-name>/  at the project root:
      models/parakeet-tdt-0.6b-v2-int8/    (offline, int8, default English)
      models/zipformer-en-2023/            (online transducer)
      models/parakeet-tdt-0.6b-v2/         (offline, fp16 — larger, more accurate)
      models/parakeet-tdt-0.6b-v2-int8/    (offline, int8 — smaller & faster)
      models/zipformer-de-2025/            (online streaming, default German)
      models/nemo-de-int8/                 (offline, default German --offline)
      models/reazonspeech-ja/              (offline, ReazonSpeech Japanese)
      models/reazonspeech-ja-en/           (offline, ReazonSpeech bilingual ja-en)
      models/reazonspeech-ja-en-mls-5k/    (offline, ReazonSpeech + MLS 5k bilingual)
      models/parakeet-ctc-ja-int8/         (offline, NeMo Parakeet CTC Japanese int8)
      models/cohere-transcribe-14-lang-int8/  (offline, Cohere Transcribe multilingual)
      models/zipformer-multilingual-2025-02-10/  (online, 9 languages including Indonesian)
      models/silero_vad.onnx               (VAD, shared for offline use)
      models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx  (diarization segmentation)
      models/nemo_en_speakerverification_speakernet.onnx        (diarization embedding)

    Online --model-type values:  (blank), transducer, zipformer, zipformer2,
                                 conformer, lstm, paraformer, ctc, wenet_ctc,
                                 zipformer2_ctc, multilingual_streaming
    Offline --model-type values: (blank), transducer, nemo_transducer, paraformer,
                                 whisper, ctc, nemo_ctc, sense_voice, moonshine,
                                 fire_red_asr, cohere_transcribe, parakeet-ctc-ja,
                                 reazonspeech-ja, reazonspeech-ja-en, reazonspeech-ja-en-mls-5k
"""

import argparse
from contextlib import nullcontext
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from . import ConfigError
from .asr_engine import build_diarization, build_offline_recognizer, build_punctuation, build_recognizer, build_vad
from .utils import download_file as _download_file
from .utils import run_cli as _run_cli
from .utils import safe_tar_members as _safe_tar_members
from .audio import denoise_gen, mic_stream, pipe_stream, read_wav, wav_duration
from .config import Config
from .streaming import run_offline_vad_streaming, run_streaming, write_srt, write_vtt, write_txt

sf = SimpleNamespace(SoundFile=None)

_console = Console()
_err_console = Console(stderr=True)
# When --json is active, info messages are redirected to stderr so that
# stdout carries only clean JSON lines.
_json_mode: bool = False


def _info(msg: str) -> None:
    dest = _err_console if _json_mode else _console
    dest.print(f"[bold green]\\[info][/bold green] {msg}")


def _error(msg: str) -> None:
    _err_console.print(f"[bold red]\\[error][/bold red] {msg}")
    raise ConfigError(msg)


def _require_soundfile():
    global sf
    if getattr(sf, "SoundFile", None) is not None:
        return sf
    try:
        import soundfile as _soundfile  # noqa: PLC0415
    except ImportError:  # pragma: no cover - depends on environment
        _error(
            "soundfile is required for reading audio files. "
            "Install it with: pip install soundfile"
        )
    sf = _soundfile
    return sf


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if args.sample_rate <= 0:
        _error(f"--sample-rate must be > 0, got {args.sample_rate}")
    if args.capture_rate <= 0:
        _error(f"--capture-rate must be > 0, got {args.capture_rate}")
    if args.capture_rate < args.sample_rate:
        _error(
            f"--capture-rate ({args.capture_rate}) must be >= --sample-rate ({args.sample_rate}). "
            "Use --capture-rate 48000 for better device compatibility."
        )
    if args.chunk_size <= 0:
        _error(f"--chunk-size must be > 0, got {args.chunk_size}")
    if args.threads <= 0:
        _error(f"--threads must be > 0, got {args.threads}")
    if args.speaker_tag and not args.diarization:
        _error("--speaker-tag requires --diarization")
    if args.num_speakers == 0 or args.num_speakers < -1:
        _error("--num-speakers must be -1 (auto) or a positive integer")
    if args.denoise and not args.wav:
        _error("--denoise is only supported with --wav (not --mic or --pipe)")
    if args.output and args.wav and len(args.wav) > 1:
        _error("--output can only be used with a single --wav file; use --output-dir for batch mode")
    if args.output_dir and not args.wav:
        _error("--output-dir requires --wav (use --output for single-file pipe output)")
    if args.translate:
        if not args.offline:
            _error(
                "--translate requires --offline "
                "(Whisper is an offline model; add --offline to enable it)"
            )
        if args.model_type.lower() != "whisper":
            _error(
                "--translate requires --model-type whisper "
                "(only multilingual Whisper supports translation; "
                "English-only *.en Whisper models do not)"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming ASR with Sherpa-ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mic", action="store_true", help="Stream from microphone")
    mode.add_argument("--wav", nargs="+", metavar="PATH", help="Transcribe WAV file(s)")
    mode.add_argument(
        "--pipe",
        action="store_true",
        help=(
            "Read raw 16-bit little-endian mono PCM from stdin. "
            "Use --capture-rate to match the input stream's sample rate. "
            "Example: arecord -f S16_LE -r 16000 -c 1 | sherox.asr --pipe"
        ),
    )

    parser.add_argument(
        "--model-dir",
        default=None,
        metavar="PATH",
        help=(
            "Path to the model directory. "
            "Default: models/parakeet-tdt-0.6b-v2-int8 for English, "
            "or a language-specific model when --lang/--language is set."
        ),
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate (Hz)")
    parser.add_argument(
        "--chunk-size", type=float, default=0.16, help="Chunk size in seconds (0.1–0.2 recommended)"
    )
    parser.add_argument("--threads", type=int, default=4, help="CPU thread count for ONNX runtime")
    parser.add_argument(
        "--model-type",
        default="",
        metavar="TYPE",
        help=(
            "Model architecture hint passed to sherpa-onnx. Leave blank for auto-detect. "
            "Online: transducer, zipformer, zipformer2, conformer, lstm, paraformer, "
            "ctc, wenet_ctc, zipformer2_ctc, multilingual_streaming. "
            "Offline: transducer, nemo_transducer, paraformer, whisper, ctc, nemo_ctc, "
            "sense_voice, moonshine, fire_red_asr, cohere_transcribe. "
            "ReazonSpeech (offline): reazonspeech-ja (Japanese), reazonspeech-ja-en (bilingual Japanese-English), "
            "reazonspeech-ja-en-mls-5k (bilingual trained on ReazonSpeech + MLS English 5k). "
            "NeMo Parakeet CTC (offline): parakeet-ctc-ja (Japanese 0.6B int8, default for --language ja). "
            "See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/"
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the offline (VAD-segmented) pipeline instead of streaming (required for Whisper, NeMo, SenseVoice, etc.)",
    )
    parser.add_argument(
        "--capture-rate",
        type=int,
        default=16000,
        metavar="HZ",
        help="Microphone capture sample rate (Hz). Use 48000 for better device compatibility; sherpa-onnx resamples internally.",
    )
    parser.add_argument(
        "--vad-model",
        dest="vad_type",
        default="silero",
        choices=["silero", "ten-vad"],
        help="VAD model type to use for offline segmentation (default: silero).",
    )
    parser.add_argument(
        "--ten-vad-model",
        default="ten-vad.int8.onnx",
        choices=["ten-vad.onnx", "ten-vad.int8.onnx"],
        help=(
            "Ten-VAD model variant to use when --vad-model is ten-vad "
            "(default: ten-vad.int8.onnx)."
        ),
    )
    parser.add_argument(
        "--language",
        "--lang",
        default="en",
        metavar="LANG",
        help="Language code for Whisper and SenseVoice models (e.g. en, zh, ja)",
    )
    parser.add_argument(
        "--no-mic-level",
        action="store_true",
        help="Suppress the live RMS energy bar during microphone capture",
    )
    parser.add_argument(
        "--diarization",
        action="store_true",
        help=(
            "Enable speaker diarization. Colours each speaker's output differently. "
            "Works with both online and offline pipelines. "
            "Diarization models are auto-downloaded on first use."
        ),
    )
    parser.add_argument(
        "--speaker-tag",
        action="store_true",
        help=(
            "Show a [Speaker N] prefix before each diarized utterance "
            "(requires --diarization). By default only the text colour differs per speaker."
        ),
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=-1,
        metavar="N",
        help=(
            "Known number of speakers for diarization (-1 = auto-detect via "
            "clustering threshold). Providing the correct count improves accuracy."
        ),
    )
    parser.add_argument(
        "--diarization-seg-model",
        default="",
        metavar="PATH",
        help="Path to pyannote segmentation model.onnx (auto-downloaded if not provided)",
    )
    parser.add_argument(
        "--diarization-emb-model",
        default="",
        metavar="PATH",
        help="Path to speaker embedding extractor .onnx (auto-downloaded if not provided)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "coreml"],
        help="ONNX Runtime execution provider for the ASR model (default: cpu)",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply noise reduction to WAV input before transcription (requires pip install sherox[denoise])",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help=(
            "Output English translation instead of transcription. "
            "Requires --offline and --model-type whisper "
            "(multilingual Whisper only; not supported by English-only *.en models). "
            "Example: sherox.asr --wav speech.wav --offline --model-type whisper "
            "--language zh --translate"
        ),
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Print per-token timing after each segment (model-dependent)",
    )
    parser.add_argument(
        "--punctuation",
        action="store_true",
        help="Restore punctuation using a CT-Transformer model (auto-downloaded). Primarily trained on zh+en.",
    )
    parser.add_argument(
        "--punct-model",
        default="",
        metavar="PATH",
        help="Path to OfflinePunctuation model directory (auto-downloaded if not set and --punctuation is used)",
    )
    parser.add_argument(
        "--output",
        default="",
        metavar="PATH",
        help="Output file for transcription (format inferred from extension: .srt, .vtt, .txt; single WAV only)",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        metavar="DIR",
        help="Output directory for batch transcription (one file per WAV, format from --output-format)",
    )
    parser.add_argument(
        "--output-format",
        default="txt",
        choices=["txt", "srt", "vtt"],
        help="Subtitle/transcript format for --output-dir (default: txt)",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Suppress intermediate partial transcripts; print only finalised segments",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        dest="no_color",
        help=(
            "Disable ANSI colour codes in transcript output. "
            "Useful when redirecting to a file or piping to tools that "
            "do not interpret colour escapes."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help=(
            "Emit each finalised segment as a JSON line to stdout. "
            "Format: {\"type\": \"segment\", \"text\": \"...\", "
            "\"start\": 0.0, \"end\": 1.5} "
            "(\"speaker\" key added when --diarization is active). "
            "Partial hypotheses are suppressed in this mode. "
            "Example: sherox.asr --mic --json | jq -r '.text'"
        ),
    )
    return parser.parse_args()


_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
)
_MODEL_ARCHIVE = "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
_MODEL_EXTRACTED = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
_MODEL_TARGET = "zipformer-en-2023"

_PARAKEET_FP16_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16.tar.bz2"
)
_PARAKEET_FP16_ARCHIVE = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16.tar.bz2"
_PARAKEET_FP16_EXTRACTED = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16"
_PARAKEET_FP16_TARGET = "parakeet-tdt-0.6b-v2"

_PARAKEET_INT8_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
)
_PARAKEET_INT8_ARCHIVE = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
_PARAKEET_INT8_EXTRACTED = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
_PARAKEET_INT8_TARGET = "parakeet-tdt-0.6b-v2-int8"

# Default English model (offline int8)
_PARAKEET_TARGET = _PARAKEET_INT8_TARGET

# ── ReazonSpeech model URLs ───────────────────────────────────────────────────
# ja: Japanese-only model (https://huggingface.co/reazon-research/reazonspeech-k2-v2)
_REAZON_JA_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2"
)
_REAZON_JA_ARCHIVE = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2"
_REAZON_JA_EXTRACTED = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"
_REAZON_JA_TARGET = "reazonspeech-ja"

# ja-en: bilingual Japanese-English model
# (https://huggingface.co/reazon-research/reazonspeech-k2-v2-ja-en)
# ja-en-mls-5k: bilingual trained on ReazonSpeech + MLS English 5k hours
# (https://huggingface.co/reazon-research/reazonspeech-k2-v2-ja-en-mls-5k-corrected)
# Both are served from the same sherpa-onnx release archive.
_REAZON_JA_EN_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17.tar.bz2"
)
_REAZON_JA_EN_ARCHIVE = "sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17.tar.bz2"
_REAZON_JA_EN_EXTRACTED = "sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17"
_REAZON_JA_EN_TARGET = "reazonspeech-ja-en"
_REAZON_JA_EN_MLS_TARGET = "reazonspeech-ja-en-mls-5k"

# ── NeMo Parakeet CTC Japanese model URLs ─────────────────────────────────────
# Japanese NeMo CTC model (0.6B parameters, 35k vocabulary, int8 quantized)
_PARAKEET_CTC_JA_INT8_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8.tar.bz2"
)
_PARAKEET_CTC_JA_INT8_ARCHIVE = "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8.tar.bz2"
_PARAKEET_CTC_JA_INT8_EXTRACTED = "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8"
_PARAKEET_CTC_JA_INT8_TARGET = "parakeet-ctc-ja-int8"

# ── NeMo CTC English model URLs ────────────────────────────────────────────────
# English NeMo Conformer CTC model (~158 MB). Auto-download default for
# `--model-type nemo_ctc --language en`; exposes per-word confidence via CTC
# token posteriors, unlike the Parakeet transducer default.
_NEMO_CTC_EN_MEDIUM_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-nemo-ctc-en-conformer-medium.tar.bz2"
)
_NEMO_CTC_EN_MEDIUM_ARCHIVE = "sherpa-onnx-nemo-ctc-en-conformer-medium.tar.bz2"
_NEMO_CTC_EN_MEDIUM_EXTRACTED = "sherpa-onnx-nemo-ctc-en-conformer-medium"
_NEMO_CTC_EN_MEDIUM_TARGET = "sherpa-onnx-nemo-ctc-en-conformer-medium"

# Lighter variant; select by passing --model-dir models/<small target> explicitly.
_NEMO_CTC_EN_SMALL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-nemo-ctc-en-conformer-small.tar.bz2"
)
_NEMO_CTC_EN_SMALL_ARCHIVE = "sherpa-onnx-nemo-ctc-en-conformer-small.tar.bz2"
_NEMO_CTC_EN_SMALL_EXTRACTED = "sherpa-onnx-nemo-ctc-en-conformer-small"
_NEMO_CTC_EN_SMALL_TARGET = "sherpa-onnx-nemo-ctc-en-conformer-small"

# ── Cohere Transcribe model URLs ──────────────────────────────────────────────
# Multilingual model supporting 14 languages
# (https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
_COHERE_TRANSCRIBE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2"
)
_COHERE_TRANSCRIBE_ARCHIVE = "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2"
_COHERE_TRANSCRIBE_EXTRACTED = "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01"
_COHERE_TRANSCRIBE_TARGET = "cohere-transcribe-14-lang-int8"

# ── Whisper Large-V3 model URLs ───────────────────────────────────────────────
# Multilingual Whisper model (~3 GB). Triggered only when --model-dir matches the
# canonical directory name; `--model-type whisper` alone is ambiguous since other
# Whisper variants (tiny/base/small/medium) share the same type.
_WHISPER_LARGE_V3_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-whisper-large-v3.tar.bz2"
)
_WHISPER_LARGE_V3_ARCHIVE = "sherpa-onnx-whisper-large-v3.tar.bz2"
_WHISPER_LARGE_V3_EXTRACTED = "sherpa-onnx-whisper-large-v3"
_WHISPER_LARGE_V3_TARGET = "sherpa-onnx-whisper-large-v3"

_WHISPER_TURBO_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-whisper-turbo.tar.bz2"
)
_WHISPER_TURBO_ARCHIVE = "sherpa-onnx-whisper-turbo.tar.bz2"
_WHISPER_TURBO_EXTRACTED = "sherpa-onnx-whisper-turbo"
_WHISPER_TURBO_TARGET = "sherpa-onnx-whisper-turbo"

_WHISPER_DISTIL_LARGE_V35_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-whisper-distil-large-v3.5.tar.bz2"
)
_WHISPER_DISTIL_LARGE_V35_ARCHIVE = "sherpa-onnx-whisper-distil-large-v3.5.tar.bz2"
_WHISPER_DISTIL_LARGE_V35_EXTRACTED = "sherpa-onnx-whisper-distil-large-v3.5"
_WHISPER_DISTIL_LARGE_V35_TARGET = "sherpa-onnx-whisper-distil-large-v3.5"

# ── SenseVoice model URLs ─────────────────────────────────────────────────────
# Multilingual model: zh / en / ja / ko / yue
_SENSE_VOICE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
)
_SENSE_VOICE_ARCHIVE = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
_SENSE_VOICE_EXTRACTED = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
_SENSE_VOICE_TARGET = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"

# ── Multilingual streaming zipformer model URLs ────────────────────────────────
# Streaming multilingual model supporting 9 languages: Arabic, English, Indonesian,
# Japanese, Russian, Thai, Vietnamese, Chinese (simplified & traditional)
_MULTILINGUAL_STREAMING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10.tar.bz2"
)
_MULTILINGUAL_STREAMING_ARCHIVE = "sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10.tar.bz2"
_MULTILINGUAL_STREAMING_EXTRACTED = "sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10"
_MULTILINGUAL_STREAMING_TARGET = "zipformer-multilingual-2025-02-10"

# ── German model URLs ─────────────────────────────────────────────────────────
# German streaming zipformer (online, default for --lang de)
_GERMAN_STREAMING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06.tar.bz2"
)
_GERMAN_STREAMING_ARCHIVE = "sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06.tar.bz2"
_GERMAN_STREAMING_EXTRACTED = "sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06"
_GERMAN_STREAMING_TARGET = "zipformer-de-2025"

# German NeMo FastConformer CTC (offline, default for --lang de --offline)
_GERMAN_NEMO_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-nemo-stt_de_fastconformer_hybrid_large_pc-int8.tar.bz2"
)
_GERMAN_NEMO_ARCHIVE = "sherpa-onnx-nemo-stt_de_fastconformer_hybrid_large_pc-int8.tar.bz2"
_GERMAN_NEMO_EXTRACTED = "sherpa-onnx-nemo-stt_de_fastconformer_hybrid_large_pc-int8"
_GERMAN_NEMO_TARGET = "nemo-de-int8"

_LANGUAGE_ALIASES = {
    "eng": "en",
    "jpn": "ja",
    "jp": "ja",
    "cmn": "zh",
    "zho": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "ind": "id",
    "deu": "de",
    "ger": "de",
    "deutsch": "de",
    "german": "de",
    "de-de": "de",
    "de-at": "de",
    "de-ch": "de",
}

_ASR_LANGUAGE_DEFAULTS = {
    "en": ("", _PARAKEET_TARGET),
    "ja": ("parakeet-ctc-ja", _PARAKEET_CTC_JA_INT8_TARGET),
    "id": ("multilingual_streaming", _MULTILINGUAL_STREAMING_TARGET),
    "zh": ("multilingual_streaming", _MULTILINGUAL_STREAMING_TARGET),
    "ar": ("multilingual_streaming", _MULTILINGUAL_STREAMING_TARGET),
    "ru": ("multilingual_streaming", _MULTILINGUAL_STREAMING_TARGET),
    "th": ("multilingual_streaming", _MULTILINGUAL_STREAMING_TARGET),
    "vi": ("multilingual_streaming", _MULTILINGUAL_STREAMING_TARGET),
    # de is handled explicitly in _resolve_default_model (online vs offline differ)
}

_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/silero_vad.onnx"
)

_TEN_VAD_MODEL_URLS = {
    "ten-vad.onnx": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "asr-models/ten-vad.onnx"
    ),
    "ten-vad.int8.onnx": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "asr-models/ten-vad.int8.onnx"
    ),
}

# ── Diarization model URLs (lightest available models) ───────────────────────
_DIAR_SEG_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
_DIAR_SEG_ARCHIVE = "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
_DIAR_SEG_EXTRACTED = "sherpa-onnx-pyannote-segmentation-3-0"
_DIAR_SEG_MODEL_FILE = "model.onnx"

# Lightest speaker embedding extractor (~22 MB)
_DIAR_EMB_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/nemo_en_speakerverification_speakernet.onnx"
)
_DIAR_EMB_FILE = "nemo_en_speakerverification_speakernet.onnx"

# ── Punctuation model URLs ────────────────────────────────────────────────────
_PUNCT_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2"
)
_PUNCT_ARCHIVE = "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2"
_PUNCT_EXTRACTED = "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"
_PUNCT_TARGET = "punct-ct-transformer-zh-en"


def _download_model(model_dir: str, model_type: str) -> None:
    """Download and extract the default model for the given model_type."""
    model_dir = Path(model_dir)

    # ReazonSpeech Japanese model
    if model_type == "reazonspeech-ja" or model_dir.name == _REAZON_JA_TARGET:
        url = _REAZON_JA_URL
        archive_name = _REAZON_JA_ARCHIVE
        extracted_name = _REAZON_JA_EXTRACTED
    # ReazonSpeech bilingual ja-en and ja-en-mls-5k (same sherpa-onnx archive)
    elif model_type in ("reazonspeech-ja-en", "reazonspeech-ja-en-mls-5k") or model_dir.name in (
        _REAZON_JA_EN_TARGET, _REAZON_JA_EN_MLS_TARGET
    ):
        url = _REAZON_JA_EN_URL
        archive_name = _REAZON_JA_EN_ARCHIVE
        extracted_name = _REAZON_JA_EN_EXTRACTED
    # NeMo Parakeet CTC Japanese model
    elif model_type == "parakeet-ctc-ja" or model_dir.name == _PARAKEET_CTC_JA_INT8_TARGET:
        url = _PARAKEET_CTC_JA_INT8_URL
        archive_name = _PARAKEET_CTC_JA_INT8_ARCHIVE
        extracted_name = _PARAKEET_CTC_JA_INT8_EXTRACTED
    # Cohere Transcribe multilingual model
    elif model_type == "cohere_transcribe" or model_dir.name == _COHERE_TRANSCRIBE_TARGET:
        url = _COHERE_TRANSCRIBE_URL
        archive_name = _COHERE_TRANSCRIBE_ARCHIVE
        extracted_name = _COHERE_TRANSCRIBE_EXTRACTED
    # Whisper Large-V3 (only triggered by canonical dir name — `whisper` type is ambiguous)
    elif model_dir.name == _WHISPER_LARGE_V3_TARGET:
        url = _WHISPER_LARGE_V3_URL
        archive_name = _WHISPER_LARGE_V3_ARCHIVE
        extracted_name = _WHISPER_LARGE_V3_EXTRACTED
    elif model_dir.name == _WHISPER_TURBO_TARGET:
        url = _WHISPER_TURBO_URL
        archive_name = _WHISPER_TURBO_ARCHIVE
        extracted_name = _WHISPER_TURBO_EXTRACTED
    elif model_dir.name == _WHISPER_DISTIL_LARGE_V35_TARGET:
        url = _WHISPER_DISTIL_LARGE_V35_URL
        archive_name = _WHISPER_DISTIL_LARGE_V35_ARCHIVE
        extracted_name = _WHISPER_DISTIL_LARGE_V35_EXTRACTED
    # SenseVoice multilingual model
    elif model_type == "sense_voice" or model_dir.name == _SENSE_VOICE_TARGET:
        url = _SENSE_VOICE_URL
        archive_name = _SENSE_VOICE_ARCHIVE
        extracted_name = _SENSE_VOICE_EXTRACTED
    # Multilingual streaming zipformer model
    elif model_type == "multilingual_streaming" or model_dir.name == _MULTILINGUAL_STREAMING_TARGET:
        url = _MULTILINGUAL_STREAMING_URL
        archive_name = _MULTILINGUAL_STREAMING_ARCHIVE
        extracted_name = _MULTILINGUAL_STREAMING_EXTRACTED
    # German streaming zipformer (online default for --lang de)
    elif model_dir.name == _GERMAN_STREAMING_TARGET:
        url = _GERMAN_STREAMING_URL
        archive_name = _GERMAN_STREAMING_ARCHIVE
        extracted_name = _GERMAN_STREAMING_EXTRACTED
    # German NeMo CTC (offline default for --lang de --offline)
    elif model_dir.name == _GERMAN_NEMO_TARGET:
        url = _GERMAN_NEMO_URL
        archive_name = _GERMAN_NEMO_ARCHIVE
        extracted_name = _GERMAN_NEMO_EXTRACTED
    # NeMo CTC English models (medium is the auto-download default; small is opt-in)
    elif model_dir.name == _NEMO_CTC_EN_MEDIUM_TARGET:
        url = _NEMO_CTC_EN_MEDIUM_URL
        archive_name = _NEMO_CTC_EN_MEDIUM_ARCHIVE
        extracted_name = _NEMO_CTC_EN_MEDIUM_EXTRACTED
    elif model_dir.name == _NEMO_CTC_EN_SMALL_TARGET:
        url = _NEMO_CTC_EN_SMALL_URL
        archive_name = _NEMO_CTC_EN_SMALL_ARCHIVE
        extracted_name = _NEMO_CTC_EN_SMALL_EXTRACTED
    # Use parakeet as the default offline model download target
    elif model_type == "nemo_transducer" or model_dir.name in (
        _PARAKEET_FP16_TARGET, _PARAKEET_INT8_TARGET
    ):
        # Choose variant based on directory name
        if "int8" in model_dir.name:
            url = _PARAKEET_INT8_URL
            archive_name = _PARAKEET_INT8_ARCHIVE
            extracted_name = _PARAKEET_INT8_EXTRACTED
        else:
            url = _PARAKEET_FP16_URL
            archive_name = _PARAKEET_FP16_ARCHIVE
            extracted_name = _PARAKEET_FP16_EXTRACTED
    else:
        url = _MODEL_URL
        archive_name = _MODEL_ARCHIVE
        extracted_name = _MODEL_EXTRACTED

    # Download into models/ alongside the target directory
    models_dir = model_dir.parent
    models_dir.mkdir(parents=True, exist_ok=True)
    archive = models_dir / archive_name
    _info("Model not found.")
    # Only download if the archive isn't already fully present (valid tarfile).
    if archive.exists():
        try:
            with tarfile.open(archive, "r:bz2") as _tf:
                _tf.getmembers()
            _info(f"Archive already present: {archive.name}")
        except Exception:  # noqa: BLE001
            _info("Existing archive is incomplete — resuming download…")
            _download_file(url, archive)
    else:
        _download_file(url, archive)

    _info("Extracting…")
    try:
        with tarfile.open(archive, "r:bz2") as tf:
            if sys.version_info >= (3, 12):
                tf.extractall(models_dir, filter="data")
            else:  # pragma: no cover
                tf.extractall(models_dir, members=_safe_tar_members(tf, models_dir))
    except Exception as exc:  # noqa: BLE001
        _error(f"Extraction failed: {exc}")

    extracted = models_dir / extracted_name
    if not extracted.is_dir():
        _error(f"Expected extracted directory '{extracted_name}' not found.")

    extracted.rename(model_dir)
    archive.unlink(missing_ok=True)
    _info(f"Model saved to '{model_dir}'.\n")


def _validate_model(model_dir: str, model_type: str) -> None:
    if not Path(model_dir).is_dir():
        _download_model(model_dir, model_type)


def _normalize_language(language: str) -> str:
    normalized = language.lower().replace("_", "-")
    return _LANGUAGE_ALIASES.get(normalized, normalized)


def _resolve_default_model(language: str, model_type: str, offline: bool) -> tuple[str, str]:
    """Return (model_type, model_dir_name) for omitted --model-dir."""
    if model_type == "parakeet-ctc-ja":
        return model_type, _PARAKEET_CTC_JA_INT8_TARGET
    if model_type in ("ja", "reazonspeech-ja"):
        return "reazonspeech-ja", _REAZON_JA_TARGET
    if model_type in ("ja-en", "reazonspeech-ja-en"):
        return "reazonspeech-ja-en", _REAZON_JA_EN_TARGET
    if model_type in ("ja-en-mls-5k", "reazonspeech-ja-en-mls-5k"):
        return "reazonspeech-ja-en-mls-5k", _REAZON_JA_EN_MLS_TARGET
    if model_type == "cohere_transcribe":
        return model_type, _COHERE_TRANSCRIBE_TARGET
    if model_type == "multilingual_streaming":
        return model_type, _MULTILINGUAL_STREAMING_TARGET
    if model_type == "nemo_transducer":
        return model_type, _PARAKEET_TARGET
    if model_type == "nemo_ctc":
        if language == "de":
            return model_type, _GERMAN_NEMO_TARGET
        return model_type, _NEMO_CTC_EN_MEDIUM_TARGET
    if model_type:
        return model_type, _PARAKEET_TARGET if offline else _MODEL_TARGET
    # German: online uses streaming zipformer, offline uses NeMo CTC
    if language == "de":
        if offline:
            return "nemo_ctc", _GERMAN_NEMO_TARGET
        return "", _GERMAN_STREAMING_TARGET
    if language in _ASR_LANGUAGE_DEFAULTS:
        return _ASR_LANGUAGE_DEFAULTS[language]
    if offline:
        return model_type, _PARAKEET_TARGET
    return model_type, _MODEL_TARGET


def _validate_vad(vad_type: str, ten_vad_model: str, offline: bool, project_dir: Path) -> str:
    if not offline:
        return ""
    if vad_type not in {"silero", "ten-vad"}:
        _error(f"Unknown --vad-model type '{vad_type}'. Supported: silero, ten-vad.")
    if vad_type == "ten-vad":
        if ten_vad_model not in _TEN_VAD_MODEL_URLS:
            _error(
                f"Unknown --ten-vad-model '{ten_vad_model}'. "
                f"Supported: {', '.join(_TEN_VAD_MODEL_URLS)}."
            )
        vad_path = project_dir / "models" / ten_vad_model
        if not vad_path.exists():
            vad_path.parent.mkdir(parents=True, exist_ok=True)
            _info(f"VAD model not found, downloading {ten_vad_model}…")
            _download_file(_TEN_VAD_MODEL_URLS[ten_vad_model], vad_path)
        return str(vad_path)
    # silero
    vad_path = project_dir / "models" / "silero_vad.onnx"
    if not vad_path.exists():
        vad_path.parent.mkdir(parents=True, exist_ok=True)
        _info("VAD model not found, downloading silero_vad.onnx…")
        _download_file(_VAD_URL, vad_path)
    return str(vad_path)


def _safe_extract_tar(tar: tarfile.TarFile, path: Path) -> None:
    """Safely extract tar contents to `path`, preventing path traversal and links.

    This emulates the behavior of `filter="data"` available in Python 3.12+,
    but is compatible with Python 3.8+.
    """
    base_path = path.resolve()
    for member in tar.getmembers():
        # Skip symlinks and hard links for safety.
        if member.issym() or member.islnk():
            continue

        member_path = (base_path / member.name).resolve()
        try:
            # Ensure the target path is within the intended base directory.
            member_path.relative_to(base_path)
        except ValueError:
            # Path traversal attempt or otherwise outside base directory; skip.
            continue

        tar.extract(member, path=base_path)


def _validate_diarization_models(
    seg_model: str, emb_model: str, project_dir: Path
) -> tuple[str, str]:
    """Return paths to diarization models, downloading them if necessary."""
    models_dir = project_dir / "models"

    # Segmentation model
    if not seg_model:
        seg_dir = models_dir / _DIAR_SEG_EXTRACTED
        seg_path = seg_dir / _DIAR_SEG_MODEL_FILE
        if not seg_path.exists():
            seg_dir.parent.mkdir(parents=True, exist_ok=True)
            _info("Diarization segmentation model not found, downloading…")
            archive = models_dir / _DIAR_SEG_ARCHIVE
            _download_file(_DIAR_SEG_URL, archive)
            _info("Extracting segmentation model…")
            try:
                with tarfile.open(archive, "r:bz2") as tf:
                    _safe_extract_tar(tf, models_dir)
            except Exception as exc:  # noqa: BLE001
                _error(f"Extraction failed: {exc}")
            archive.unlink(missing_ok=True)
            if not seg_path.exists():
                _error(f"Segmentation model not found after extraction: {seg_path}")
            _info(f"Segmentation model saved to '{seg_path}'.")
        seg_model = str(seg_path)
    elif not Path(seg_model).exists():
        _error(f"Diarization segmentation model not found: {seg_model}")

    # Embedding model
    if not emb_model:
        emb_path = models_dir / _DIAR_EMB_FILE
        if not emb_path.exists():
            models_dir.mkdir(parents=True, exist_ok=True)
            _info("Diarization embedding model not found, downloading…")
            _download_file(_DIAR_EMB_URL, emb_path)
            _info(f"Embedding model saved to '{emb_path}'.")
        emb_model = str(emb_path)
    elif not Path(emb_model).exists():
        _error(f"Diarization embedding model not found: {emb_model}")

    return seg_model, emb_model


def _validate_punct(punct_model: str, project_dir: Path) -> str:
    """Return path to punctuation model directory, downloading if necessary."""
    if punct_model:
        if not Path(punct_model).is_dir():
            _error(f"Punctuation model directory not found: {punct_model}")
        return punct_model
    models_dir = project_dir / "models"
    punct_dir = models_dir / _PUNCT_TARGET
    if punct_dir.is_dir():
        return str(punct_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    archive = models_dir / _PUNCT_ARCHIVE
    _info("Punctuation model not found, downloading…")
    if archive.exists():
        try:
            with tarfile.open(archive, "r:bz2") as _tf:
                _tf.getmembers()
        except Exception:  # noqa: BLE001
            _download_file(_PUNCT_URL, archive)
    else:
        _download_file(_PUNCT_URL, archive)
    _info("Extracting punctuation model…")
    try:
        with tarfile.open(archive, "r:bz2") as tf:
            if sys.version_info >= (3, 12):
                tf.extractall(models_dir, filter="data")
            else:  # pragma: no cover
                tf.extractall(models_dir, members=_safe_tar_members(tf, models_dir))
    except Exception as exc:  # noqa: BLE001
        _error(f"Extraction failed: {exc}")
    extracted = models_dir / _PUNCT_EXTRACTED
    if not extracted.is_dir():
        _error(f"Expected punctuation model directory '{_PUNCT_EXTRACTED}' not found after extraction.")
    extracted.rename(punct_dir)
    archive.unlink(missing_ok=True)
    _info(f"Punctuation model saved to '{punct_dir}'.")
    return str(punct_dir)


def _validate_wav(path: str, sample_rate: int) -> None:
    p = Path(path)
    if not p.exists():
        _error(f"Audio file not found: {path}")
    try:
        sf = _require_soundfile()
        with sf.SoundFile(path) as f:
            if f.channels != 1:
                _error(
                    f"Audio must be mono (1 channel), got {f.channels}.\n"
                    f"  Convert: ffmpeg -i {path} -ar {sample_rate} -ac 1 out.wav"
                )
            if f.samplerate != sample_rate:
                _info(
                    f"Audio is {f.samplerate} Hz; resampling to {sample_rate} Hz."
                )
    except Exception as exc:
        _error(f"Cannot read audio file: {exc}")


def _validate_mic() -> None:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        inputs = [d for d in devices if d["max_input_channels"] > 0]
        if not inputs:
            _error("No input audio device found.")
    except Exception as exc:  # noqa: BLE001
        _error(f"Microphone check failed: {exc}")


def main() -> None:
    _run_cli(_main_impl)


def _main_impl() -> None:
    args = parse_args()
    _validate_runtime_args(args)
    # Normalize once so all downstream comparisons are case-insensitive.
    args.model_type = args.model_type.lower()
    # Canonicalize short Japanese aliases regardless of whether --model-dir was supplied.
    _JA_ALIASES = {"ja": "reazonspeech-ja", "ja-en": "reazonspeech-ja-en", "ja-en-mls-5k": "reazonspeech-ja-en-mls-5k"}
    args.model_type = _JA_ALIASES.get(args.model_type, args.model_type)
    args.language = _normalize_language(args.language)

    # Resolve paths relative to the project root (one level above src/).
    project_dir = Path(__file__).resolve().parent.parent
    # Use a type-specific default dir when the user didn't pass --model-dir explicitly.
    if args.model_dir is None:
        args.model_type, model_dir_name = _resolve_default_model(
            args.language, args.model_type, args.offline
        )
        raw_model_dir = f"models/{model_dir_name}"
    else:
        raw_model_dir = args.model_dir
    model_dir = Path(raw_model_dir)
    if not model_dir.is_absolute():
        model_dir = project_dir / model_dir

    cfg = Config(
        model_dir=str(model_dir),
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        num_threads=args.threads,
        model_type=args.model_type,
        offline=args.offline,
        vad_type=args.vad_type,
        ten_vad_model=args.ten_vad_model,
        language=args.language,
        show_mic_level=not args.no_mic_level,
        diarization=args.diarization,
        diarization_seg_model=args.diarization_seg_model,
        diarization_emb_model=args.diarization_emb_model,
        diarization_num_speakers=args.num_speakers,
        device=args.device,
        denoise=args.denoise,
        word_timestamps=args.word_timestamps,
        punctuation=args.punctuation,
        punct_model=args.punct_model,
        translate=args.translate,
        no_color=args.no_color,
        json_output=args.json_output,
    )

    global _json_mode
    _json_mode = cfg.json_output

    _validate_model(cfg.model_dir, cfg.model_type)

    # Auto-detect offline-only models and switch automatically.
    _OFFLINE_ONLY_TYPES = {"nemo_transducer", "whisper", "nemo_ctc", "sense_voice", "moonshine", "fire_red_asr", "cohere_transcribe", "reazonspeech-ja", "reazonspeech-ja-en", "reazonspeech-ja-en-mls-5k", "parakeet-ctc-ja"}
    _OFFLINE_ONLY_NAME_PATTERNS = ("parakeet", "nemo", "whisper", "sense_voice", "moonshine", "fire_red_asr", "cohere", "reazonspeech")
    model_name_lower = Path(cfg.model_dir).name.lower()
    if not cfg.offline and (
        cfg.model_type in _OFFLINE_ONLY_TYPES
        or any(pat in model_name_lower for pat in _OFFLINE_ONLY_NAME_PATTERNS)
    ):
        _info(
            f"Model '{Path(cfg.model_dir).name}' is offline-only — "
            "enabling --offline automatically."
        )
        cfg.offline = True

    # Remap model-type aliases that sherpa-onnx doesn't accept in from_transducer.
    # Use "" so sherpa-onnx auto-detects the architecture from the model files.
    _TRANSDUCER_AUTODETECT_ALIASES = {"reazonspeech-ja", "reazonspeech-ja-en", "reazonspeech-ja-en-mls-5k", "multilingual_streaming"}
    if cfg.model_type in _TRANSDUCER_AUTODETECT_ALIASES:
        cfg.model_type = ""

    # Remap convenience CTC aliases to the sherpa-onnx nemo_ctc model type.
    _CTC_TYPE_ALIASES = {"parakeet-ctc-ja"}
    if cfg.model_type in _CTC_TYPE_ALIASES:
        cfg.model_type = "nemo_ctc"

    cfg.vad_model = _validate_vad(cfg.vad_type, cfg.ten_vad_model, cfg.offline, project_dir)

    if args.wav:
        for wav_path in args.wav:
            _validate_wav(wav_path, cfg.sample_rate)
    elif not args.pipe:
        _validate_mic()

    # Download punctuation model if requested.
    punctuator = None
    if cfg.punctuation:
        cfg.punct_model = _validate_punct(cfg.punct_model, project_dir)

    # Validate / download diarization models if requested.
    diarizer = None
    if cfg.diarization:
        cfg.diarization_seg_model, cfg.diarization_emb_model = _validate_diarization_models(
            cfg.diarization_seg_model, cfg.diarization_emb_model, project_dir
        )

    model_name = Path(cfg.model_dir).name
    _info(f"Loading model '{model_name}' ({cfg.num_threads} threads)…")

    # Build models once before any batch processing.
    if cfg.offline:
        recognizer = build_offline_recognizer(cfg)
        vad_sample_rate = cfg.sample_rate  # WAV path uses cfg.sample_rate; mic path overrides below
        cfg.sample_rate = vad_sample_rate
        vad = build_vad(cfg)
    else:
        recognizer = build_recognizer(cfg)
        vad = None

    if cfg.diarization:
        _info("Loading diarization models…")
        diarizer = build_diarization(cfg)

    if cfg.punctuation:
        punctuator = build_punctuation(cfg)

    _info("Model ready.\n")

    if args.wav:
        # Batch WAV mode: iterate over all provided paths.
        output_dir = Path(args.output_dir) if args.output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn  # noqa: PLC0415

        for wav_path in args.wav:
            _info(f"Transcribing: {wav_path}\n")
            audio = read_wav(wav_path, target_sr=cfg.sample_rate, chunk_size=cfg.chunk_size)
            if cfg.denoise:
                audio = denoise_gen(audio, cfg.sample_rate)

            # Subtitle collection for this file.
            subtitles: list[tuple[float, float, str]] = []

            # Progress callback using total duration.
            try:
                total_s = wav_duration(wav_path)
            except Exception:
                total_s = 0.0

            use_progress = not args.final_only or cfg.offline
            progress_ctx = (
                Progress(
                    "[progress.description]{task.description}",
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=Console(stderr=True),
                    transient=True,
                )
                if use_progress
                else nullcontext()
            )
            with progress_ctx as prog:
                task_id = prog.add_task("Transcribing…", total=total_s or None) if use_progress else None

                def _progress_cb(elapsed: float, _tid=task_id, _prog=prog) -> None:
                    if use_progress and _tid is not None:
                        _prog.update(_tid, completed=elapsed)

                if cfg.offline:
                    run_offline_vad_streaming(
                        recognizer=recognizer,
                        vad=vad,
                        audio_gen=audio,
                        sample_rate=cfg.sample_rate,
                        show_mic_level=False,
                        diarization=diarizer,
                        show_speaker_tag=args.speaker_tag,
                        word_timestamps=cfg.word_timestamps,
                        punctuation=punctuator,
                        subtitles=subtitles,
                        progress_callback=_progress_cb,
                        json_output=cfg.json_output,
                        no_color=cfg.no_color,
                    )
                else:
                    run_streaming(
                        recognizer,
                        audio,
                        sample_rate=cfg.sample_rate,
                        show_mic_level=False,
                        diarization=diarizer,
                        show_speaker_tag=args.speaker_tag,
                        word_timestamps=cfg.word_timestamps,
                        punctuation=punctuator,
                        subtitles=subtitles,
                        final_only=args.final_only,
                        json_output=cfg.json_output,
                        no_color=cfg.no_color,
                    )

            # Write output file if requested.
            if subtitles:
                if args.output:
                    _write_subtitles(subtitles, args.output)
                    _info(f"Output written to: {args.output}")
                elif output_dir:
                    stem = Path(wav_path).stem
                    fmt = args.output_format
                    out_path = str(output_dir / f"{stem}.{fmt}")
                    _write_subtitles(subtitles, out_path)
                    _info(f"Output written to: {out_path}")
    elif args.pipe:
        # Stdin/pipe mode — reads raw 16-bit LE mono PCM from stdin.
        _info("Reading PCM from stdin — send EOF (Ctrl+D) to stop.\n")
        subtitles: list[tuple[float, float, str]] | None = [] if args.output else None
        if cfg.offline:
            cfg.sample_rate = args.capture_rate
            vad = build_vad(cfg)
            run_offline_vad_streaming(
                recognizer=recognizer,
                vad=vad,
                audio_gen=pipe_stream(capture_rate=args.capture_rate, chunk_size=cfg.chunk_size),
                sample_rate=args.capture_rate,
                show_mic_level=False,
                diarization=diarizer,
                show_speaker_tag=args.speaker_tag,
                word_timestamps=cfg.word_timestamps,
                punctuation=punctuator,
                json_output=cfg.json_output,
                no_color=cfg.no_color,
            )
        else:
            run_streaming(
                recognizer,
                pipe_stream(capture_rate=args.capture_rate, chunk_size=cfg.chunk_size),
                sample_rate=args.capture_rate,
                show_mic_level=False,
                diarization=diarizer,
                show_speaker_tag=args.speaker_tag,
                word_timestamps=cfg.word_timestamps,
                punctuation=punctuator,
                final_only=args.final_only,
                subtitles=subtitles,
                json_output=cfg.json_output,
                no_color=cfg.no_color,
            )
        if subtitles and args.output:
            _write_subtitles(subtitles, args.output)
            _info(f"Output written to: {args.output}")
    else:
        # Microphone mode — rebuild VAD with capture_rate so it matches input.
        if cfg.offline:
            cfg.sample_rate = args.capture_rate
            vad = build_vad(cfg)
            _info("Listening on microphone — press Ctrl+C to stop.\n")
            run_offline_vad_streaming(
                recognizer=recognizer,
                vad=vad,
                audio_gen=mic_stream(capture_rate=args.capture_rate, chunk_size=cfg.chunk_size),
                sample_rate=args.capture_rate,
                show_mic_level=cfg.show_mic_level,
                diarization=diarizer,
                show_speaker_tag=args.speaker_tag,
                word_timestamps=cfg.word_timestamps,
                punctuation=punctuator,
                json_output=cfg.json_output,
                no_color=cfg.no_color,
            )
        else:
            _info("Listening on microphone — press Ctrl+C to stop.\n")
            run_streaming(
                recognizer,
                mic_stream(capture_rate=args.capture_rate, chunk_size=cfg.chunk_size),
                sample_rate=args.capture_rate,
                show_mic_level=cfg.show_mic_level,
                diarization=diarizer,
                show_speaker_tag=args.speaker_tag,
                word_timestamps=cfg.word_timestamps,
                punctuation=punctuator,
                final_only=args.final_only,
                json_output=cfg.json_output,
                no_color=cfg.no_color,
            )


def _write_subtitles(subtitles: list[tuple[float, float, str]], path: str) -> None:
    """Write subtitles to *path*, inferring format from the file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".srt":
        write_srt(subtitles, path)
    elif ext == ".vtt":
        write_vtt(subtitles, path)
    else:
        write_txt(subtitles, path)


if __name__ == "__main__":  # pragma: no cover
    main()
