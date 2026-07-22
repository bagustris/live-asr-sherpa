"""Text-to-speech synthesis — entry point.

Usage:
    # Synthesise from inline text (Indonesian, default):
    sherox.tts --text "Selamat pagi, apa kabar?"

    # Synthesise English:
    sherox.tts --text "Hello, how are you today?" --lang eng

    # Synthesise Chinese (Mandarin, 8 kHz VITS, plain Chinese text):
    sherox.tts --text "你好，今天天气不错。" --lang zho

    # Synthesise German:
    sherox.tts --text "Guten Morgen, wie geht es Ihnen?" --lang deu

    # Synthesise French:
    sherox.tts --text "Bonjour, comment allez-vous?" --lang fra

    # Synthesise Spanish:
    sherox.tts --text "Hola, ¿cómo estás hoy?" --lang spa

    # Synthesise Japanese with Piper Plus:
    sherox.tts --text "こんにちは、今日は良い天気ですね。" --lang jpn

    # Synthesise Japanese with Sarashina2.2-TTS (zero-shot voice cloning):
    sherox.tts --text "こんにちは。" --lang jpn-sarashina \\
        --audio-prompt prompt.wav --audio-prompt-text "プロンプトの文章。"

    # Sarashina without voice cloning (default voice):
    sherox.tts --text "こんにちは。" --lang jpn-sarashina

    # Synthesise Japanese with the torch-free Sarashina ONNX runtime
    #   (auto-downloads from huggingface.co/Bagus/Sarashina2.2-TTS-ONNX on first use):
    sherox.tts --text "こんにちは。" --lang jpn-sarashina-onnx

    # Read from file:
    sherox.tts --file input.txt --lang ind

    # Read from stdin:
    echo "Halo dunia" | sherox.tts --lang ind

    # Save to a specific output file:
    sherox.tts --text "Halo" --output halo.wav

    # Play through the system speaker (requires sounddevice):
    sherox.tts --text "Halo" --play

    # Play without saving a WAV file:
    sherox.tts --text "Halo" --play --no-save

    # Control speech speed:
    sherox.tts --text "Halo" --speed 0.85

    # Synthesise English with Kitten TTS (quantized, 24 kHz):
    sherox.tts --text "Hello, how are you today?" --lang eng-kitten

    # Synthesise Korean (Supertonic-3, 24 kHz):
    sherox.tts --text "안녕하세요." --lang kor

    # Synthesise Russian (Supertonic-3, 24 kHz):
    sherox.tts --text "Привет, как дела?" --lang rus

    # Synthesise Hindi (Supertonic-3, 24 kHz):
    sherox.tts --text "नमस्ते, आप कैसे हैं?" --lang hin

    # Synthesise Vietnamese (Supertonic-3, 24 kHz):
    sherox.tts --text "Xin chào, bạn khỏe không?" --lang vie

Supported languages (ISO 639-3 code → model):
    eng           English US   — vits-piper-en_US-amy-medium       (22050 Hz, 1 speaker)
    eng-kitten    English      — Kitten TTS Nano v0.8, quantized    (24000 Hz, 8 speakers)
    deu           German       — vits-piper-de_DE-thorsten-medium   (22050 Hz, 1 speaker)
    fra           French       — vits-piper-fr_FR-upmc-medium       (22050 Hz, 1 speaker)
    spa           Spanish      — vits-piper-es_ES-mls_10246-medium  (22050 Hz, 1 speaker)
    ind           Indonesian   — vits-piper-id_ID-news_tts-medium   (22050 Hz, 1 speaker)
    ind-supertonic Indonesian   — Supertonic-3 (24000 Hz, 10 speakers)
    zho           Chinese      — vits-icefall-zh-aishell3           (8 kHz, 174 speakers)
    jpn           Japanese     — piper-plus tsukuyomi               (22050 Hz, 1 speaker)
    jpn-sarashina Japanese     — Sarashina2.2-TTS, zero-shot        (24000 Hz, voice cloning)
    jpn-sarashina-onnx Japanese — Sarashina2.2-TTS, ONNX runtime     (24000 Hz, torch-free)
    jpn-supertonic Japanese     — Supertonic-3 (24000 Hz, 10 speakers)
    kor           Korean       — Supertonic-3 (24000 Hz, 10 speakers)
    ara           Arabic       — Supertonic-3 (24000 Hz, 10 speakers)
    bul           Bulgarian    — Supertonic-3 (24000 Hz, 10 speakers)
    ces           Czech        — Supertonic-3 (24000 Hz, 10 speakers)
    dan           Danish       — Supertonic-3 (24000 Hz, 10 speakers)
    ell           Greek        — Supertonic-3 (24000 Hz, 10 speakers)
    est           Estonian     — Supertonic-3 (24000 Hz, 10 speakers)
    fin           Finnish      — Supertonic-3 (24000 Hz, 10 speakers)
    hin           Hindi        — Supertonic-3 (24000 Hz, 10 speakers)
    hrv           Croatian     — Supertonic-3 (24000 Hz, 10 speakers)
    hun           Hungarian    — Supertonic-3 (24000 Hz, 10 speakers)
    ita           Italian      — Supertonic-3 (24000 Hz, 10 speakers)
    lit           Lithuanian   — Supertonic-3 (24000 Hz, 10 speakers)
    lav           Latvian      — Supertonic-3 (24000 Hz, 10 speakers)
    nld           Dutch        — Supertonic-3 (24000 Hz, 10 speakers)
    pol           Polish       — Supertonic-3 (24000 Hz, 10 speakers)
    por           Portuguese   — Supertonic-3 (24000 Hz, 10 speakers)
    ron           Romanian     — Supertonic-3 (24000 Hz, 10 speakers)
    rus           Russian      — Supertonic-3 (24000 Hz, 10 speakers)
    slk           Slovak       — Supertonic-3 (24000 Hz, 10 speakers)
    slv           Slovenian    — Supertonic-3 (24000 Hz, 10 speakers)
    swe           Swedish      — Supertonic-3 (24000 Hz, 10 speakers)
    tur           Turkish      — Supertonic-3 (24000 Hz, 10 speakers)
    ukr           Ukrainian    — Supertonic-3 (24000 Hz, 10 speakers)
    vie           Vietnamese   — Supertonic-3 (24000 Hz, 10 speakers)

Language aliases (short forms also accepted):
    en / eng-us                 → eng
    eng-kitten                 → eng-kitten
    de / ger                    → deu
    fr                          → fra
    es                          → spa
    id / id-id                  → ind
    zh / zh-cn / zh-tw / cmn    → zho
    ja / jp / ja-jp             → jpn
    sarashina / jpn_sarashina   → jpn-sarashina
    ko                          → kor
    ar                          → ara
    bg                          → bul
    cs                          → ces
    da                          → dan
    el                          → ell
    et                          → est
    fi                          → fin
    hi                          → hin
    hr                          → hrv
    hu                          → hun
    it                          → ita
    lt                          → lit
    lv                          → lav
    nl / dut                    → nld
    pl                          → pol
    pt                          → por
    ro / rum                    → ron
    ru                          → rus
    sk                          → slk
    sl                          → slv
    sv                          → swe
    tr                          → tur
    uk                          → ukr
    vi                          → vie

Notes:
    Chinese (zho): input must be plain Simplified Chinese text; numbers and mixed
    scripts may not normalise well. Use 8 kHz output; quality is acceptable for
    voice assistants and dev/test use.

    Supertonic-3: shared model for 25+ languages (24 kHz, 10 speakers).  First use
    downloads ~120 MB.  Select speaker with --speaker-id 0-9, language via --lang.

Models are auto-downloaded on first use into  models/<model-dir>/  at the project root.
"""

import argparse
import io
import sys
import tarfile
import wave
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
from rich.console import Console

from . import AudioError, ConfigError
from .config import TtsConfig
from .utils import download_file as _download_file
from .utils import run_cli as _run_cli
from .utils import safe_tar_members as _safe_tar_members

sf = SimpleNamespace(write=None)
piper_runtime = None
sarashina_runtime = None

_console = Console()
_err_console = Console(stderr=True)

# ── Model registry (ISO 639-3 → model metadata) ──────────────────────────────

_TTS_MODELS: dict[str, dict] = {
    "eng": {
        "backend": "sherpa_onnx",
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/vits-piper-en_US-amy-medium.tar.bz2"
        ),
        "archive": "vits-piper-en_US-amy-medium.tar.bz2",
        "extracted": "vits-piper-en_US-amy-medium",
        "model": "en_US-amy-medium.onnx",
        "tokens": "tokens.txt",
        "data_dir": "espeak-ng-data",
        "sample_rate": 22050,
        "description": "English US (Piper VITS, Amy, medium quality)",
    },
    "deu": {
        "backend": "sherpa_onnx",
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/vits-piper-de_DE-thorsten-medium.tar.bz2"
        ),
        "archive": "vits-piper-de_DE-thorsten-medium.tar.bz2",
        "extracted": "vits-piper-de_DE-thorsten-medium",
        "model": "de_DE-thorsten-medium.onnx",
        "tokens": "tokens.txt",
        "data_dir": "espeak-ng-data",
        "sample_rate": 22050,
        "description": "German (Piper VITS, Thorsten, medium quality)",
    },
    "fra": {
        "backend": "sherpa_onnx",
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/vits-piper-fr_FR-upmc-medium.tar.bz2"
        ),
        "archive": "vits-piper-fr_FR-upmc-medium.tar.bz2",
        "extracted": "vits-piper-fr_FR-upmc-medium",
        "model": "fr_FR-upmc-medium.onnx",
        "tokens": "tokens.txt",
        "data_dir": "espeak-ng-data",
        "sample_rate": 22050,
        "description": "French (Piper VITS, UPMC, medium quality)",
    },
    "spa": {
        "backend": "sherpa_onnx",
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/vits-piper-es_ES-mls_10246-medium.tar.bz2"
        ),
        "archive": "vits-piper-es_ES-mls_10246-medium.tar.bz2",
        "extracted": "vits-piper-es_ES-mls_10246-medium",
        "model": "es_ES-mls_10246-medium.onnx",
        "tokens": "tokens.txt",
        "data_dir": "espeak-ng-data",
        "sample_rate": 22050,
        "description": "Spanish (Piper VITS, MLS, medium quality)",
    },
    "ind": {
        "backend": "sherpa_onnx",
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/vits-piper-id_ID-news_tts-medium.tar.bz2"
        ),
        "archive": "vits-piper-id_ID-news_tts-medium.tar.bz2",
        "extracted": "vits-piper-id_ID-news_tts-medium",
        "model": "id_ID-news_tts-medium.onnx",
        "tokens": "tokens.txt",
        "data_dir": "espeak-ng-data",
        "sample_rate": 22050,
        "description": "Indonesian (Piper VITS, medium quality)",
    },
    "zho": {
        # Chinese Mandarin — VITS with AiShell3 corpus (174 speakers, 8 kHz).
        #
        # Input must be plain Simplified Chinese text. Numbers and mixed-script
        # text may not normalise correctly; the model does not include a
        # text-normalisation frontend.
        #
        # Unlike the Piper-VITS models, this model uses a lexicon file instead
        # of espeak-ng data for G2P conversion.
        #
        # Model source:
        #   https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
        # Direct download:
        #   vits-icefall-zh-aishell3.tar.bz2
        "backend": "sherpa_onnx",
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/vits-icefall-zh-aishell3.tar.bz2"
        ),
        "archive": "vits-icefall-zh-aishell3.tar.bz2",
        "extracted": "vits-icefall-zh-aishell3",
        "model": "model.onnx",
        "tokens": "tokens.txt",
        # lexicon.txt provides the pronunciation dictionary for Chinese characters.
        # data_dir is empty — this model does not use espeak-ng.
        "lexicon": "lexicon.txt",
        "data_dir": "",
        "sample_rate": 8000,
        "description": "Chinese Mandarin (VITS, AiShell3, 174 speakers, 8 kHz)",
    },
    "jpn": {
        "backend": "piper_plus",
        "voice_name": "ja_JP-tsukuyomi-chan-medium",
        "language_id": 0,
        "sample_rate": 22050,
        "description": "Japanese (Piper Plus Tsukuyomi)",
    },
    "jpn-sarashina": {
        "backend": "sarashina",
        "model_id": "sbintuitions/Sarashina-TTS",
        "sample_rate": 24000,
        "description": "Japanese (Sarashina2.2-TTS, zero-shot voice cloning)",
    },
    "jpn-sarashina-onnx": {
        "backend": "sarashina_onnx",
        "sample_rate": 24000,
        "description": "Japanese (Sarashina2.2-TTS, ONNX runtime, torch-free)",
    },
    "eng-kitten": {
        "backend": "kitten",
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "tts-models/kitten-nano-en-v0_8-int8.tar.bz2"
        ),
        "archive": "kitten-nano-en-v0_8-int8.tar.bz2",
        "extracted": "kitten-nano-en-v0_8-int8",
        "model": "model.int8.onnx",
        "voices": "voices.bin",
        "tokens": "tokens.txt",
        "data_dir": "espeak-ng-data",
        "sample_rate": 24000,
        "description": "English (Kitten TTS Nano v0.8, quantized)",
    },
}

# ── Supertonic-3 shared model metadata ───────────────────────────────────────
_SUPERTONIC_BASE: dict = {
    "backend": "supertonic",
    "url": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2"
    ),
    "archive": "sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2",
    "extracted": "sherpa-onnx-supertonic-3-tts-int8-2026-05-11",
    "files": {
        "duration_predictor": "duration_predictor.int8.onnx",
        "text_encoder": "text_encoder.int8.onnx",
        "vector_estimator": "vector_estimator.int8.onnx",
        "vocoder": "vocoder.int8.onnx",
        "tts_json": "tts.json",
        "unicode_indexer": "unicode_indexer.bin",
        "voice_style": "voice.bin",
    },
    "sample_rate": 24000,
}

# Supertonic-3 supported languages: en, ko, ja, ar, bg, cs, da, de, el, es, et,
# fi, fr, hi, hr, hu, id, it, lt, lv, nl, pl, pt, ro, ru, sk, sl, sv, tr, uk, vi
#
# Languages that already have a dedicated model in _TTS_MODELS (eng, deu, fra,
# spa, ind, zho, jpn) keep their existing default.  All others use supertonic-3.
# ind-supertonic provides an alternative Supertonic-3 model for Indonesian.
# jpn-supertonic provides an alternative Supertonic-3 model for Japanese.
_SUPERTONIC_LANGUAGES: dict[str, dict] = {
    "kor": {
        "lang_code": "ko",
        "description": "Korean (Supertonic-3, 10 speakers)",
    },
    "ara": {
        "lang_code": "ar",
        "description": "Arabic (Supertonic-3, 10 speakers)",
    },
    "bul": {
        "lang_code": "bg",
        "description": "Bulgarian (Supertonic-3, 10 speakers)",
    },
    "ces": {
        "lang_code": "cs",
        "description": "Czech (Supertonic-3, 10 speakers)",
    },
    "dan": {
        "lang_code": "da",
        "description": "Danish (Supertonic-3, 10 speakers)",
    },
    "ell": {
        "lang_code": "el",
        "description": "Greek (Supertonic-3, 10 speakers)",
    },
    "est": {
        "lang_code": "et",
        "description": "Estonian (Supertonic-3, 10 speakers)",
    },
    "fin": {
        "lang_code": "fi",
        "description": "Finnish (Supertonic-3, 10 speakers)",
    },
    "hin": {
        "lang_code": "hi",
        "description": "Hindi (Supertonic-3, 10 speakers)",
    },
    "hrv": {
        "lang_code": "hr",
        "description": "Croatian (Supertonic-3, 10 speakers)",
    },
    "hun": {
        "lang_code": "hu",
        "description": "Hungarian (Supertonic-3, 10 speakers)",
    },
    "ita": {
        "lang_code": "it",
        "description": "Italian (Supertonic-3, 10 speakers)",
    },
    "lit": {
        "lang_code": "lt",
        "description": "Lithuanian (Supertonic-3, 10 speakers)",
    },
    "lav": {
        "lang_code": "lv",
        "description": "Latvian (Supertonic-3, 10 speakers)",
    },
    "nld": {
        "lang_code": "nl",
        "description": "Dutch (Supertonic-3, 10 speakers)",
    },
    "pol": {
        "lang_code": "pl",
        "description": "Polish (Supertonic-3, 10 speakers)",
    },
    "por": {
        "lang_code": "pt",
        "description": "Portuguese (Supertonic-3, 10 speakers)",
    },
    "ron": {
        "lang_code": "ro",
        "description": "Romanian (Supertonic-3, 10 speakers)",
    },
    "rus": {
        "lang_code": "ru",
        "description": "Russian (Supertonic-3, 10 speakers)",
    },
    "slk": {
        "lang_code": "sk",
        "description": "Slovak (Supertonic-3, 10 speakers)",
    },
    "slv": {
        "lang_code": "sl",
        "description": "Slovenian (Supertonic-3, 10 speakers)",
    },
    "swe": {
        "lang_code": "sv",
        "description": "Swedish (Supertonic-3, 10 speakers)",
    },
    "tur": {
        "lang_code": "tr",
        "description": "Turkish (Supertonic-3, 10 speakers)",
    },
    "ukr": {
        "lang_code": "uk",
        "description": "Ukrainian (Supertonic-3, 10 speakers)",
    },
    "vie": {
        "lang_code": "vi",
        "description": "Vietnamese (Supertonic-3, 10 speakers)",
    },
    "ind-supertonic": {
        "lang_code": "id",
        "description": "Indonesian (Supertonic-3, 10 speakers)",
    },
    "jpn-supertonic": {
        "lang_code": "ja",
        "description": "Japanese (Supertonic-3, 10 speakers)",
    },
}

# Merge supertonic entries into _TTS_MODELS
for _code, _meta in _SUPERTONIC_LANGUAGES.items():
    _TTS_MODELS[_code] = {
        **_SUPERTONIC_BASE,
        "lang_code": _meta["lang_code"],
        "description": _meta["description"],
    }

_SUPPORTED_LANGS = ", ".join(
    f"{code} ({meta['description']})" for code, meta in _TTS_MODELS.items()
)

_LANGUAGE_ALIASES = {
    # English
    "en": "eng",
    "en-us": "eng",
    "en-gb": "eng",
    # English Kitten
    "eng-kitten": "eng-kitten",
    # Korean
    "ko": "kor",
    "kor": "kor",
    # Japanese
    "ja": "jpn",
    "jp": "jpn",
    "ja-jp": "jpn",
    # Japanese Sarashina (zero-shot)
    "sarashina": "jpn-sarashina",
    "jpn_sarashina": "jpn-sarashina",
    # Japanese Sarashina ONNX (torch-free runtime)
    "sarashina-onnx": "jpn-sarashina-onnx",
    "jpn_sarashina_onnx": "jpn-sarashina-onnx",
    # Japanese Supertonic-3
    "jpn-supertonic": "jpn-supertonic",
    # Arabic
    "ar": "ara",
    "ara": "ara",
    # Bulgarian
    "bg": "bul",
    "bul": "bul",
    # Czech
    "cs": "ces",
    "ces": "ces",
    # Danish
    "da": "dan",
    "dan": "dan",
    # German
    "de": "deu",
    "ger": "deu",
    "de-de": "deu",
    # Greek
    "el": "ell",
    "ell": "ell",
    # Spanish
    "es": "spa",
    "es-es": "spa",
    # Estonian
    "et": "est",
    "est": "est",
    # Finnish
    "fi": "fin",
    "fin": "fin",
    # French
    "fr": "fra",
    "fre": "fra",
    "fr-fr": "fra",
    # Hindi
    "hi": "hin",
    "hin": "hin",
    # Croatian
    "hr": "hrv",
    "hrv": "hrv",
    # Hungarian
    "hu": "hun",
    "hun": "hun",
    # Indonesian
    "id": "ind",
    "id-id": "ind",
    "ind": "ind",
    "ind-supertonic": "ind-supertonic",
    # Italian
    "it": "ita",
    "ita": "ita",
    # Lithuanian
    "lt": "lit",
    "lit": "lit",
    # Latvian
    "lv": "lav",
    "lav": "lav",
    # Dutch
    "nl": "nld",
    "nld": "nld",
    "dut": "nld",
    # Polish
    "pl": "pol",
    "pol": "pol",
    # Portuguese
    "pt": "por",
    "por": "por",
    # Romanian
    "ro": "ron",
    "ron": "ron",
    "rum": "ron",
    # Russian
    "ru": "rus",
    "rus": "rus",
    # Slovak
    "sk": "slk",
    "slk": "slk",
    # Slovenian
    "sl": "slv",
    "slv": "slv",
    # Swedish
    "sv": "swe",
    "swe": "swe",
    # Turkish
    "tr": "tur",
    "tur": "tur",
    # Ukrainian
    "uk": "ukr",
    "ukr": "ukr",
    # Vietnamese
    "vi": "vie",
    "vie": "vie",
    # Chinese
    "zh": "zho",
    "zh-cn": "zho",
    "zh-tw": "zho",
    "cmn": "zho",
    "chi": "zho",
}


def _info(msg: str) -> None:
    _console.print(f"[bold green]\\[info][/bold green] {msg}")


def _error(msg: str) -> None:
    _err_console.print(f"[bold red]\\[error][/bold red] {msg}")
    raise ConfigError(msg)


def _normalize_language(language: str) -> str:
    normalized = language.lower().replace("_", "-")
    return _LANGUAGE_ALIASES.get(normalized, normalized)


def _require_soundfile():
    global sf
    if getattr(sf, "write", None) is not None:
        return sf
    try:
        import soundfile as _soundfile  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise AudioError(
            "soundfile is required for writing synthesized audio. "
            "Install it with: pip install soundfile"
        ) from exc
    sf = _soundfile
    return sf


def _require_piper_plus():
    global piper_runtime
    if piper_runtime is not None:
        return piper_runtime
    try:
        from piper.download import ensure_voice_exists, find_voice, get_voices  # noqa: PLC0415
        from piper.voice import PiperVoice  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        _error(
            "piper-plus is required for Japanese TTS. "
            "Install it with: pip install piper-plus"
        )
        raise AssertionError("unreachable") from exc
    piper_runtime = SimpleNamespace(
        ensure_voice_exists=ensure_voice_exists,
        find_voice=find_voice,
        get_voices=get_voices,
        PiperVoice=PiperVoice,
    )
    return piper_runtime


def _require_sarashina():
    global sarashina_runtime
    if sarashina_runtime is not None:
        return sarashina_runtime
    try:
        from sarashina_tts.generate.generate import SarashinaTTSGenerator  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        _error(
            "sarashina-tts is required for Sarashina Japanese TTS. "
            "Install it with: pip install 'sherox[tts-ja-sarashina]' or "
            "git clone https://github.com/sbintuitions/sarashina2.2-tts && pip install -e sarashina2.2-tts"
        )
        raise AssertionError("unreachable") from exc
    sarashina_runtime = SimpleNamespace(SarashinaTTSGenerator=SarashinaTTSGenerator)
    return sarashina_runtime


def _quantize_sarashina_llm(generator) -> None:
    """Dynamically int8-quantize the Sarashina LLM's linear layers for faster CPU decoding.

    torch's dynamic quantization only has CPU kernels (fbgemm/qnnpack) — it's
    unavailable on CUDA, so this is only applied on the CPU-only code path.
    Benchmarked ~4.5x faster LLM decode on CPU with no observed change in
    generated semantic tokens.
    """
    import torch  # noqa: PLC0415

    text_generator = getattr(generator, "text_generator", None)
    llm = getattr(text_generator, "model", None)
    if llm is None:
        return
    quantized = torch.ao.quantization.quantize_dynamic(
        llm.float(), {torch.nn.Linear}, dtype=torch.qint8
    )
    text_generator.model = quantized


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if args.speaker_id < 0:
        _error(f"--speaker-id must be >= 0, got {args.speaker_id}")
    if args.speed <= 0:
        _error(f"--speed must be > 0, got {args.speed}")
    if args.threads <= 0:
        _error(f"--threads must be > 0, got {args.threads}")
    if (args.no_save or _output_disables_save(args.output)) and not args.play:
        _error("--no-save, --output none, and --output - require --play so generated audio is used.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text-to-speech synthesis with Sherpa-ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    text_src = parser.add_mutually_exclusive_group()
    text_src.add_argument("--text", metavar="TEXT", help="Text to synthesise")
    text_src.add_argument("--file", metavar="PATH", help="Read text from a file")

    parser.add_argument(
        "--lang",
        default="ind",
        metavar="LANG",
        help=f"ISO 639-3 language code. Supported: {_SUPPORTED_LANGS}",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        metavar="PATH",
        help=(
            "Path to a custom TTS model directory. "
            "If omitted, the default model for --lang is used."
        ),
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        metavar="N",
        help="Speaker identity index for multi-speaker models",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="F",
        help="Speech rate multiplier (0.5 = slower, 2.0 = faster)",
    )
    parser.add_argument(
        "--output",
        default="output.wav",
        metavar="PATH",
        help="Output WAV file path. Use 'none' or '-' with --play to disable saving.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play audio through the default output device after synthesis",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write a WAV file. Requires --play.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="CPU thread count for ONNX runtime",
    )
    parser.add_argument(
        "--audio-prompt",
        default=None,
        metavar="PATH",
        help="Reference WAV for zero-shot voice cloning (jpn-sarashina backend only)",
    )
    parser.add_argument(
        "--audio-prompt-text",
        default="",
        metavar="TEXT",
        help="Transcript of the --audio-prompt reference audio",
    )
    parser.add_argument(
        "--watermark",
        action="store_true",
        help="Embed an inaudible SilentCipher watermark (jpn-sarashina backend only). "
        "Off by default: adds ~15s to model load and ~40%% to each synthesis call.",
    )
    return parser.parse_args()


# ── Model download helpers ────────────────────────────────────────────────────

# Pre-exported ONNX artifacts for jpn-sarashina-onnx, published from this
# project — see sherox/sarashina_onnx_hf.py. Downloading these means end users
# never need torch or the original PyTorch checkpoint for default-voice use.
_SARASHINA_ONNX_HF_REPO = "Bagus/Sarashina2.2-TTS-ONNX"


def _ensure_sarashina_onnx_model(target_dir: Path) -> None:
    """Download the pre-exported Sarashina ONNX artifacts if not already present."""
    if (target_dir / "meta.json").is_file():
        return

    from . import model_cache  # noqa: PLC0415

    if model_cache.try_link(target_dir, "tts_jpn-sarashina-onnx"):
        return

    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        _error(
            "huggingface_hub is required to auto-download the Sarashina ONNX model. "
            "Install it with: pip install 'sherox[tts-ja-sarashina-onnx]'"
        )
        raise AssertionError("unreachable") from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    _info(f"Sarashina ONNX model not found. Downloading from {_SARASHINA_ONNX_HF_REPO}…")
    snapshot_download(_SARASHINA_ONNX_HF_REPO, local_dir=str(target_dir))

    if not (target_dir / "meta.json").is_file():
        _error(f"Expected 'meta.json' not found in downloaded model at {target_dir}")

    model_cache.migrate(target_dir, "tts_jpn-sarashina-onnx")
    _info(f"Model saved to '{target_dir}'.\n")


def _ensure_model(lang: str, model_dir: Optional[Path], project_dir: Path) -> Path:
    """Return the resolved TTS model directory, downloading if needed."""
    lang = _normalize_language(lang)
    if lang not in _TTS_MODELS:
        _error(
            f"Unsupported language '{lang}'. Supported: {list(_TTS_MODELS.keys())}"
        )

    meta = _TTS_MODELS[lang]
    if meta["backend"] != "sherpa_onnx":
        _error(
            f"Language '{lang}' uses the '{meta['backend']}' backend and does not "
            "support sherpa-onnx model auto-resolution."
        )
    if model_dir is not None:
        if not model_dir.is_dir():
            _error(f"Model directory not found: {model_dir}")
        return model_dir

    models_root = project_dir / "models"
    target_dir = models_root / meta["extracted"]

    if target_dir.is_dir():
        return target_dir

    from . import model_cache

    if model_cache.try_link(target_dir, f"tts_{lang}"):
        return target_dir

    # Download and extract
    models_root.mkdir(parents=True, exist_ok=True)
    archive = models_root / meta["archive"]
    _info(f"TTS model for '{lang}' not found.")
    _download_file(meta["url"], archive)

    _info("Extracting…")
    try:
        with tarfile.open(archive, "r:bz2") as tf:
            if sys.version_info >= (3, 12):
                tf.extractall(models_root, filter="data")
            else:  # pragma: no cover
                tf.extractall(models_root, members=_safe_tar_members(tf, models_root))
    except Exception as exc:
        _error(f"Extraction failed: {exc}")

    archive.unlink(missing_ok=True)

    if not target_dir.is_dir():
        _error(f"Expected model directory not found after extraction: {target_dir}")

    model_cache.migrate(target_dir, f"tts_{lang}")
    _info(f"Model saved to '{target_dir}'.\n")
    return target_dir


# ── TTS engine ────────────────────────────────────────────────────────────────

def build_tts(cfg: TtsConfig, project_dir: Path):
    """Build a TTS backend instance from *cfg*."""

    lang = _normalize_language(cfg.language)
    if lang not in _TTS_MODELS:
        _error(
            f"Unsupported language '{lang}'. Supported: {list(_TTS_MODELS.keys())}"
        )

    meta = _TTS_MODELS[lang]
    if meta["backend"] == "piper_plus":
        if cfg.model_dir:
            _error(
                "--model-dir is not currently supported for Piper Plus models. "
                "Use the built-in 'jpn' language model."
            )
        piper_plus_mod = _require_piper_plus()
        models_root = project_dir / "models" / "piper-plus"
        models_root.mkdir(parents=True, exist_ok=True)
        voices = piper_plus_mod.get_voices(models_root, update_voices=False)
        piper_plus_mod.ensure_voice_exists(
            meta["voice_name"],
            [models_root],
            models_root,
            voices,
        )
        model_path, config_path = piper_plus_mod.find_voice(meta["voice_name"], [models_root])
        return SimpleNamespace(
            backend="piper_plus",
            model=piper_plus_mod.PiperVoice.load(model_path, config_path),
            language_id=meta["language_id"],
        )

    if meta["backend"] == "sarashina":
        sarashina_mod = _require_sarashina()
        models_root = project_dir / "models" / "sarashina"
        models_root.mkdir(parents=True, exist_ok=True)
        model_dir = cfg.model_dir if cfg.model_dir else str(models_root)
        import torch  # noqa: PLC0415
        use_cuda = torch.cuda.is_available()
        generator = sarashina_mod.SarashinaTTSGenerator(
            model_dir=model_dir,
            decoder_fp16=use_cuda,
            watermark=cfg.watermark,
        )
        if not use_cuda:
            _quantize_sarashina_llm(generator)
        return SimpleNamespace(
            backend="sarashina",
            model=generator,
            prompt_cache=OrderedDict(),
        )

    if meta["backend"] == "sarashina_onnx":
        from .sarashina_onnx import SarashinaOnnxRuntime  # noqa: PLC0415

        models_root = project_dir / "models" / "sarashina-onnx"
        if cfg.model_dir:
            model_dir = cfg.model_dir
            if not (Path(model_dir) / "meta.json").is_file():
                _error(
                    f"ONNX artifacts not found in '{model_dir}'. Export them with:\n"
                    "  python -m sherox.sarashina_onnx_export "
                    "--model-dir models/sarashina --out-dir <model_dir>"
                )
        else:
            _ensure_sarashina_onnx_model(models_root)
            model_dir = str(models_root)
        runtime = SarashinaOnnxRuntime(model_dir, num_threads=cfg.num_threads)
        return SimpleNamespace(
            backend="sarashina_onnx",
            model=runtime,
            model_dir=model_dir,
            # Zero-shot cloning re-uses the torch-based feature extractors, which
            # need the original checkpoint (flow.pt, campplus, …) — not the ONNX
            # artifacts. Point them at the standard sarashina checkpoint dir.
            torch_model_dir=str(project_dir / "models" / "sarashina"),
            prompt_cache=OrderedDict(),
        )

    if meta["backend"] == "kitten":
        models_root = project_dir / "models" / "kitten"
        models_root.mkdir(parents=True, exist_ok=True)
        target_dir = models_root / meta["extracted"]

        if target_dir.is_dir():
            return SimpleNamespace(
                backend="kitten",
                model=str(target_dir),
            )

        # Download and extract
        _info(f"TTS model for '{lang}' not found.")
        archive = models_root / meta["archive"]
        _download_file(meta["url"], archive)

        _info("Extracting…")
        try:
            with tarfile.open(archive, "r:bz2") as tf:
                if sys.version_info >= (3, 12):
                    tf.extractall(models_root, filter="data")
                else:  # pragma: no cover
                    tf.extractall(models_root, members=_safe_tar_members(tf, models_root))
        except Exception as exc:
            _error(f"Extraction failed: {exc}")

        archive.unlink(missing_ok=True)

        if not target_dir.is_dir():
            _error(f"Expected model directory not found after extraction: {target_dir}")

        _info(f"Model saved to '{target_dir}'.\n")
        return SimpleNamespace(
            backend="kitten",
            model=str(target_dir),
        )

    if meta["backend"] == "supertonic":
        models_root = project_dir / "models" / "supertonic"
        models_root.mkdir(parents=True, exist_ok=True)
        target_dir = models_root / meta["extracted"]

        if not target_dir.is_dir():
            _info(f"TTS model for supertonic-3 not found.")
            archive = models_root / meta["archive"]
            _download_file(meta["url"], archive)

            _info("Extracting…")
            try:
                with tarfile.open(archive, "r:bz2") as tf:
                    if sys.version_info >= (3, 12):
                        tf.extractall(models_root, filter="data")
                    else:  # pragma: no cover
                        tf.extractall(models_root, members=_safe_tar_members(tf, models_root))
            except Exception as exc:
                _error(f"Extraction failed: {exc}")

            archive.unlink(missing_ok=True)

            if not target_dir.is_dir():
                _error(f"Expected model directory not found after extraction: {target_dir}")

            _info(f"Model saved to '{target_dir}'.\n")

        import sherpa_onnx  # noqa: PLC0415

        model_dir_override = Path(cfg.model_dir) if cfg.model_dir else None
        model_dir = model_dir_override if model_dir_override else target_dir

        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                    duration_predictor=str(model_dir / meta["files"]["duration_predictor"]),
                    text_encoder=str(model_dir / meta["files"]["text_encoder"]),
                    vector_estimator=str(model_dir / meta["files"]["vector_estimator"]),
                    vocoder=str(model_dir / meta["files"]["vocoder"]),
                    tts_json=str(model_dir / meta["files"]["tts_json"]),
                    unicode_indexer=str(model_dir / meta["files"]["unicode_indexer"]),
                    voice_style=str(model_dir / meta["files"]["voice_style"]),
                ),
                num_threads=cfg.num_threads,
            ),
        )

        if not config.validate():
            _error(
                "TTS config is invalid — check that all model files exist and are valid."
            )

        return SimpleNamespace(
            backend="supertonic",
            model=sherpa_onnx.OfflineTts(config),
            lang_code=meta["lang_code"],
            sample_rate=meta["sample_rate"],
        )

    import sherpa_onnx  # noqa: PLC0415

    model_dir_override = Path(cfg.model_dir) if cfg.model_dir else None
    model_dir = _ensure_model(lang, model_dir_override, project_dir)

    # Build path strings for optional model fields.
    # Only join model_dir with the field when the field is non-empty — an empty
    # data_dir or lexicon should pass "" to sherpa-onnx, not str(model_dir / "").
    lexicon_path = str(model_dir / meta["lexicon"]) if meta.get("lexicon") else ""
    data_dir_path = str(model_dir / meta["data_dir"]) if meta.get("data_dir") else ""

    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(model_dir / meta["model"]),
                lexicon=lexicon_path,
                data_dir=data_dir_path,
                tokens=str(model_dir / meta["tokens"]),
            ),
            num_threads=cfg.num_threads,
        ),
    )

    if not config.validate():
        _error(
            "TTS config is invalid — check that all model files exist and are valid."
        )

    return SimpleNamespace(
        backend="sherpa_onnx",
        model=sherpa_onnx.OfflineTts(config),
    )


def synthesise(tts, text: str, cfg: TtsConfig) -> tuple[np.ndarray, int]:
    """Synthesise *text* and return (samples, sample_rate)."""
    explicit_backend = getattr(tts, "__dict__", {}).get("backend")
    engine = getattr(tts, "__dict__", {}).get("model", tts) if explicit_backend else tts
    audio = engine.generate(text=text, sid=cfg.speaker_id, speed=cfg.speed)
    samples = np.array(audio.samples, dtype=np.float32)
    return samples, audio.sample_rate


def _output_disables_save(output: str) -> bool:
    return output.lower() in {"none", "-"}


def _should_save(cfg: TtsConfig) -> bool:
    return not cfg.no_save and not _output_disables_save(cfg.output)


# Zero-shot voice cloning re-encodes the reference wav (semantic tokens, speaker
# embedding, mel features) on every call. Callers that reuse the same
# --audio-prompt across many requests (e.g. tts_server with a fixed voice)
# shouldn't pay that cost more than once per distinct file.
_PROMPT_CACHE_MAX = 16


def _prompt_cache_key(path: str) -> tuple:
    st = Path(path).stat()
    return (path, st.st_mtime_ns, st.st_size)


def _get_cached_audio_prompt(generator, audio_prompt_path: str, cache: OrderedDict):
    """Return (tokens, flow_embedding, feat) for *audio_prompt_path*, cached by mtime+size."""
    key = _prompt_cache_key(audio_prompt_path)
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached

    result = (
        generator._extract_audio_prompt_tokens(audio_prompt_path),
        generator._extract_zero_shot_embedding(audio_prompt_path),
        generator._extract_audio_prompt_feat(audio_prompt_path),
    )
    cache[key] = result
    if len(cache) > _PROMPT_CACHE_MAX:
        cache.popitem(last=False)
    return result


def synthesise_to_file(tts, text: str, cfg: TtsConfig) -> Optional[tuple[np.ndarray, int]]:
    """Synthesise *text*, optionally writing cfg.output.

    Returns audio when available in memory for playback.
    """
    backend = getattr(tts, "__dict__", {}).get("backend", "sherpa_onnx")
    should_save = _should_save(cfg)

    if backend == "sherpa_onnx":
        samples, sample_rate = synthesise(tts, text, cfg)
        if should_save:
            soundfile = _require_soundfile()
            soundfile.write(cfg.output, samples, samplerate=sample_rate)
        return samples, sample_rate

    if backend == "piper_plus":
        length_scale = 1.0 / cfg.speed
        wav_target = cfg.output if should_save else io.BytesIO()
        with wave.open(wav_target, "wb") as wav_file:
            tts.model.synthesize(
                text,
                wav_file,
                speaker_id=cfg.speaker_id,
                length_scale=length_scale,
                language_id=getattr(tts, "language_id", None),
            )
        # Always return in-memory samples, matching every other backend's
        # contract - callers that only want the audio bytes (an API server,
        # for example) shouldn't have to request playback to get them. The
        # CLI's own cfg.play check decides whether to actually play them.
        soundfile = _require_soundfile()
        if should_save:
            samples, sample_rate = soundfile.read(cfg.output, dtype="float32")
        else:
            wav_target.seek(0)
            samples, sample_rate = soundfile.read(wav_target, dtype="float32")
        return np.asarray(samples, dtype=np.float32), sample_rate

    if backend == "sarashina":
        generator = tts.model
        audio_prompt_path = cfg.audio_prompt or None
        audio_prompt_text = cfg.audio_prompt_text or ""

        if audio_prompt_path:
            prompt_cache = getattr(tts, "prompt_cache", None)
            if prompt_cache is None:
                prompt_cache = OrderedDict()
                tts.prompt_cache = prompt_cache
            audio_prompt_tokens, flow_embedding, audio_prompt_feat = _get_cached_audio_prompt(
                generator, audio_prompt_path, prompt_cache,
            )
            wavs = generator.generate(
                [text],
                flow_embedding=flow_embedding,
                audio_prompt_text=audio_prompt_text,
                audio_prompt_tokens=audio_prompt_tokens,
                audio_prompt_feat=audio_prompt_feat,
                audio_prompt_path=audio_prompt_path,
            )
        else:
            wavs = generator.generate([text], flow_embedding=None)

        # generator.generate() returns (1, T) tensors (channel dim first); flatten
        # to the plain 1-D array every other backend in this module produces.
        samples = wavs[0].squeeze(0).cpu().numpy().astype(np.float32)
        sample_rate = 24000
        if should_save:
            soundfile = _require_soundfile()
            soundfile.write(cfg.output, samples, samplerate=sample_rate)
        return samples, sample_rate

    if backend == "sarashina_onnx":
        runtime = tts.model
        audio_prompt_path = cfg.audio_prompt or None

        if audio_prompt_path:
            from .sarashina_onnx import extract_prompt_features  # noqa: PLC0415

            prompt_cache = getattr(tts, "prompt_cache", None)
            if prompt_cache is None:
                prompt_cache = OrderedDict()
                tts.prompt_cache = prompt_cache
            key = _prompt_cache_key(audio_prompt_path)
            cached = prompt_cache.get(key)
            if cached is None:
                torch_dir = getattr(tts, "torch_model_dir", tts.model_dir)
                cached = extract_prompt_features(audio_prompt_path, torch_dir)
                prompt_cache[key] = cached
            else:
                prompt_cache.move_to_end(key)
            audio_prompt_tokens, flow_embedding, prompt_feat = cached
            samples, sample_rate = runtime.synthesise(
                text,
                audio_prompt_text=cfg.audio_prompt_text or "",
                audio_prompt_tokens=audio_prompt_tokens,
                flow_embedding=flow_embedding,
                prompt_feat=prompt_feat,
            )
        else:
            samples, sample_rate = runtime.synthesise(text)

        samples = np.asarray(samples, dtype=np.float32)
        if should_save:
            soundfile = _require_soundfile()
            soundfile.write(cfg.output, samples, samplerate=sample_rate)
        return samples, sample_rate

    if backend == "kitten":
        import sherpa_onnx  # noqa: PLC0415

        lang = _normalize_language(cfg.language)
        meta = _TTS_MODELS[lang]

        model_path = Path(tts.model) / meta["model"]
        voices_path = Path(tts.model) / meta["voices"]
        tokens_path = Path(tts.model) / meta["tokens"]
        data_dir_path = Path(tts.model) / meta["data_dir"]

        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kitten=sherpa_onnx.OfflineTtsKittenModelConfig(
                    model=str(model_path),
                    voices=str(voices_path),
                    tokens=str(tokens_path),
                    data_dir=str(data_dir_path),
                ),
                num_threads=cfg.num_threads,
            ),
        )

        if not config.validate():
            _error(
                "TTS config is invalid — check that all model files exist and are valid."
            )

        tts_instance = sherpa_onnx.OfflineTts(config)
        audio = tts_instance.generate(text=text, sid=cfg.speaker_id, speed=cfg.speed)
        samples = np.array(audio.samples, dtype=np.float32)
        sample_rate = audio.sample_rate

        if should_save:
            soundfile = _require_soundfile()
            soundfile.write(cfg.output, samples, samplerate=sample_rate)
        return samples, sample_rate

    if backend == "supertonic":
        import sherpa_onnx  # noqa: PLC0415

        lang_code = getattr(tts, "lang_code", "en")

        gen_config = sherpa_onnx.GenerationConfig()
        gen_config.sid = cfg.speaker_id
        gen_config.num_steps = 8
        gen_config.speed = cfg.speed
        gen_config.extra = {"lang": lang_code}

        audio = tts.model.generate(text, gen_config)
        samples = np.array(audio.samples, dtype=np.float32)
        sample_rate = tts.sample_rate

        if should_save:
            soundfile = _require_soundfile()
            soundfile.write(cfg.output, samples, samplerate=sample_rate)
        return samples, sample_rate

    _error(f"Unsupported TTS backend: {backend}")
    raise AssertionError("unreachable")


def _play(samples: np.ndarray, sample_rate: int) -> None:
    try:
        import sounddevice as sd  # noqa: PLC0415
    except ImportError:
        _error(
            "sounddevice is required for --play. "
            "Install it with: pip install sounddevice"
        )
    sd.play(samples, samplerate=sample_rate)
    sd.wait()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    _run_cli(_main_impl)


def _main_impl() -> None:
    args = parse_args()
    _validate_runtime_args(args)

    project_dir = Path(__file__).resolve().parent.parent

    # Resolve text input
    if args.text:
        text = args.text
    elif args.file:
        p = Path(args.file)
        if not p.exists():
            _error(f"Input file not found: {args.file}")
        text = p.read_text(encoding="utf-8").strip()
    else:
        if sys.stdin.isatty():
            _console.print("[dim]Reading text from stdin (Ctrl+D to finish)…[/dim]")
        text = sys.stdin.read().strip()

    if not text:
        _error("No text provided. Use --text, --file, or pipe text via stdin.")

    if args.audio_prompt and not Path(args.audio_prompt).exists():
        _error(f"Audio prompt file not found: {args.audio_prompt}")

    model_dir_arg = Path(args.model_dir) if args.model_dir else None
    cfg = TtsConfig(
        model_dir=str(model_dir_arg) if model_dir_arg else "",
        language=_normalize_language(args.lang),
        speaker_id=args.speaker_id,
        speed=args.speed,
        output=args.output,
        play=args.play,
        no_save=args.no_save or _output_disables_save(args.output),
        num_threads=args.threads,
        audio_prompt=args.audio_prompt or "",
        audio_prompt_text=args.audio_prompt_text or "",
        watermark=args.watermark,
    )

    _info(f"Language: {cfg.language}  |  speed: {cfg.speed}  |  speaker: {cfg.speaker_id}")
    _info("Loading TTS model…")
    tts = build_tts(cfg, project_dir)
    _info("Synthesising…")

    result = synthesise_to_file(tts, text, cfg)

    if cfg.play:
        if result is None:
            _error("Playback requested but no audio samples were returned.")
        samples, sample_rate = result
        _info("Playing audio…")
        _play(samples, sample_rate)

    if _should_save(cfg):
        _info(f"Saved → {cfg.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
