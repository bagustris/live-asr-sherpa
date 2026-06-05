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

    # Read from file:
    sherox.tts --file input.txt --lang ind

    # Read from stdin:
    echo "Halo dunia" | sherox.tts --lang ind

    # Save to a specific output file:
    sherox.tts --text "Halo" --output halo.wav

    # Play through the system speaker (requires sounddevice):
    sherox.tts --text "Halo" --play

    # Control speech speed:
    sherox.tts --text "Halo" --speed 0.85

Supported languages (ISO 639-3 code → model):
    eng           English US   — vits-piper-en_US-amy-medium       (22050 Hz, 1 speaker)
    deu           German       — vits-piper-de_DE-thorsten-medium   (22050 Hz, 1 speaker)
    fra           French       — vits-piper-fr_FR-upmc-medium       (22050 Hz, 1 speaker)
    spa           Spanish      — vits-piper-es_ES-mls_10246-medium  (22050 Hz, 1 speaker)
    ind           Indonesian   — vits-piper-id_ID-news_tts-medium   (22050 Hz, 1 speaker)
    zho           Chinese      — vits-icefall-zh-aishell3           (8 kHz, 174 speakers)
    jpn           Japanese     — piper-plus tsukuyomi               (22050 Hz, 1 speaker)
    jpn-sarashina Japanese     — Sarashina2.2-TTS, zero-shot        (24000 Hz, voice cloning)

Language aliases (short forms also accepted):
    en / eng-us                 → eng
    de / ger                    → deu
    fr                          → fra
    es                          → spa
    id / id-id                  → ind
    zh / zh-cn / zh-tw / cmn    → zho
    ja / jp / ja-jp             → jpn
    sarashina / jpn_sarashina   → jpn-sarashina

Notes:
    Chinese (zho): input must be plain Simplified Chinese text; numbers and mixed
    scripts may not normalise well. Use 8 kHz output; quality is acceptable for
    voice assistants and dev/test use.

Models are auto-downloaded on first use into  models/<model-dir>/  at the project root.
"""

import argparse
import sys
import tarfile
import wave
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
}

_SUPPORTED_LANGS = ", ".join(
    f"{code} ({meta['description']})" for code, meta in _TTS_MODELS.items()
)

_LANGUAGE_ALIASES = {
    # English
    "en": "eng",
    "en-us": "eng",
    "en-gb": "eng",
    # German
    "de": "deu",
    "ger": "deu",
    "de-de": "deu",
    # French
    "fr": "fra",
    "fre": "fra",
    "fr-fr": "fra",
    # Spanish
    "es": "spa",
    "es-es": "spa",
    # Indonesian
    "id": "ind",
    "id-id": "ind",
    # Chinese
    "zh": "zho",
    "zh-cn": "zho",
    "zh-tw": "zho",
    "cmn": "zho",
    "chi": "zho",
    # Japanese
    "ja": "jpn",
    "jp": "jpn",
    "ja-jp": "jpn",
    # Japanese Sarashina (zero-shot)
    "sarashina": "jpn-sarashina",
    "jpn_sarashina": "jpn-sarashina",
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


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if args.speaker_id < 0:
        _error(f"--speaker-id must be >= 0, got {args.speaker_id}")
    if args.speed <= 0:
        _error(f"--speed must be > 0, got {args.speed}")
    if args.threads <= 0:
        _error(f"--threads must be > 0, got {args.threads}")


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
        help="Output WAV file path",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play audio through the default output device after synthesis",
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
    return parser.parse_args()


# ── Model download helpers ────────────────────────────────────────────────────

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
        )
        return SimpleNamespace(
            backend="sarashina",
            model=generator,
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


def synthesise_to_file(tts, text: str, cfg: TtsConfig) -> Optional[tuple[np.ndarray, int]]:
    """Synthesise *text* to cfg.output. Returns audio when available in memory."""
    backend = getattr(tts, "__dict__", {}).get("backend", "sherpa_onnx")

    if backend == "sherpa_onnx":
        samples, sample_rate = synthesise(tts, text, cfg)
        soundfile = _require_soundfile()
        soundfile.write(cfg.output, samples, samplerate=sample_rate)
        return samples, sample_rate

    if backend == "piper_plus":
        length_scale = 1.0 / cfg.speed
        with wave.open(cfg.output, "wb") as wav_file:
            tts.model.synthesize(
                text,
                wav_file,
                speaker_id=cfg.speaker_id,
                length_scale=length_scale,
                language_id=getattr(tts, "language_id", None),
            )
        if not cfg.play:
            return None
        soundfile = _require_soundfile()
        samples, sample_rate = soundfile.read(cfg.output, dtype="float32")
        return np.asarray(samples, dtype=np.float32), sample_rate

    if backend == "sarashina":
        generator = tts.model
        audio_prompt_path = cfg.audio_prompt or None
        audio_prompt_text = cfg.audio_prompt_text or ""

        if audio_prompt_path:
            audio_prompt_tokens = generator._extract_audio_prompt_tokens(audio_prompt_path)
            flow_embedding = generator._extract_zero_shot_embedding(audio_prompt_path)
            audio_prompt_feat = generator._extract_audio_prompt_feat(audio_prompt_path)
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

        samples = wavs[0].cpu().numpy().astype(np.float32)
        sample_rate = 24000
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
        num_threads=args.threads,
        audio_prompt=args.audio_prompt or "",
        audio_prompt_text=args.audio_prompt_text or "",
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

    _info(f"Saved → {cfg.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
