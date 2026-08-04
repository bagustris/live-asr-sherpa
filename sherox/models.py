"""sherox list-models — display all auto-downloadable models.

Usage::

    sherox list-models               # table of all modules
    sherox list-models --module asr  # only ASR models
    sherox list-models --module tts  # only TTS models

Output columns: Module, Name (target directory / model key), Language(s),
Pipeline (online/offline/both), Approx. size.

Sizes are the approximate *download* size of the archive (not extracted).
They are hard-coded here because the models may not be installed yet.

Example::

    $ sherox list-models
    ┌──────────┬─────────────────────────────────┬────────────────┬──────────┬──────────┐
    │ Module   │ Name                            │ Language       │ Pipeline │ Size     │
    ├──────────┼─────────────────────────────────┼────────────────┼──────────┼──────────┤
    │ asr      │ parakeet-tdt-0.6b-v2-int8       │ en             │ offline  │ ~330 MB  │
    │ asr      │ parakeet-tdt-0.6b-v2 (fp16)     │ en             │ offline  │ ~660 MB  │
    │ ...      │ ...                             │ ...            │ ...      │ ...      │
    └──────────┴─────────────────────────────────┴────────────────┴──────────┴──────────┘
"""
from __future__ import annotations

import argparse
from typing import Optional

from rich.console import Console
from rich.table import Table

# ── ASR model registry ────────────────────────────────────────────────────────
# Each entry: (name, languages, pipeline, approx_size_mb, description)
#   pipeline: "online" | "offline" | "both"
#   approx_size_mb: download archive size in MB (approximate)
#
# Sources:
#   https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
#   https://huggingface.co/reazon-research (ReazonSpeech)
#   https://huggingface.co/CohereLabs/cohere-transcribe-03-2026

_ASR_MODELS: list[tuple[str, str, str, str, str]] = [
    # (target_name, languages, pipeline, size, notes)
    (
        "parakeet-tdt-0.6b-v2-int8",
        "en",
        "offline",
        "~330 MB",
        "NeMo Parakeet TDT 0.6B int8 (default English)",
    ),
    (
        "parakeet-tdt-0.6b-v2 (fp16)",
        "en",
        "offline",
        "~660 MB",
        "NeMo Parakeet TDT 0.6B fp16 (higher quality)",
    ),
    (
        "zipformer-en-2023",
        "en",
        "online",
        "~70 MB",
        "Streaming Zipformer English 2023",
    ),
    (
        "reazonspeech-ja",
        "ja",
        "offline",
        "~130 MB",
        "ReazonSpeech Zipformer Japanese",
    ),
    (
        "reazonspeech-ja-en",
        "ja, en",
        "offline",
        "~140 MB",
        "ReazonSpeech Zipformer bilingual Japanese-English",
    ),
    (
        "reazonspeech-ja-en-mls-5k",
        "ja, en",
        "offline",
        "~140 MB",
        "ReazonSpeech + MLS English 5k bilingual",
    ),
    (
        "parakeet-ctc-ja-int8",
        "ja",
        "offline",
        "~280 MB",
        "NeMo Parakeet CTC 0.6B int8 Japanese",
    ),
    (
        "cohere-transcribe-14-lang-int8",
        "ar,de,en,es,fr,hi,id,it,ja,ko,nl,pt,ru,zh",
        "offline",
        "~800 MB",
        "Cohere Transcribe 14-language multilingual",
    ),
    (
        "sherpa-onnx-whisper-large-v3",
        "multilingual (99 langs)",
        "offline",
        "~3.0 GB",
        "Whisper Large-V3 multilingual",
    ),
    (
        "sherpa-onnx-whisper-turbo",
        "multilingual (99 langs)",
        "offline",
        "~540 MB",
        "Whisper Turbo multilingual",
    ),
    (
        "sherpa-onnx-whisper-distil-large-v3.5",
        "multilingual (99 langs)",
        "offline",
        "~505 MB",
        "Whisper Distil Large-V3.5 multilingual",
    ),
    (
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        "zh, en, ja, ko, yue",
        "offline",
        "~220 MB",
        "SenseVoice multilingual 5-language",
    ),
    (
        "zipformer-multilingual-2025-02-10",
        "ar, en, id, ja, ru, th, vi, zh",
        "online",
        "~100 MB",
        "Streaming Zipformer 8-language multilingual",
    ),
    (
        "zipformer-de-2025",
        "de",
        "online",
        "~60 MB",
        "Streaming Zipformer German (KroKo 2025)",
    ),
    (
        "nemo-de-int8",
        "de",
        "offline",
        "~140 MB",
        "NeMo FastConformer German int8",
    ),
]

# ── TTS model registry ────────────────────────────────────────────────────────

_TTS_MODELS: list[tuple[str, str, str, str, str]] = [
    ("eng (vits-piper-en_US-amy-medium)",   "en",  "offline", "~50 MB",  "Piper VITS English US (Amy)"),
    ("eng-kitten (kitten-nano-en-v0_8-int8)", "en", "offline", "~25 MB", "Kitten TTS Nano v0.8, quantized, 8 speakers"),
    ("deu (vits-piper-de_DE-thorsten-medium)", "de", "offline", "~50 MB", "Piper VITS German (Thorsten)"),
    ("fra (vits-piper-fr_FR-upmc-medium)",   "fr",  "offline", "~50 MB",  "Piper VITS French (UPMC)"),
    ("spa (vits-piper-es_ES-mls_10246-medium)", "es", "offline", "~50 MB", "Piper VITS Spanish (MLS)"),
    ("ind (vits-piper-id_ID-news_tts-medium)", "id", "offline", "~50 MB", "Piper VITS Indonesian"),
    ("zho (vits-icefall-zh-aishell3)", "zh", "offline", "~50 MB", "VITS Mandarin Chinese (AiShell3, 174 speakers)"),
    ("jpn (vits-piper-ja_JA-nakamura-medium)", "ja", "offline", "~50 MB", "Piper VITS Japanese (Nakamura)"),
    (
        "supertonic-3 (shared model)",
        "ko, ar, bg, cs, da, el, et, fi, hi, hr, hu, it, lt, lv, nl, "
        "pl, pt, ro, ru, sk, sl, sv, tr, uk, vi, id",
        "offline",
        "~120 MB",
        "Supertonic-3 multilingual TTS, 25 languages + alt Indonesian, 10 speakers each",
    ),
]

# ── SID / KWS / VAD model registry ────────────────────────────────────────────

_OTHER_MODELS: list[tuple[str, str, str, str, str, str]] = [
    # (module, name, languages, pipeline, size, description)
    ("sid",     "nemo_en_titanet_large",       "en",         "offline", "~96 MB",  "NeMo TitaNet-Large speaker ID"),
    ("kws",     "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
                "en", "online",  "~3 MB",   "Zipformer keyword spotter (GigaSpeech 3.3M)"),
    ("vad",     "silero_vad",                  "language-independent", "both", "~2 MB",   "Silero VAD (offline segmentation)"),
    ("vad",     "ten-vad.int8.onnx",           "language-independent", "both", "~2 MB",   "TEN-VAD int8 (faster alternative)"),
    ("punct",   "punct-ct-transformer-zh-en",  "zh, en",     "offline", "~40 MB",  "CT-Transformer punctuation restoration"),
    ("diarization", "sherpa-onnx-pyannote-segmentation-3-0", "en", "offline", "~30 MB", "Pyannote speaker segmentation"),
    ("diarization", "nemo_en_speakerverification_speakernet", "en", "offline", "~22 MB", "NeMo speaker embedding extractor"),
    ("wake", "user-provided ONNX (livekit-wakeword)", "language-independent", "online", "varies", "Custom wake-word models trained via livekit-wakeword"),
]

_console = Console()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sherox list-models",
        description=(
            "Show all auto-downloadable models with their languages, pipeline, "
            "and approximate download size."
        ),
    )
    parser.add_argument(
        "--module",
        choices=["asr", "tts", "other", "all"],
        default="all",
        help="Filter by module (default: all)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        dest="no_color",
        help="Disable colour output",
    )
    return parser


def print_model_table(module_filter: str = "all", no_color: bool = False) -> None:
    """Print a Rich table of all downloadable models.

    Parameters
    ----------
    module_filter:
        ``"all"`` shows every module; otherwise one of ``"asr"``, ``"tts"``,
        or ``"other"`` restricts the output.
    no_color:
        When ``True``, strip ANSI colour from the output (useful for piping).
    """
    console = Console(no_color=True) if no_color else _console

    table = Table(
        title="sherox — downloadable models",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
        row_styles=["", "dim"],
    )
    table.add_column("Module",   style="green",  no_wrap=True)
    table.add_column("Name",     style="white",  no_wrap=False, max_width=45)
    table.add_column("Language", style="yellow", no_wrap=False, max_width=30)
    table.add_column("Pipeline", style="cyan",   no_wrap=True)
    table.add_column("Size",     style="magenta",no_wrap=True)
    table.add_column("Notes",    style="dim white", no_wrap=False)

    if module_filter in ("asr", "all"):
        for name, lang, pipeline, size, notes in _ASR_MODELS:
            table.add_row("asr", name, lang, pipeline, size, notes)

    if module_filter in ("tts", "all"):
        for name, lang, pipeline, size, notes in _TTS_MODELS:
            table.add_row("tts", name, lang, pipeline, size, notes)

    if module_filter in ("other", "all"):
        for mod, name, lang, pipeline, size, notes in _OTHER_MODELS:
            table.add_row(mod, name, lang, pipeline, size, notes)

    console.print(table)
    console.print(
        "\n[dim]Models are auto-downloaded on first use to the [cyan]models/[/cyan] directory.[/dim]\n"
        if not no_color
        else "\nModels are auto-downloaded on first use to the models/ directory.\n"
    )


def main() -> None:
    """Entry point: sherox list-models."""
    args = _build_parser().parse_args()
    print_model_table(module_filter=args.module, no_color=args.no_color)
