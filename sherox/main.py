"""sherox — top-level CLI entry point.

Usage:
    sherox --version
    sherox -v
    sherox --help
    sherox list-models [--module asr|tts|other|all] [--no-color]
"""

import argparse
import sys

from sherox import __version__

_SUBCOMMANDS = {
    "asr":          "Streaming / offline automatic speech recognition",
    "tts":          "Text-to-speech synthesis",
    "sid":          "Speaker identification",
    "lid":          "Language identification",
    "segment":      "VAD-based audio segmentation",
    "kws":          "Keyword / wake-word spotting",
    "wake":         "Wake-word detection (livekit-wakeword, custom ONNX)",
    "server":       "HTTP/WebSocket ASR server",
    "list-models":  "Show all auto-downloadable models with sizes and languages",
}


def _build_parser() -> argparse.ArgumentParser:
    lines = ["Available subcommands:"]
    for name, desc in _SUBCOMMANDS.items():
        lines.append(f"  {'sherox.' + name if name != 'list-models' else 'sherox list-models':<26}  {desc}")
    epilog = "\n".join(lines) + "\n\nRun 'sherox.<sub> --help' for per-command options."

    parser = argparse.ArgumentParser(
        prog="sherox",
        description="SHErpa OnnX toolkit — streaming ASR, VAD segmentation, and TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"sherox {__version__}",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    # Intercept 'sherox list-models' before argparse sees it, so it works as
    # a two-word subcommand (sherox list-models ...) without a subparser tree.
    if len(sys.argv) >= 2 and sys.argv[1] == "list-models":
        from sherox.models import main as models_main  # noqa: PLC0415
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        models_main()
        return
    if len(sys.argv) == 1:
        parser.print_help()
        return
    parser.parse_args()
