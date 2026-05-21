"""sherox — top-level CLI entry point.

Usage:
    sherox --version
    sherox -v
    sherox --help
"""

import argparse

from sherox import __version__

_SUBCOMMANDS = {
    "asr":     "Streaming / offline automatic speech recognition",
    "tts":     "Text-to-speech synthesis",
    "sid":     "Speaker identification",
    "lid":     "Language identification",
    "segment": "VAD-based audio segmentation",
    "kws":     "Keyword / wake-word spotting",
    "server":  "HTTP/WebSocket ASR server",
}


def _build_parser() -> argparse.ArgumentParser:
    lines = ["Available subcommands:"]
    for name, desc in _SUBCOMMANDS.items():
        lines.append(f"  sherox.{name:<10}  {desc}")
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
    import sys
    parser = _build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return
    parser.parse_args()
