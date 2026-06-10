"""Shared utilities for the sherox package."""
import sys
import urllib.request
from pathlib import Path
from typing import Callable

from audiokit import safe_tar_members  # noqa: F401  re-exported; shared impl
from rich.console import Console

from . import SherpaError

_console = Console()
_err_console = Console(stderr=True)


def _info(msg: str) -> None:
    _console.print(f"[bold green]\\[info][/bold green] {msg}")


def _error(msg: str) -> None:
    _err_console.print(f"[bold red]\\[error][/bold red] {msg}")
    raise SherpaError(msg)


def render_mic_level(chunk, prefix: str = "  ") -> None:
    """Write a live RMS energy bar for ``chunk`` to stdout in place (no newline).

    Used by all microphone-capture commands to give visual feedback on input
    level. Overwrites the current line with ``\\r``; callers are responsible for
    clearing or advancing the line before printing other output.
    """
    import numpy as np  # noqa: PLC0415

    energy = float(np.sqrt(np.mean(chunk ** 2)))
    bar = "█" * min(int(energy * 500), 40)
    sys.stdout.write(f"\r{prefix}mic: {bar:<40} {energy:.4f}")
    sys.stdout.flush()


def download_file(url: str, dest: Path | str) -> None:
    """Download ``url`` to ``dest`` with a simple percentage progress callback.
    Supports resuming interrupted downloads."""
    dest = Path(dest)
    _info(f"Downloading from:\n  {url}")
    _info("This may take a few minutes…")

    # Check if partial download exists
    existing_size = 0
    if dest.exists():
        existing_size = dest.stat().st_size
        if existing_size > 0:
            _info(f"Resuming from {existing_size} bytes…")

    req = urllib.request.Request(url)
    if existing_size > 0:
        req.add_header("Range", f"bytes={existing_size}-")

    try:
        with urllib.request.urlopen(req) as response:
            # If we requested a range but server ignored it (200 OK), don't append
            if existing_size > 0 and response.status != 206:
                _info("Server does not support resume — restarting download…")
                existing_size = 0
                dest.write_bytes(b"")

            # Get total size from Content-Range header if resuming, or Content-Length otherwise
            if "Content-Range" in response.headers:
                # Format: "bytes start-end/total"
                content_range = response.headers["Content-Range"]
                total_size = int(content_range.split("/")[-1])
            else:
                total_size = int(response.headers.get("Content-Length", 0))

            mode = "ab" if existing_size > 0 else "wb"
            with dest.open(mode) as f:
                downloaded = existing_size
                chunk_size = 8192
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = min(100, downloaded * 100 // total_size)
                        sys.stdout.write(f"\r  {pct}%")
                        sys.stdout.flush()
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and existing_size > 0:
            _info("Server rejected resume range — restarting download…")
            dest.unlink(missing_ok=True)
            download_file(url, dest)
            return
        _error(f"Download failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        _error(f"Download failed: {exc}")
    print()


def run_cli(impl: Callable[[], None]) -> None:
    """Run a CLI entrypoint, mapping library exceptions to exit codes.

    ``SherpaError`` (and subclasses) → exit 1; ``KeyboardInterrupt`` → exit 130.
    """
    try:
        impl()
    except SherpaError:
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
