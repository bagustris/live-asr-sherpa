"""Shared utilities for the sherox package."""
import sys
import urllib.request
from pathlib import Path

from rich.console import Console

_console = Console()
_err_console = Console(stderr=True)


def _info(msg: str) -> None:
    _console.print(f"[bold green]\\[info][/bold green] {msg}")


def _error(msg: str) -> None:
    _err_console.print(f"[bold red]\\[error][/bold red] {msg}")
    sys.exit(1)


def download_file(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` with a simple percentage progress callback.
    Supports resuming interrupted downloads."""
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
