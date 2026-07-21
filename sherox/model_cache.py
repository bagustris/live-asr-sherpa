"""Cross-project model cache.

Downloaded model directories are large (tens of MB to 1 GB+) and often
identical across every project that depends on sherox — e.g. two sibling
repos both pulling in the Parakeet TDT ASR model or the same Piper TTS
voice. Rather than each project's local `models/` holding its own full
copy, the bytes live once in a shared cache (mirroring how
`huggingface_hub` caches downloads under `~/.cache/huggingface/hub`) and
each project's `models/<name>` becomes a symlink into it.

Override the cache location with the `SHEROX_CACHE_DIR` env var.
"""
import errno
import os
import shutil
from pathlib import Path
from typing import Callable


def cache_root() -> Path:
    override = os.environ.get("SHEROX_CACHE_DIR")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "sherox" / "models"


def _cache_key(project_dir: Path, model_type: str) -> str:
    """`model_type` is folded into the key because different model types can
    land in identically-named project directories (e.g. a generic `models/model`
    override), which would otherwise collide in the shared cache."""
    return f"{model_type}__{project_dir.name}" if model_type else project_dir.name


def try_link(project_dir: Path, model_type: str = "") -> bool:
    """If a cached copy keyed by `project_dir` (and `model_type`) exists,
    symlink `project_dir` to it and return True. Returns False (no-op) if
    nothing is cached yet, or if the cache entry exists but isn't a usable
    directory (e.g. a plain file or a broken symlink left over from a
    previous failure)."""
    cached = cache_root() / _cache_key(project_dir, model_type)
    if not cached.is_dir():
        return False
    _relink(project_dir, cached)
    return True


def migrate(project_dir: Path, model_type: str = "") -> None:
    """Move an already-downloaded `project_dir` into the shared cache and
    replace it with a symlink, so sibling projects reuse it instead of
    downloading their own copy. No-op if `project_dir` is already a symlink
    (e.g. a previous migration already ran)."""
    if project_dir.is_symlink():
        return
    cached = cache_root() / _cache_key(project_dir, model_type)
    if cached.is_dir():
        # Another project already populated the cache for this model
        # first (race or pre-existing migration) — reuse it and drop this
        # copy rather than overwriting the cache.
        shutil.rmtree(project_dir)
    else:
        if cached.exists() or cached.is_symlink():
            # A file or broken symlink occupies the cache slot — clear it
            # rather than silently failing to migrate.
            cached.unlink()
        cached.parent.mkdir(parents=True, exist_ok=True)
        try:
            project_dir.rename(cached)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            # project_dir and the cache root are on different filesystems
            # (e.g. Docker overlay mounts) — rename() can't do a cross-device
            # move, so copy the bytes across and drop the original.
            try:
                shutil.copytree(project_dir, cached)
            except FileExistsError:
                pass  # a concurrent migrate() populated it first
            shutil.rmtree(project_dir)
    _relink(project_dir, cached)


def _relink(project_dir: Path, cached_dir: Path) -> None:
    project_dir.parent.mkdir(parents=True, exist_ok=True)
    if project_dir.is_symlink() and not project_dir.exists():
        project_dir.unlink()
    if not project_dir.exists():
        try:
            project_dir.symlink_to(cached_dir, target_is_directory=True)
        except OSError:
            # Symlinks aren't available (e.g. no Developer Mode on Windows,
            # or a filesystem like FAT32/exFAT that doesn't support them) —
            # fall back to a real copy so the model is still usable.
            shutil.copytree(cached_dir, project_dir)


def ensure_model(model_dir: str, model_type: str, download_fn: Callable[[str, str], None]) -> str:
    """Ensure `model_dir` exists: untouched if already a real directory,
    symlinked if another project already cached it, or freshly downloaded
    via `download_fn(model_dir, model_type)` and then migrated into the
    shared cache for reuse by sibling projects."""
    path = Path(model_dir)
    if path.is_dir():
        return str(path)
    if try_link(path, model_type):
        return str(path)

    download_fn(model_dir, model_type)

    if path.is_dir() and not path.is_symlink():
        migrate(path, model_type)

    return str(path)
