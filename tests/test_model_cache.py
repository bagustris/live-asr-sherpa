import errno
from pathlib import Path

import pytest

from sherox import model_cache


def _make_dir(path, filename="f.txt", content="hi"):
    path.mkdir(parents=True, exist_ok=True)
    (path / filename).write_text(content)
    return path


def test_cache_root_honors_sherox_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEROX_CACHE_DIR", str(tmp_path / "custom"))
    assert model_cache.cache_root() == tmp_path / "custom"


def test_cache_root_falls_back_to_xdg_cache_home(tmp_path, monkeypatch):
    monkeypatch.delenv("SHEROX_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert model_cache.cache_root() == tmp_path / "xdg" / "sherox" / "models"


def test_cache_key_includes_model_type(tmp_path):
    project_dir = tmp_path / "models" / "current"
    assert model_cache._cache_key(project_dir, "nemo_ctc") == "nemo_ctc__current"
    assert model_cache._cache_key(project_dir, "multilingual_streaming") != (
        model_cache._cache_key(project_dir, "nemo_ctc")
    )
    # No model_type given: key is just the directory name (backward compatible).
    assert model_cache._cache_key(project_dir, "") == "current"


def test_try_link_no_cache_entry_is_noop(tmp_path):
    project_dir = tmp_path / "proj" / "models" / "foo"
    assert model_cache.try_link(project_dir, "asr") is False
    assert not project_dir.exists()


def test_try_link_links_existing_cache_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEROX_CACHE_DIR", str(tmp_path / "cache"))
    cached = _make_dir(tmp_path / "cache" / "asr__foo")
    project_dir = tmp_path / "proj" / "models" / "foo"

    assert model_cache.try_link(project_dir, "asr") is True
    assert project_dir.is_symlink()
    assert project_dir.resolve() == cached.resolve()
    assert (project_dir / "f.txt").read_text() == "hi"


def test_try_link_ignores_cache_entry_that_is_a_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEROX_CACHE_DIR", str(tmp_path / "cache"))
    cache_slot = tmp_path / "cache" / "asr__foo"
    cache_slot.parent.mkdir(parents=True)
    cache_slot.write_text("not a directory")
    project_dir = tmp_path / "proj" / "models" / "foo"

    assert model_cache.try_link(project_dir, "asr") is False
    assert not project_dir.exists()


def test_migrate_moves_into_cache_and_symlinks(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEROX_CACHE_DIR", str(tmp_path / "cache"))
    project_dir = _make_dir(tmp_path / "proj" / "models" / "foo")

    model_cache.migrate(project_dir, "asr")

    assert project_dir.is_symlink()
    cached = tmp_path / "cache" / "asr__foo"
    assert cached.is_dir()
    assert (project_dir / "f.txt").read_text() == "hi"


def test_migrate_is_noop_when_already_symlink(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEROX_CACHE_DIR", str(tmp_path / "cache"))
    cached = _make_dir(tmp_path / "cache" / "asr__foo")
    project_dir = tmp_path / "proj" / "models" / "foo"
    project_dir.parent.mkdir(parents=True)
    project_dir.symlink_to(cached, target_is_directory=True)

    model_cache.migrate(project_dir, "asr")  # should not raise or touch anything

    assert project_dir.resolve() == cached.resolve()


def test_migrate_drops_local_copy_when_cache_already_populated(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEROX_CACHE_DIR", str(tmp_path / "cache"))
    cached = _make_dir(tmp_path / "cache" / "asr__foo", content="cached-version")
    project_dir = _make_dir(tmp_path / "proj" / "models" / "foo", content="local-version")

    model_cache.migrate(project_dir, "asr")

    assert project_dir.is_symlink()
    # The pre-existing cache content wins; the local copy was discarded.
    assert (project_dir / "f.txt").read_text() == "cached-version"
    assert cached.is_dir()


def test_migrate_clears_stale_file_occupying_cache_slot(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEROX_CACHE_DIR", str(tmp_path / "cache"))
    cache_slot = tmp_path / "cache" / "asr__foo"
    cache_slot.parent.mkdir(parents=True)
    cache_slot.write_text("stray file")
    project_dir = _make_dir(tmp_path / "proj" / "models" / "foo", content="real-model")

    model_cache.migrate(project_dir, "asr")

    assert project_dir.is_symlink()
    assert cache_slot.is_dir()
    assert (project_dir / "f.txt").read_text() == "real-model"


def test_migrate_falls_back_to_copy_on_cross_device_rename(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEROX_CACHE_DIR", str(tmp_path / "cache"))
    project_dir = _make_dir(tmp_path / "proj" / "models" / "foo")

    def _raise_exdev(self, target):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(Path, "rename", _raise_exdev)

    model_cache.migrate(project_dir, "asr")

    assert project_dir.is_symlink()
    cached = tmp_path / "cache" / "asr__foo"
    assert cached.is_dir()
    assert (project_dir / "f.txt").read_text() == "hi"


def test_migrate_reraises_non_exdev_oserror(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEROX_CACHE_DIR", str(tmp_path / "cache"))
    project_dir = _make_dir(tmp_path / "proj" / "models" / "foo")

    def _raise_other(self, target):
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(Path, "rename", _raise_other)

    with pytest.raises(OSError):
        model_cache.migrate(project_dir, "asr")


def test_relink_replaces_symlink_pointing_elsewhere(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEROX_CACHE_DIR", str(tmp_path / "cache"))
    wrong_target = _make_dir(tmp_path / "wrong", content="wrong")
    cached = _make_dir(tmp_path / "cache" / "asr__foo", content="correct")
    project_dir = tmp_path / "proj" / "models" / "foo"
    project_dir.parent.mkdir(parents=True)
    project_dir.symlink_to(wrong_target, target_is_directory=True)

    model_cache._relink(project_dir, cached)

    assert project_dir.resolve() == cached.resolve()
    assert (project_dir / "f.txt").read_text() == "correct"


def test_relink_replaces_stray_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEROX_CACHE_DIR", str(tmp_path / "cache"))
    cached = _make_dir(tmp_path / "cache" / "asr__foo", content="correct")
    project_dir = tmp_path / "proj" / "models" / "foo"
    project_dir.parent.mkdir(parents=True)
    project_dir.write_text("stray file")

    model_cache._relink(project_dir, cached)

    assert project_dir.is_symlink()
    assert (project_dir / "f.txt").read_text() == "correct"


def test_relink_falls_back_to_copy_when_symlinks_unsupported(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEROX_CACHE_DIR", str(tmp_path / "cache"))
    cached = _make_dir(tmp_path / "cache" / "asr__foo", content="correct")
    project_dir = tmp_path / "proj" / "models" / "foo"

    def _raise_oserror(self, target, target_is_directory=False):
        raise OSError("symlinks not supported")

    monkeypatch.setattr(Path, "symlink_to", _raise_oserror)

    model_cache._relink(project_dir, cached)

    assert project_dir.is_dir()
    assert not project_dir.is_symlink()
    assert (project_dir / "f.txt").read_text() == "correct"


def test_ensure_model_returns_existing_directory_without_downloading(tmp_path):
    project_dir = _make_dir(tmp_path / "proj" / "models" / "foo")
    calls = []

    result = model_cache.ensure_model(
        str(project_dir), "asr", lambda d, t: calls.append((d, t))
    )

    assert result == str(project_dir)
    assert calls == []


def test_ensure_model_links_from_cache_without_downloading(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEROX_CACHE_DIR", str(tmp_path / "cache"))
    _make_dir(tmp_path / "cache" / "asr__foo")
    project_dir = tmp_path / "proj" / "models" / "foo"
    calls = []

    result = model_cache.ensure_model(
        str(project_dir), "asr", lambda d, t: calls.append((d, t))
    )

    assert result == str(project_dir)
    assert project_dir.is_symlink()
    assert calls == []


def test_ensure_model_downloads_and_migrates(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEROX_CACHE_DIR", str(tmp_path / "cache"))
    project_dir = tmp_path / "proj" / "models" / "foo"

    def _download(model_dir, model_type):
        _make_dir(Path(model_dir))

    result = model_cache.ensure_model(str(project_dir), "asr", _download)

    assert result == str(project_dir)
    assert project_dir.is_symlink()
    assert (tmp_path / "cache" / "asr__foo").is_dir()


def test_ensure_model_raises_when_download_fn_does_not_create_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEROX_CACHE_DIR", str(tmp_path / "cache"))
    project_dir = tmp_path / "proj" / "models" / "foo"

    with pytest.raises(FileNotFoundError):
        model_cache.ensure_model(str(project_dir), "asr", lambda d, t: None)
