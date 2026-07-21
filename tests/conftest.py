import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Add repo root so `from sherox.x import ...` works without installing the package.
sys.path.insert(0, str(_REPO_ROOT))

# Allow test modules to import from benchmark/
sys.path.insert(0, str(_REPO_ROOT / "benchmark"))


@pytest.fixture(autouse=True)
def _isolate_model_cache(tmp_path_factory, monkeypatch):
    """Point sherox.model_cache at a throwaway dir so no test ever migrates
    fake/mocked model directories into the real ~/.cache/sherox/models."""
    monkeypatch.setenv("SHEROX_CACHE_DIR", str(tmp_path_factory.mktemp("sherox_cache")))
