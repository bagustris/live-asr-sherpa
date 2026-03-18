import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Add repo root so `from sherox.x import ...` works without installing the package.
sys.path.insert(0, str(_REPO_ROOT))

# Allow test modules to import from benchmark/
sys.path.insert(0, str(_REPO_ROOT / "benchmark"))
