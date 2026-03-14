import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Allow test modules to import from src/ the same way the source files do
# (i.e. `from config import Config` rather than `from src.config import Config`)
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Allow test modules to import from benchmark/
sys.path.insert(0, str(_REPO_ROOT / "benchmark"))
