import sys
from pathlib import Path

# Allow test modules to import from src/ the same way the source files do
# (i.e. `from config import Config` rather than `from src.config import Config`)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
