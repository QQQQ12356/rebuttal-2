"""Allow direct CLI execution without an editable package installation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def require_linux():
    if sys.platform != "linux":
        raise SystemExit("Run this experiment command on Linux. Windows is for code editing only.")
