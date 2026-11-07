"""Support direct script execution from Linux."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def require_linux():
    if sys.platform != "linux":
        raise SystemExit("Run RULER QA on Linux; Windows is for editing only.")
