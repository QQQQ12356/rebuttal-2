"""Direct Linux entrypoints for the independent NIAH workflow."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def require_linux():
    if sys.platform != "linux":
        raise SystemExit("NIAH experiments run on Linux; Windows is for editing and static checks only.")
