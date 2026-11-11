"""Set the project root for direct ICAE pretraining entry points."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ["COMPRESSION_METHOD"] = "icae"


def require_linux():
    if sys.platform != "linux":
        raise SystemExit("Run ICAE pretraining on Linux")
