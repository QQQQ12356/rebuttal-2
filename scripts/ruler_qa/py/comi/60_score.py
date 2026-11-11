"""Run COMI ruler_qa."""
import os
from pathlib import Path
import runpy
import sys

BACKBONE = "Llama-3.2-1B-Instruct"
DEFAULT_ARGS = ()  # Optional CLI defaults, e.g. ("--model", "/models/backbone").


if __name__ == "__main__":
    os.environ["COMPRESSION_METHOD"] = "comi"
    os.environ.setdefault("BACKBONE", BACKBONE)
    shared = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(shared))
    sys.argv[1:1] = DEFAULT_ARGS
    runpy.run_path(str(shared / "60_score.py"), run_name="__main__")
