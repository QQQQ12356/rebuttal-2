"""Static NIAH checks; safe on Windows because no experiment code is executed."""
import ast
import json
import re

from niah_bootstrap import ROOT


def main():
    files = [path for folder in ("seco", "scripts/niah/py", "tests/niah") for path in (ROOT / folder).rglob("*.py")]
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in (ROOT / "configs/niah").glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    for path in (ROOT / "scripts/niah/sh").glob("*.sh"):
        text = path.read_text(encoding="utf-8")
        if b"\r" in path.read_bytes():
            raise ValueError(f"CRLF shell: {path}")
        if path.stem != "_common" and path.stem != "00_setup_environment":
            if not (ROOT / "scripts/niah/py" / f"{path.stem}.py").exists():
                raise ValueError(f"Missing NIAH Python entry: {path}")
        for relative in re.findall(r"scripts/niah/py/([\w]+\.py)", text):
            if not (ROOT / "scripts/niah/py" / relative).is_file():
                raise ValueError(f"Missing shell target: {relative}")
    print(f"NIAH static checks passed: {len(files)} Python files, JSON and shell entrypoints/LF")


if __name__ == "__main__":
    main()
