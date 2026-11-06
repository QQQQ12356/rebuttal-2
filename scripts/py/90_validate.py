"""Static contract checks; no project ML modules are imported or executed."""
import ast
import json
from _bootstrap import ROOT


def main():
    files = [path for folder in ("seco", "scripts/py", "scripts/niah/py", "tests") for path in (ROOT / folder).rglob("*.py")]
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in (ROOT / "configs").rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    for path in (ROOT / "scripts/sh").glob("*.sh"):
        if b"\r" in path.read_bytes():
            raise ValueError(f"Shell file contains CRLF: {path}")
    original = ast.parse((ROOT / "legacy/py/modeling_seco_cluster.py").read_text(encoding="utf-8"))
    current = ast.parse((ROOT / "seco/compression.py").read_text(encoding="utf-8"))

    def methods(tree):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name in ("_compress_one", "_compress_batch")}

    if methods(original) != methods(current):
        raise ValueError("SeCo compression methods differ from the original baseline")
    for path in (ROOT / "scripts/sh").glob("[0-9]*.sh"):
        if not (ROOT / "scripts/py" / f"{path.stem}.py").exists():
            # Setup calls pip plus the separate environment inspector; validation
            # wraps the static checker and Linux unittest suite.
            if path.stem != "00_setup_environment":
                raise ValueError(f"Missing matching Python command: {path}")
    print(f"Static checks passed: {len(files)} Python files; JSON, LF shell files and original compression AST")


if __name__ == "__main__":
    main()
