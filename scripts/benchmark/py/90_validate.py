"""Static contract checks; no project ML modules are imported or executed."""
import ast
import json
from _bootstrap import ROOT


class StripGroupStats(ast.NodeTransformer):
    """Remove optional instrumentation before comparing compression to legacy."""

    def visit_FunctionDef(self, node):
        for index in range(len(node.args.kwonlyargs) - 1, -1, -1):
            if node.args.kwonlyargs[index].arg in {"group_sizes", "group_stats"}:
                del node.args.kwonlyargs[index]
                del node.args.kw_defaults[index]
        return self.generic_visit(node)

    def visit_If(self, node):
        guards = (ast.parse(f"{name} is not None", mode="eval").body for name in ("group_sizes", "group_stats"))
        if any(ast.dump(node.test) == ast.dump(guard) for guard in guards):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        setup = ast.parse("group_sizes = [] if group_stats is not None else None").body[0]
        if ast.dump(node) == ast.dump(setup):
            return None
        return self.generic_visit(node)

    def visit_keyword(self, node):
        if node.arg == "group_sizes" and isinstance(node.value, ast.Name) and node.value.id == "group_sizes":
            return None
        return node


def main():
    files = [path for folder in ("seco", "comi", "icae", "ram", "scripts", "tests")
             for path in (ROOT / folder).rglob("*.py")]
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in (ROOT / "configs").rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    for path in (ROOT / "scripts").rglob("*.sh"):
        if b"\r" in path.read_bytes():
            raise ValueError(f"Shell file contains CRLF: {path}")
    original = ast.parse((ROOT / "legacy/py/modeling_seco_cluster.py").read_text(encoding="utf-8"))
    current = ast.parse((ROOT / "seco/compression.py").read_text(encoding="utf-8"))

    def methods(tree):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name in ("_compress_one", "_compress_batch")}

    if methods(original) != methods(StripGroupStats().visit(current)):
        raise ValueError("SeCo compression methods differ from the original baseline")
    for task in ("benchmark", "niah", "ruler_qa"):
        scripts = ROOT / "scripts" / task
        for method in ("seco", "comi", "icae", "ram"):
            config = json.loads((ROOT / "configs" / task / method / "train.json").read_text(encoding="utf-8"))
            if config["model"]["method"] != method:
                raise ValueError(f"Method mismatch in {task}/{method} training config")
            for entry in (scripts / "sh" / method).glob("[0-9]*.sh"):
                shared_entry = scripts / "sh" / entry.name
                if not shared_entry.is_file():
                    raise ValueError(f"Missing shared shell target: {entry}")
                if entry.stem == "80_export" and task in ("niah", "ruler_qa"):
                    if "-m seco.export_metrics" not in shared_entry.read_text(encoding="utf-8"):
                        raise ValueError(f"Missing metrics export module invocation: {shared_entry}")
                elif entry.stem != "00_setup_environment" and not (
                    scripts / "py" / method / f"{entry.stem}.py"
                ).is_file():
                    raise ValueError(f"Missing matching Python command: {entry}")
    print(f"Static checks passed: {len(files)} Python files; JSON, LF shell files and baseline compression logic")


if __name__ == "__main__":
    main()
