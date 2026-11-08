"""Recompute deterministic NIAH grid metrics from saved generation results."""
import argparse

from niah_bootstrap import require_linux
from seco.io import fingerprint, iter_jsonl, project_path, read_json, write_json
from seco.niah.metrics import aggregate


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--grid-manifest", help="Otherwise read the embedded manifest from benchmark.json")
    parser.add_argument("--output", help="Default: sibling metrics.json")
    parser.add_argument("--allow-incomplete", action="store_true", help="Mark partial cells explicitly; never fill them with zeros")
    args = parser.parse_args()
    source = project_path(args.predictions)
    metadata = {} if args.grid_manifest else read_json(source.parent.parent / "benchmark.json")
    manifest = read_json(project_path(args.grid_manifest)) if args.grid_manifest else metadata["grid_manifest"]
    rows = list(iter_jsonl(source))
    selection = metadata.get("sample_selection")
    if selection and (len(rows) != selection["count"]
                      or fingerprint(sorted(row["id"] for row in rows)) != selection["id_hash"]):
        raise ValueError("Prediction IDs differ from the selected NIAH samples")
    metrics = aggregate(rows, manifest["grid"], allow_incomplete=args.allow_incomplete or bool(selection))
    ratios = {row["ratio"] for row in rows}
    if len(ratios) != 1:
        raise ValueError("Score one compression ratio per prediction file")
    metrics["ratio"] = ratios.pop()
    metrics["grid_identity"] = fingerprint([manifest, selection] if selection else manifest)
    output = project_path(args.output) if args.output else source.parent / "metrics.json"
    if output.exists():
        metrics = {**read_json(output), **metrics}
    write_json(output, metrics)
    print(f"NIAH metrics: {output}; complete={metrics['complete']}; {metrics['summary']}")


if __name__ == "__main__":
    main()
