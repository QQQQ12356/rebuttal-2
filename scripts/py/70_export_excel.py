"""Import scored benchmark JSON/JSONL artifacts into one Excel workbook."""
import argparse
import glob

from _bootstrap import require_linux
from seco.io import project_path
from seco.reporting import export_excel


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=["artifacts/benchmarks/**/metrics.json"],
                        help="Paths or quoted recursive glob patterns")
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-details", action="store_true", help="Load sibling predictions.jsonl contents too")
    args = parser.parse_args()
    paths = []
    for pattern in args.inputs:
        matches = glob.glob(str(project_path(pattern)), recursive=True)
        if not matches:
            raise FileNotFoundError(f"No metric files matched: {pattern}")
        paths.extend(project_path(path) for path in matches)
    result = export_excel(paths, project_path(args.output), include_details=args.include_details)
    print(f"Exported {result}: {project_path(args.output)}")


if __name__ == "__main__":
    main()
