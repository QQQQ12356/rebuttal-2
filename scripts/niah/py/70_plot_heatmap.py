"""Plot complete NIAH length/depth grids as one figure with one panel per ratio."""
import argparse
import glob

from niah_bootstrap import require_linux
from seco.io import project_path
from seco.niah.plotting import plot_heatmaps


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Metrics paths or quoted glob patterns")
    parser.add_argument("--output", required=True, help="New .png file; also writes PDF and cell CSV")
    parser.add_argument("--metric", choices=("accuracy", "em", "f1", "hit_rate"), default="accuracy")
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--title", default="SeCo Needle-in-a-Haystack")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--annotate-all", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    paths = []
    for pattern in args.inputs:
        found = glob.glob(str(project_path(pattern)), recursive=True)
        if not found:
            raise FileNotFoundError(f"No NIAH metrics matched: {pattern}")
        paths.extend(project_path(path) for path in found)
    # Sort numerically by ratio rather than lexical ratio_16 before ratio_4.
    from seco.io import read_json
    paths = sorted(set(paths), key=lambda path: (read_json(path).get("ratio", 0), str(path)))
    outputs = plot_heatmaps(paths, project_path(args.output), metric=args.metric, columns=args.columns,
                            title=args.title, dpi=args.dpi, allow_incomplete=args.allow_incomplete,
                            annotate_all=args.annotate_all, pdf=not args.no_pdf)
    print("NIAH figure artifacts:\n" + "\n".join(map(str, outputs)))


if __name__ == "__main__":
    main()
