"""Stream selected dataset splits into raw JSONL and record immutable source commits."""
import argparse
from itertools import islice

from _bootstrap import require_linux
from seco.io import project_path, read_json, sha256_file, write_json, write_jsonl


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/datasets.json")
    parser.add_argument("--output", default="artifacts/raw")
    parser.add_argument("--role", choices=("all", "training", "benchmark"), default="all")
    parser.add_argument("--datasets", nargs="+", help="Canonical names; omit for all in selected roles")
    parser.add_argument("--train-limit", type=int, default=30000, help="First N raw rows per training source; 0 = all")
    args = parser.parse_args()
    if args.train_limit < 0:
        parser.error("--train-limit must be >= 0")
    from datasets import load_dataset
    from huggingface_hub import HfApi

    registry, output = read_json(project_path(args.registry)), project_path(args.output)
    roles = ("training", "benchmark") if args.role == "all" else (args.role,)
    for role in roles:
        selected = args.datasets or list(registry[role])
        unknown = set(selected) - set(registry[role])
        if unknown:
            parser.error(f"Unknown {role} datasets: {sorted(unknown)}")
        for name in selected:
            spec = registry[role][name]
            destination = output / role / f"{name}.jsonl"
            manifest_path = destination.with_suffix(".source.json")
            commit = HfApi().dataset_info(spec["repo"], revision=spec["revision"]).sha
            limit = args.train_limit if role == "training" else 0
            identity = {"spec": spec, "resolved_revision": commit, "limit": limit}
            if destination.exists() or manifest_path.exists():
                if destination.is_file() and manifest_path.is_file():
                    old = read_json(manifest_path)
                    if old["identity"] == identity and old["sha256"] == sha256_file(destination):
                        print(f"Verified existing raw data: {destination}")
                        continue
                raise FileExistsError(f"Raw source changed or incomplete: {destination}; use a new output directory")
            if spec.get("converted_parquet"):
                # Avoid executing an old dataset loading script. The official HF
                # conversion branch exposes this source as ordinary Parquet files.
                files = HfApi().list_repo_files(spec["repo"], repo_type="dataset", revision=commit)
                prefix = f"{spec['config']}/{spec['split']}/"
                files = sorted(path for path in files if path.startswith(prefix) and path.endswith(".parquet"))
                if not files:
                    raise ValueError(f"No converted parquet files under {prefix}: {spec['repo']}@{commit}")
                urls = [f"hf://datasets/{spec['repo']}@{commit}/{path}" for path in files]
                source = load_dataset("parquet", data_files={spec["split"]: urls},
                                      split=spec["split"], streaming=True)
            else:
                source = load_dataset(spec["repo"], name=spec["config"], revision=commit,
                                      split=spec["split"], streaming=True)
            count = 0

            def rows():
                nonlocal count
                iterator = islice(source, limit) if limit else source
                for row in iterator:
                    count += 1
                    yield row

            write_jsonl(destination, rows())
            if count == 0:
                raise ValueError(f"Empty source: {spec}")
            write_json(manifest_path, {"identity": identity, "count": count, "sha256": sha256_file(destination)})
            print(f"Downloaded {role}/{name}: {count} rows from {commit}")


if __name__ == "__main__":
    main()
