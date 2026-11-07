"""Download pinned dataset shards with retries, then convert local records to raw JSONL."""
import argparse
from itertools import islice
import time

from _bootstrap import require_linux
from seco.downloads import iter_local_shards, retry_call
from seco.io import project_path, read_json, sha256_file, write_json, write_jsonl


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/datasets.json")
    parser.add_argument("--output", default="artifacts/raw")
    parser.add_argument("--role", choices=("all", "training", "benchmark"), default="all")
    parser.add_argument("--datasets", nargs="+", help="Canonical names; omit for all in selected roles")
    parser.add_argument("--train-limit", type=int, default=30000, help="First N raw rows per training source; 0 = all")
    parser.add_argument("--download-mode", choices=("local", "stream"), default="local")
    parser.add_argument("--source-cache", default="artifacts/source_cache", help="Persistent Hub shard cache; retain for retries")
    parser.add_argument("--download-attempts", type=int, default=6)
    parser.add_argument("--metadata-timeout", type=float, default=60, help="Hub dataset-info request timeout in seconds")
    parser.add_argument("--retry-delay", type=float, default=2, help="Initial backoff seconds, capped at 30s")
    parser.add_argument("--row-batch-size", type=int, default=16, help="Local Parquet batch size, with Arrow threads disabled")
    parser.add_argument("--progress-every", type=int, default=100, help="Log every N written records")
    args = parser.parse_args()
    if (args.train_limit < 0 or min(args.download_attempts, args.row_batch_size, args.progress_every) < 1
            or not 0 <= args.retry_delay <= 30 or not 0 < args.metadata_timeout <= 3600):
        parser.error("Invalid row limit, retry, batch or progress settings")
    from huggingface_hub import HfApi

    def log(message):
        print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message, flush=True)

    registry, output = read_json(project_path(args.registry)), project_path(args.output)
    roles = ("training", "benchmark") if args.role == "all" else (args.role,)
    api = HfApi()
    for role in roles:
        selected = args.datasets or list(registry[role])
        unknown = set(selected) - set(registry[role])
        if unknown or len(selected) != len(set(selected)):
            parser.error(f"Unknown or duplicate {role} datasets: {sorted(unknown)}")
        for name in selected:
            started = time.monotonic()
            spec = registry[role][name]
            destination = output / role / f"{name}.jsonl"
            manifest_path = destination.with_suffix(".source.json")
            log(f"[start] {role}/{name}: {spec['repo']} ({spec['config']}/{spec['split']}), mode={args.download_mode}")
            info = retry_call(lambda: api.dataset_info(spec["repo"], revision=spec["revision"], timeout=args.metadata_timeout),
                              description="dataset metadata", attempts=args.download_attempts,
                              delay=args.retry_delay, log=log)
            limit = args.train_limit if role == "training" else 0
            identity = {"spec": spec, "resolved_revision": info.sha, "limit": limit}
            if destination.exists() or manifest_path.exists():
                if destination.is_file() and manifest_path.is_file():
                    old = read_json(manifest_path)
                    if old["identity"] == identity and old["sha256"] == sha256_file(destination):
                        log(f"[reuse] Verified existing raw data: {destination}")
                        continue
                raise FileExistsError(f"Raw source changed or incomplete: {destination}; use a new output directory")
            transport = {"mode": args.download_mode}
            if args.download_mode == "local":
                source = iter_local_shards(spec, info, cache_dir=project_path(args.source_cache), limit=limit,
                                           attempts=args.download_attempts, delay=args.retry_delay,
                                           batch_size=args.row_batch_size, transport=transport, log=log)
            else:
                from datasets import load_dataset
                log("[stream] Opening remote dataset; network reads occur inside Arrow and are not shard-resumable")
                if spec.get("converted_parquet"):
                    prefix = f"{spec['config']}/{spec['split']}/"
                    files = sorted(entry.rfilename for entry in info.siblings
                                   if entry.rfilename.startswith(prefix) and entry.rfilename.endswith(".parquet"))
                    if not files:
                        raise ValueError(f"No converted parquet files under {prefix}")
                    urls = [f"hf://datasets/{spec['repo']}@{info.sha}/{path}" for path in files]
                    dataset = load_dataset("parquet", data_files={spec["split"]: urls}, split=spec["split"], streaming=True)
                else:
                    dataset = load_dataset(spec["repo"], name=spec["config"], revision=info.sha,
                                           split=spec["split"], streaming=True)
                source = iter(dataset)
            count = 0

            def rows():
                nonlocal count
                try:
                    iterator = islice(source, limit) if limit else source
                    for row in iterator:
                        count += 1
                        yield row
                        if count == 1 or count % args.progress_every == 0:
                            log(f"[rows] {role}/{name}: {count} records; elapsed={time.monotonic() - started:.1f}s")
                    if count == 0:
                        raise ValueError(f"Empty source: {spec}")
                finally:
                    close = getattr(source, "close", None)
                    if close:
                        close()

            write_jsonl(destination, rows())
            write_json(manifest_path, {"identity": identity, "count": count,
                       "sha256": sha256_file(destination), "transport": transport})
            log(f"[done] {role}/{name}: {count} records from {info.sha}; elapsed={time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
