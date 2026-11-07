"""Resumable shard download followed by local, single-threaded record decoding."""
from __future__ import annotations

from fnmatch import fnmatchcase
import gzip
import json
from pathlib import Path
import re
import time


def retry_call(operation, *, description: str, attempts: int, delay: float, log=print):
    """Retry truncated HTTP bodies; fail fast on permission, disk and permanent HTTP errors."""
    import requests

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except (requests.RequestException, TimeoutError, ConnectionError, OSError) as error:
            transient, cause = False, error
            while cause is not None:
                status = getattr(getattr(cause, "response", None), "status_code", None)
                if status is not None and status < 500 and status not in (408, 425, 429):
                    transient = False
                    break
                transient |= (isinstance(cause, (requests.RequestException, TimeoutError, ConnectionError))
                              or "Consistency check failed" in str(cause))
                cause = cause.__cause__
            if not transient or attempt == attempts:
                raise
            wait = min(delay * 2 ** (attempt - 1), 30)
            # Exception strings can contain signed URLs. Log only the error class.
            log(f"[retry {attempt}/{attempts}] {description}: {type(error).__name__}; "
                f"retrying in {wait:g}s; partial Hub downloads are retained")
            time.sleep(wait)


def _patterns_for_split(value, split: str) -> list[str]:
    if isinstance(value, dict):
        selected = value.get(split, [])
        return [selected] if isinstance(selected, str) else list(selected)
    if isinstance(value, list):
        patterns = []
        for entry in value:
            if not isinstance(entry, dict) or "split" not in entry or "path" not in entry:
                raise ValueError("data_files entries must explicitly specify split and path")
            if entry["split"] == split:
                paths = entry["path"]
                patterns.extend([paths] if isinstance(paths, str) else paths)
        return patterns
    raise ValueError("data_files must explicitly map split names to file patterns")


def select_source_files(spec: dict, files: list[str], card: dict) -> list[str]:
    """Honor explicit split mappings; fail rather than silently selecting another config."""
    patterns = []
    if "data_files" in spec:
        patterns = _patterns_for_split(spec["data_files"], spec["split"])
    elif spec.get("converted_parquet"):
        patterns = [f"{spec['config']}/{spec['split']}/*.parquet"]
    else:
        configs = card.get("configs", []) or []
        for config in configs:
            if config.get("config_name", "default") == spec["config"] and "data_files" in config:
                patterns = _patterns_for_split(config["data_files"], spec["split"])
                break
        if not patterns:
            if len(configs) > 1:
                raise ValueError("No unambiguous config mapping; add data_files to the registry")
            supported = [path for path in files if _supported(path)]
            token = re.compile(r"(?:^|[/_.-])" + re.escape(spec["split"]) + r"(?:$|[/_.-])")
            selected = [path for path in supported if token.search(path)]
            info = card.get("dataset_info", {})
            if isinstance(info, list):
                info = next((entry for entry in info if entry.get("config_name", "default") == spec["config"]), {})
            declared = {entry["name"] for entry in info.get("splits", [])}
            if not selected and declared == {spec["split"]}:
                selected = supported
            if not selected:
                raise ValueError("Cannot identify split files; add an explicit data_files split mapping to the registry")
            return sorted(selected)
    selected = []
    for pattern in patterns:
        matches = sorted(path for path in files if fnmatchcase(path, pattern) and _supported(path))
        if not matches:
            raise ValueError(f"No supported dataset files match: {pattern}")
        for path in matches:
            if path not in selected:
                selected.append(path)
    if not selected:
        raise ValueError(f"Empty file mapping for {spec['config']}/{spec['split']}")
    return selected


def _supported(path: str) -> bool:
    return (path.endswith((".parquet", ".json", ".jsonl", ".json.gz", ".jsonl.gz"))
            and Path(path).name not in ("dataset_infos.json", "config.json"))


def read_local_records(path: Path, filename: str, *, batch_size: int):
    if filename.endswith(".parquet"):
        import pyarrow.parquet as pq

        # No remote filesystem and no Arrow background I/O.
        parquet = pq.ParquetFile(str(path), pre_buffer=False)
        try:
            for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
                yield from batch.to_pylist()
        finally:
            parquet.close()
    else:
        opener = gzip.open if filename.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8-sig") as handle:
            first = ""
            while not first:
                char = handle.read(1)
                if not char:
                    return
                if not char.isspace():
                    first = char
            handle.seek(0)
            if first == "[":
                for row in json.load(handle):
                    if not isinstance(row, dict):
                        raise ValueError(f"Expected JSON record objects in {filename}")
                    yield row
            else:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError(f"Expected JSONL record objects in {filename}")
                        yield row


def iter_local_shards(spec: dict, info, *, cache_dir: Path, limit: int, attempts: int,
                      delay: float, batch_size: int, transport: dict, log=print):
    from huggingface_hub import DatasetCard, hf_hub_download
    from seco.io import sha256_file

    files = [entry.rfilename for entry in info.siblings]

    def download(filename):
        return Path(retry_call(lambda: hf_hub_download(repo_id=spec["repo"], filename=filename,
                    repo_type="dataset", revision=info.sha, cache_dir=str(cache_dir)),
                    description=f"download {filename}", attempts=attempts, delay=delay, log=log))

    card = {}
    if "README.md" in files and "data_files" not in spec and not spec.get("converted_parquet"):
        log("[files] Reading pinned dataset card split mapping")
        card = DatasetCard(download("README.md").read_text(encoding="utf-8-sig")).data.to_dict()
    selected = select_source_files(spec, files, card)
    transport.update({"mode": "local", "available_shards": len(selected), "shards": []})
    log(f"[files] Selected {len(selected)} shard(s) for {spec['config']}/{spec['split']}; "
        "only shards needed by the row limit will be downloaded")
    count = 0
    for index, filename in enumerate(selected, 1):
        log(f"[download {index}/{len(selected)}] {filename}")
        local = download(filename)
        log(f"[local] Complete file, {local.stat().st_size / 2**20:.1f} MiB; reading {filename}")
        transport["shards"].append({"filename": filename, "size": local.stat().st_size,
                                     "sha256": sha256_file(local)})
        records = read_local_records(local, filename, batch_size=batch_size)
        try:
            for row in records:
                yield row
                count += 1
                if limit and count >= limit:
                    return
        finally:
            records.close()
