"""Download independent NIAH training splits and held-out haystack corpora."""
import argparse
from io import BytesIO
from itertools import islice
import json
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import ZipFile

from niah_bootstrap import require_linux
from seco.io import project_path, read_json, sha256_file, write_json, write_jsonl


def github_rows(spec):
    revision = quote(spec["revision"], safe="")
    request = Request(f"https://api.github.com/repos/{spec['repo']}/commits/{revision}",
                      headers={"User-Agent": "SeCo-NIAH", "Accept": "application/vnd.github+json"})
    with urlopen(request, timeout=120) as response:
        commit = json.load(response)["sha"]
    with urlopen(f"https://codeload.github.com/{spec['repo']}/zip/{commit}", timeout=180) as response:
        payload = response.read()

    def rows():
        # Read only selected text members; never extract archive paths to disk.
        with ZipFile(BytesIO(payload)) as archive:
            for name in sorted(archive.namelist()):
                relative = name.split("/", 1)[-1]
                if relative.startswith(spec["subdirectory"].rstrip("/") + "/") and relative.endswith(spec["suffix"]):
                    yield {"id": relative, "text": archive.read(name).decode("utf-8-sig"),
                           "source_url": f"https://github.com/{spec['repo']}/blob/{commit}/{relative}"}
    return commit, rows()


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/niah/sources.json")
    parser.add_argument("--output", default="artifacts/niah/raw")
    parser.add_argument("--sources", nargs="+", help="Default: enabled entries in registry")
    parser.add_argument("--limit", type=int, help="Override maximum raw rows per source; 0 = all")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")
    from datasets import load_dataset
    from huggingface_hub import HfApi

    registry = read_json(project_path(args.registry))
    names = args.sources or [name for name, spec in registry.items() if spec.get("enabled", True)]
    if len(set(names)) != len(names) or set(names) - set(registry):
        parser.error("Unknown or duplicate --sources")
    for name in names:
        spec = registry[name]
        if spec["role"] in ("haystack_train", "supervised_train") and spec.get("split") != "train":
            raise ValueError(f"Training source {name} must explicitly use split=train")
        output = project_path(args.output) / f"{name}.jsonl"
        sidecar = output.with_suffix(".source.json")
        limit = spec.get("limit", 0) if args.limit is None else args.limit
        if spec["kind"] == "hf":
            commit = HfApi().dataset_info(spec["repo"], revision=spec["revision"]).sha
            source = load_dataset(spec["repo"], name=spec["config"], revision=commit,
                                  split=spec["split"], streaming=True)
        elif spec["kind"] == "github":
            commit, source = github_rows(spec)
        else:
            raise ValueError(f"Unsupported source kind: {spec['kind']}")
        identity = {"spec": spec, "resolved_revision": commit, "limit": limit}
        if output.exists() or sidecar.exists():
            if output.is_file() and sidecar.is_file():
                saved = read_json(sidecar)
                if saved["identity"] == identity and saved["sha256"] == sha256_file(output):
                    print(f"Verified NIAH source: {name}")
                    continue
            raise FileExistsError(f"Changed/incomplete raw data: {output}; choose a new output root")
        count = 0

        def rows():
            nonlocal count
            for index, row in enumerate(islice(source, limit) if limit else source):
                fields = spec.get("fields")
                result = {key: row[key] for key in fields if key in row} if fields else dict(row)
                result.setdefault("id", str(row.get("url", index)))
                result["source"], result["source_split"] = name, spec.get("split", "heldout_files")
                count += 1
                yield result

        write_jsonl(output, rows())
        if not count:
            raise ValueError(f"Empty source: {name}")
        write_json(sidecar, {"identity": identity, "count": count, "sha256": sha256_file(output)})
        print(f"Downloaded {name}: {count} rows @ {commit}")


if __name__ == "__main__":
    main()
