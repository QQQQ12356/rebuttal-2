"""Download the original train/dev data and verify every source checksum."""
import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from ruler_bootstrap import require_linux
from seco.io import project_path, read_json, sha256_file


def download(path, sources):
    if path.is_file():
        if sha256_file(path) in {digest for _, digest in sources}:
            print(f"Verified existing {path}")
            return
        raise ValueError(f"Unexpected checksum for {path}; move it aside before retrying")
    with TemporaryDirectory(dir=path.parent) as directory:
        partial = Path(directory) / path.name
        for url, digest in sources:
            print(f"Downloading {url}", flush=True)
            result = subprocess.run(["curl", "--fail", "--location", "--retry", "2",
                                     "--connect-timeout", "15", "--speed-limit", "1024",
                                     "--speed-time", "60", "--output", str(partial), url])
            if result.returncode == 0 and sha256_file(partial) == digest:
                partial.replace(path)
                return
            print(f"Transfer or SHA-256 check failed for {url}", flush=True)
    raise RuntimeError(f"All configured sources failed for {path.name}")


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="configs/ruler_qa/sources.json")
    parser.add_argument("--output", default="artifacts/ruler_qa/raw")
    args = parser.parse_args()
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for filename, sources in read_json(project_path(args.sources)).items():
        download(output / filename, sources)


if __name__ == "__main__":
    main()
