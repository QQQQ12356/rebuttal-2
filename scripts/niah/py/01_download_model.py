"""Download the default Qwen3 Instruct snapshot for NIAH without the QA data workflow."""
import argparse

from niah_bootstrap import require_linux
from seco.io import project_path, write_json


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", default="artifacts/models/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    commit = HfApi().model_info(args.model, revision=args.revision).sha
    output = project_path(args.output)
    snapshot_download(args.model, revision=commit, local_dir=output,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    write_json(output / "download.json", {"repo": args.model, "revision": commit})
    print(f"NIAH model snapshot: {output} @ {commit}")


if __name__ == "__main__":
    main()
