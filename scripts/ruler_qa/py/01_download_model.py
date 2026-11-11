"""Download the Llama Instruct snapshot and record its Hub revision."""
import argparse

from ruler_bootstrap import require_linux
from seco.experiment import MODEL_ID, default_model_path
from seco.io import project_path, write_json


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", default=default_model_path())
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    commit = HfApi().model_info(args.model, revision=args.revision).sha
    output = project_path(args.output)
    snapshot_download(args.model, revision=commit, local_dir=output,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    write_json(output / "download.json", {"repo": args.model, "revision": commit})
    print(f"RULER QA model: {output} @ {commit}")


if __name__ == "__main__":
    main()
