"""Export the merged language models, tokenizer, and learned method parameters."""
import argparse
from pathlib import Path
import tempfile
import os


from seco.io import project_path, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="New pipeline final or checkpoint-N directory")
    parser.add_argument("--output", required=True, help="New merged directory")
    parser.add_argument("--model", help="Relocated base model path; must contain the same weights")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    from seco.model import load_checkpoint_model

    source, output = project_path(args.checkpoint), project_path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if (source / "artifact.json").exists():
        raise ValueError("This checkpoint is already merged")
    model, metadata = load_checkpoint_model(source, model_path=args.model)
    model.to(args.device).eval()
    model.merge_for_inference()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Leave an incomplete export under a unique temporary parent on failure.
    # The published output only appears once both HF model folders are complete.
    temporary = Path(tempfile.mkdtemp(prefix=".seco-merge-", dir=output.parent))
    payload = temporary / "model"
    model.save_merged(payload, {"source_checkpoint": str(source),
                              "source_sha256": sha256_file(source / "model.safetensors")})
    write_json(payload / "experiment.json", metadata)
    os.rename(payload, output)
    temporary.rmdir()
    print(f"Merged {model.method} model saved to {output}")
