"""Convert an original SeCo dual-LoRA safetensors checkpoint to the new artifact format."""
import argparse
from _bootstrap import require_linux
from seco.config import load_config
from seco.data import tokenizer_fingerprint
from seco.io import fingerprint, model_source, project_path, sha256_file, write_json


def main():
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Original unmerged dual-LoRA model.safetensors")
    parser.add_argument("--config", default="configs/train.json", help="Must match original model and LoRA settings")
    parser.add_argument("--model", help="Original base model path; do not use the default Qwen for Llama checkpoints")
    parser.add_argument("--output", required=True, help="New adapter directory; inference only, no optimizer restore")
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from seco.model import SECO

    config = load_config(project_path(args.config))
    if args.model:
        config["model"]["name_or_path"] = model_source(args.model)
    source, output = project_path(args.source), project_path(args.output)
    if output.exists():
        raise FileExistsError(output)
    model = SECO(config["model"])
    state = model.state_dict()
    expected = {key for key in state if ".lora_" in key}
    adapters = {}
    with safe_open(source, framework="pt", device="cpu") as handle:
        source_keys = set(handle.keys())
        if expected - source_keys or source_keys - set(state):
            raise ValueError("Original checkpoint keys do not match this model/LoRA configuration")
        for key in source_keys:
            value = handle.get_tensor(key)
            if value.shape != state[key].shape:
                raise ValueError(f"Wrong model or LoRA rank: {key}")
            if key in expected:
                adapters[key] = value.contiguous()
            elif not torch.equal(value, state[key]):
                raise ValueError(f"Frozen base weight differs: {key}. Refusing to discard custom base weights.")
    config["model"]["revision"] = getattr(model.config, "_commit_hash", None) or config["model"]["revision"]
    output.mkdir(parents=True)
    save_file(adapters, str(output / "model.safetensors"))
    model.tokenizer.save_pretrained(output / "tokenizer")
    write_json(output / "experiment.json", {"format": "seco-adapters-v1", "config": config,
               "config_hash": fingerprint(config), "tokenizer_fingerprint": tokenizer_fingerprint(model.tokenizer),
               "legacy_import": {"source": str(source), "sha256": sha256_file(source),
                                 "optimizer_state_imported": False}})
    print(f"Converted original dual-LoRA checkpoint: {output}")


if __name__ == "__main__":
    main()
