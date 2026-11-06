"""Print Linux dependency/GPU information before experiments."""
from _bootstrap import require_linux


def main():
    require_linux()
    import importlib.metadata
    import json
    import platform
    import torch

    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "datasets", "deepspeed", "openpyxl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    print(json.dumps({"python": platform.python_version(), "packages": versions,
                      "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                      "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]},
                     ensure_ascii=False, indent=2))
    if not torch.cuda.is_available():
        raise SystemExit("No visible CUDA GPU. Data preprocessing and report export still work on Linux CPU.")


if __name__ == "__main__":
    main()
