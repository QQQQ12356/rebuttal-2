"""Method and backbone defaults shared by the experiment entry points."""
import math
import os


METHODS = ("seco", "comi", "icae", "ram")
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
BACKBONE = "Llama-3.2-1B-Instruct"


def artifact_path(task, relative=""):
    method = os.environ.get("COMPRESSION_METHOD", "seco")
    backbone = os.environ.get("BACKBONE", BACKBONE)
    root = os.environ.get(f"{task.upper()}_ROOT")
    if not root:
        root = f"{os.environ.get('ARTIFACT_ROOT', 'artifacts')}/{method}/{backbone}/{task}"
    return f"{root}/{relative}".rstrip("/")


def default_model_path():
    method = os.environ.get("COMPRESSION_METHOD", "seco")
    backbone = os.environ.get("BACKBONE", BACKBONE)
    return f"{os.environ.get('ARTIFACT_ROOT', 'artifacts')}/{method}/{backbone}/model"


def training_config(task):
    method = os.environ.get("COMPRESSION_METHOD", "seco")
    return f"configs/{task}/{method}/train.json"


def check_method(model_config):
    expected = os.environ.get("COMPRESSION_METHOD")
    method = model_config.get("method", "seco")
    if method not in METHODS:
        raise ValueError(f"Unknown compression method: {method}")
    if expected and method != expected:
        raise ValueError(f"This entry uses {expected}, but the config or checkpoint uses {method}")


def validate_ratios(model_config, ratios):
    if model_config.get("method", "seco") != "seco" and any(int(ratio) != ratio for ratio in ratios):
        raise ValueError("COMI, ICAE and RAM require integer compression ratios")


def memory_token_bound(model_config, tokens, ratio):
    """Decoder memory length bound, excluding the prompt."""
    method = model_config.get("method", "seco")
    if method == "comi":
        segment = model_config["segment_size"]
        full, tail = divmod(tokens, segment)
        return full * math.ceil(segment / ratio) + math.ceil(tail / ratio)
    if method == "icae":
        if model_config.get("full_tail_slots"):
            slots = model_config["memory_slots"]
            return math.ceil(tokens / (slots * ratio)) * slots + 1
        return math.ceil(tokens / ratio) + 1
    if method == "ram":
        segments = math.ceil(tokens / ratio)
        keep = int(segments * model_config["keep_ratio"])
        return min(tokens, segments + keep * (int(ratio) - 1))
    return min(tokens, max(2, math.ceil(tokens / ratio)))
