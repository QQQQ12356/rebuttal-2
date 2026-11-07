"""Linux torchrun entry: one GPU, DDP, ZeRO-2 and CPU-offloaded ZeRO-2/3."""
import argparse
import copy
import os
from pathlib import Path
import platform


from seco.config import deepspeed_config, load_config
from seco.data import Collator, tokenizer_fingerprint
from seco.io import fingerprint, model_source, project_path, read_json, sha256_file, write_json


def main(default_config="configs/benchmark/train.json", expected_task="qa"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--output", required=True, help="Unique run directory; shell creates the name")
    parser.add_argument("--backend", choices=("ddp", "zero2", "zero2_cpu", "zero3_cpu"), default="zero2")
    parser.add_argument("--model", help="Override base path, typically a downloaded local model")
    parser.add_argument("--prepared-dir")
    parser.add_argument("--resume", help="Native checkpoint-N directory inside this run")
    parser.add_argument("--max-steps", type=int, help="Override cumulative optimizer-step target")
    parser.add_argument("--plan-only", action="store_true", help="Validate files/settings; no model or CUDA allocation")
    args = parser.parse_args()
    config = load_config(project_path(args.config))
    if args.model:
        config["model"]["name_or_path"] = model_source(args.model)
    if args.prepared_dir:
        config["data"]["prepared_dir"] = str(project_path(args.prepared_dir))
    if args.max_steps is not None:
        if args.max_steps <= 0:
            parser.error("--max-steps must be positive")
        config["training"]["max_steps"] = args.max_steps
    output, prepared = project_path(args.output), Path(config["data"]["prepared_dir"])
    manifest = read_json(prepared / "manifest.json")
    if manifest.get("task", "qa") != expected_task:
        raise ValueError(f"Expected {expected_task} training data; got {manifest.get('task', 'qa')}")
    if manifest["format"] != "seco-data-v1":
        raise ValueError("Unsupported prepared data format")
    if expected_task == "ruler_qa" and (manifest.get("kind") != "train" or not manifest.get("question_keys")):
        raise ValueError("Expected RULER QA training data with question identities")
    if sha256_file(prepared / "train.jsonl") != manifest["files"]["train.jsonl"]:
        raise ValueError("Training data hash mismatch")
    ratios = config["model"]["random_ratios"] if config["model"]["ratio_mode"] == "random" else [config["model"]["compress_ratio"]]
    if min(ratios) < manifest["arguments"]["min_ratio"]:
        raise ValueError("Training ratio is below preprocessing --min-ratio; preprocess again")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch = world_size * config["training"]["per_device_train_batch_size"] * config["training"]["gradient_accumulation_steps"]
    resume = project_path(args.resume) if args.resume else None
    if resume:
        if resume.parent != output or not (resume / "trainer_state.json").is_file():
            raise ValueError("--resume must name a native checkpoint-N inside --output")
        old = read_json(resume / "experiment.json")
        if config["model"]["revision"] == "main":
            config["model"]["revision"] = old["config"]["model"]["revision"]
        old_config, current_config = copy.deepcopy(old["config"]), copy.deepcopy(config)
        old_config["training"].pop("max_steps", None)
        current_config["training"].pop("max_steps", None)
        if old_config != current_config or old["backend"] != args.backend or old["world_size"] != world_size:
            raise ValueError("Resume requires the same model/data/config/backend/world size; only max_steps may change")
        if old["data_manifest_sha256"] != sha256_file(prepared / "manifest.json"):
            raise ValueError("Prepared data identity changed since checkpoint")
        if config["training"]["max_steps"] > 0 and config["training"]["max_steps"] <= read_json(resume / "trainer_state.json")["global_step"]:
            raise ValueError("Resume target already reached; increase --max-steps")
    elif output.exists():
        raise FileExistsError(f"Run directory already exists: {output}; set --resume or choose a new path")
    plan = {"config": config, "backend": args.backend, "world_size": world_size,
            "effective_batch": effective_batch, "resume": str(resume) if resume else None,
            "data_manifest_sha256": sha256_file(prepared / "manifest.json")}
    if args.plan_only:
        import json
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    import datasets
    import torch
    import transformers
    from datasets import load_dataset
    from transformers import AutoTokenizer, TrainingArguments, set_seed
    from seco.model import SECO
    from seco.training import SeCoTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training requires a visible GPU")
    # TrainingArguments registers HfTrainerDeepSpeedConfig BEFORE from_pretrained,
    # so ZeRO-3 partitions the frozen base weights during initialization too.
    training_args = TrainingArguments(output_dir=str(output), deepspeed=deepspeed_config(args.backend),
                                      bf16=config["model"]["dtype"] == "bfloat16",
                                      fp16=config["model"]["dtype"] == "float16",
                                      remove_unused_columns=False, ddp_find_unused_parameters=False,
                                      gradient_checkpointing_kwargs={"use_reentrant": False},
                                      save_safetensors=True, save_only_model=False, **config["training"])
    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name_or_path"],
                                              revision=config["model"]["revision"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer_fingerprint(tokenizer) != manifest["tokenizer_fingerprint"]:
        raise ValueError("Tokenizer differs from preprocessing; regenerate prepared data")
    dataset = load_dataset("json", data_files=str(prepared / "train.jsonl"), split="train")
    with training_args.main_process_first(desc="load tokenized training rows"):
        dataset = dataset.select_columns(["input_ids", "prompt_mask", "labels"])
    model = SECO(config["model"])
    # Resolve a remote 'main' to the actual loaded commit in saved metadata.
    config["model"]["revision"] = getattr(model.config, "_commit_hash", None) or config["model"]["revision"]
    if model.config.max_position_embeddings < manifest["arguments"]["max_input_tokens"]:
        raise ValueError("Prepared encoder length exceeds the model position limit")
    if model.config.max_position_embeddings < manifest["arguments"]["max_decoder_tokens"]:
        raise ValueError("Prepared decoder length exceeds the model position limit")
    experiment = {**plan, "config": config, "task": expected_task, "format": "seco-adapters-v1", "config_hash": fingerprint(config),
                  "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
                  "versions": {"python": platform.python_version(), "torch": torch.__version__,
                               "transformers": transformers.__version__, "datasets": datasets.__version__}}
    if expected_task == "niah":
        pool = manifest.get("training_haystack") or {}
        experiment["niah_training_identity"] = {
            "document_hashes": [row["text_sha256"] for row in pool.get("documents", [])],
            "answer_hashes": manifest.get("answer_hashes", [])}
    if expected_task == "ruler_qa":
        experiment["ruler_qa_training_question_keys"] = manifest["question_keys"]
    trainer = SeCoTrainer(model=model, args=training_args, train_dataset=dataset,
                          data_collator=Collator(tokenizer.pad_token_id), experiment=experiment)
    if trainer.is_world_process_zero():
        write_json(output / "experiment.json", experiment)
        write_json(output / "data_manifest.json", manifest)
        print(f"Run: {output}\nEffective batch: {effective_batch}")
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    # All ranks participate: necessary for ZeRO-3 gathering.
    trainer.save_model(str(output / "final"))
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
