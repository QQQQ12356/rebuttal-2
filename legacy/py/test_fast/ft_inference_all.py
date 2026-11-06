import json
import os
import random
from collections import defaultdict

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import HfArgumentParser

from eval_utils import compute_exact, compute_f1, normalize_answer
from modeling_seco_cluster_no_lora import SECO, DataArguments, ModelArguments, TrainingArguments


ood_subsets = {"DROP", "BioASQ", "DuoRC.ParaphraseRC", "TextbookQA", "RelationExtraction", "RACE"}
id_subsets = {"SQuAD", "NewsQA", "TriviaQA-web", "SearchQA", "HotpotQA", "NaturalQuestionsShort"}


class JsonlDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return distributed, rank, world_size, local_rank, device


def barrier_if_needed(distributed):
    if distributed and dist.is_initialized():
        dist.barrier()


def load_checkpoint(model, restore_from):
    if not restore_from:
        return

    print(f"Loading checkpoint: {restore_from}")
    if restore_from.endswith(".safetensors"):
        state_dict = load_file(restore_from)
    else:
        state_dict = torch.load(restore_from, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=False)


def load_eval_dataset(test_file, max_samples, shuffle_eval, seed):
    with open(test_file, "r", encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle]

    if shuffle_eval:
        rng = random.Random(seed)
        rng.shuffle(data)

    if max_samples > 0:
        data = data[: min(max_samples, len(data))]
    return data


def collate_batch(features, tokenizer, max_length):
    contexts = [item["input"] for item in features]
    questions = [item["prompt"] for item in features]
    
    context_tokens = tokenizer(contexts, add_special_tokens=False).input_ids
    context_lens = [len(ids) for ids in context_tokens]
    
    full_texts = [ctx + question for ctx, question in zip(contexts, questions)]

    encoded = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    prompt_mask = torch.zeros_like(encoded["input_ids"])
    for idx, ctx_len in enumerate(context_lens):
        ctx_len = min(ctx_len, max_length)
        prompt_mask[idx, ctx_len:] = 1

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "prompt_mask": prompt_mask,
        "raw_batch": features,
    }


def compute_metrics(results):
    all_em = []
    all_f1 = []
    subset_stats = defaultdict(lambda: {"em_scores": [], "f1_scores": [], "latencies": [], "count": 0})
    ood_em_scores = []
    ood_f1_scores = []
    id_em_scores = []
    id_f1_scores = []
    sample_details = []

    for item in results:
        prediction = item["model_output"]
        ground_truth = item["ground_truth"]
        subset = item["subset"]
        
        latency_info = {
            "t1": item.get("t1", 0),
            "t2": item.get("t2", 0),
            "t3": item.get("t3", 0),
            "latency_1_2": item.get("latency_1_2", 0),
            "latency_2_3": item.get("latency_2_3", 0),
            "latency_1_3": item.get("latency_1_3", 0)
        }

        if isinstance(ground_truth, list):
            em_score = max(
                compute_exact(normalize_answer(gt), normalize_answer(prediction))
                for gt in ground_truth
            )
            f1_score = max(
                compute_f1(normalize_answer(prediction), normalize_answer(gt))
                for gt in ground_truth
            )
        else:
            em_score = compute_exact(normalize_answer(ground_truth), normalize_answer(prediction))
            f1_score = compute_f1(normalize_answer(prediction), normalize_answer(ground_truth))

        all_em.append(em_score)
        all_f1.append(f1_score)
        subset_stats[subset]["em_scores"].append(em_score)
        subset_stats[subset]["f1_scores"].append(f1_score)
        subset_stats[subset]["latencies"].append(latency_info)
        subset_stats[subset]["count"] += 1

        if subset in ood_subsets:
            ood_em_scores.append(em_score)
            ood_f1_scores.append(f1_score)
        elif subset in id_subsets:
            id_em_scores.append(em_score)
            id_f1_scores.append(f1_score)

        sample_details.append(
            {
                "subset": subset,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "em": em_score,
                "f1": f1_score,
                "latencies": latency_info,
            }
        )

    subset_summary = {}
    for subset, stats in subset_stats.items():
        em_scores = stats["em_scores"]
        f1_scores = stats["f1_scores"]
        latencies = stats["latencies"]
        
        avg_latencies = {
            "avg_latency_1_2": sum(l["latency_1_2"] for l in latencies) / len(latencies) if latencies else 0,
            "avg_latency_2_3": sum(l["latency_2_3"] for l in latencies) / len(latencies) if latencies else 0,
            "avg_latency_1_3": sum(l["latency_1_3"] for l in latencies) / len(latencies) if latencies else 0,
        }
        
        subset_summary[subset] = {
            "count": stats["count"],
            "avg_em": (sum(em_scores) / len(em_scores)) if em_scores else 0,
            "avg_f1": (sum(f1_scores) / len(f1_scores)) if f1_scores else 0,
            "avg_latencies": avg_latencies,
        }

    return {
        "summary": {
            "total_samples": len(all_em),
            "avg_em": (sum(all_em) / len(all_em)) if all_em else 0,
            "avg_f1": (sum(all_f1) / len(all_f1)) if all_f1 else 0,
            "ood_em": (sum(ood_em_scores) / len(ood_em_scores)) if ood_em_scores else 0,
            "ood_f1": (sum(ood_f1_scores) / len(ood_f1_scores)) if ood_f1_scores else 0,
            "id_em": (sum(id_em_scores) / len(id_em_scores)) if id_em_scores else 0,
            "id_f1": (sum(id_f1_scores) / len(id_f1_scores)) if id_f1_scores else 0,
        },
        "subset_summary": subset_summary,
        # "details": sample_details,
    }


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.train = False

    distributed, rank, world_size, local_rank, device = setup_distributed()
    random.seed(training_args.seed + rank)
    torch.manual_seed(training_args.seed + rank)

    model = SECO(model_args, training_args)
    model.to(device)
    model.eval()

    tokenizer = model.tokenizer
    dataset = load_eval_dataset(
        data_args.test_file,
        data_args.eval_samples,
        data_args.shuffle_eval,
        training_args.seed,
    )

    if rank == 0:
        print(f"Loaded {len(dataset)} samples for inference.")
        print(f"World size: {world_size}, per-device batch size: {training_args.per_device_eval_batch_size}")

    shard = dataset[rank::world_size]
    dataloader = DataLoader(
        JsonlDataset(shard),
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, training_args.model_max_length),
    )

    model_tag = os.path.basename(model_args.model_name_or_path.rstrip("/"))
    run_name = (
        f"cr{model_args.compress_ratio}"
        f"_samples{data_args.eval_samples}"
    )
    output_dir = os.path.join(data_args.eval_output_dir, model_tag, run_name)
    os.makedirs(output_dir, exist_ok=True)

    part_path = os.path.join(output_dir, f"predictions.rank{rank}.jsonl")
    results = []

    # Optional Warm-up for torch.compile
    warmup_steps = 50 # Run 50 batches to ensure full compilation
    if rank == 0:
        print(f"Starting warm-up phase ({warmup_steps} batches to trigger torch.compile)...")

    with torch.inference_mode():
        warmup_count = 0
        for i, batch in enumerate(dataloader):
            if warmup_count >= warmup_steps:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            prompt_mask = batch["prompt_mask"].to(device, non_blocking=True)
            _ = model(
                input_ids=input_ids,
                prompt_mask=prompt_mask,
                attention_mask=attention_mask,
                labels=None,
            )
            warmup_count += 1
        
        barrier_if_needed(distributed)
        if rank == 0:
            print("Warm-up completed. Starting timed inference on all samples.")

        iterator = tqdm(dataloader, desc=f"Inference rank {rank}", disable=rank != 0)
        for i, batch in enumerate(iterator):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            prompt_mask = batch["prompt_mask"].to(device, non_blocking=True)

            generated_ids, t1, t2, t3  = model(
                input_ids=input_ids,
                prompt_mask=prompt_mask,
                attention_mask=attention_mask,
                labels=None,
            )

            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for sample, prediction in zip(batch["raw_batch"], responses):
                results.append(
                    {
                        "subset": sample.get("subset", ""),
                        "context": sample["input"],
                        "question": sample["prompt"],
                        "model_output": prediction.strip(),
                        "ground_truth": sample.get("answer", ""),
                        "t1": t1,
                        "t2": t2,
                        "t3": t3,
                        "latency_1_2": t2 - t1,
                        "latency_2_3": t3 - t2,
                        "latency_1_3": t3 - t1,
                    }
                )

    with open(part_path, "w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    barrier_if_needed(distributed)

    if rank == 0:
        merged_results = []
        for shard_rank in range(world_size):
            shard_path = os.path.join(output_dir, f"predictions.rank{shard_rank}.jsonl")
            with open(shard_path, "r", encoding="utf-8") as handle:
                merged_results.extend(json.loads(line) for line in handle)

        metrics = compute_metrics(merged_results)
        merged_path = os.path.join(output_dir, "mrqa_inference_results.jsonl")
        metrics_path = os.path.join(output_dir, "mrqa_inference_metrics.json")

        with open(merged_path, "w", encoding="utf-8") as handle:
            for item in merged_results:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)

        print(f"Inference saved to: {merged_path}")
        print(f"Metrics saved to: {metrics_path}")
        print(f"Final EM: {metrics['summary']['avg_em']:.4f}")
        print(f"Final F1: {metrics['summary']['avg_f1']:.4f}")
        print(f"OOD EM: {metrics['summary']['ood_em']:.4f}")
        print(f"OOD F1: {metrics['summary']['ood_f1']:.4f}")
        print(f"ID EM: {metrics['summary']['id_em']:.4f}")
        print(f"ID F1: {metrics['summary']['id_f1']:.4f}")

    barrier_if_needed(distributed)
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
