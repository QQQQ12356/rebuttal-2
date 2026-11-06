import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional
from safetensors.torch import load_file
import time


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    compress_ratio: int = field(default=16)
    use_flash_attention_2: bool = field(default=True)
    train: bool = field(default=False)


@dataclass
class DataArguments:
    train_file: str = field(default="./data/mrqa_train.jsonl")
    test_file: str = field(default="./data/mrqa_test.jsonl")
    debug_data: bool = field(default=False)
    test_sample_index: int = field(default=0)
    eval_samples: int = field(default=100)
    shuffle_eval: bool = field(default=True)
    eval_output_dir: str = field(default="./eval_result")


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=28000)
    report_to: Optional[str] = field(default="wandb")
    project_name: Optional[str] = field(default="cluster")
    max_steps: int = field(default=10000)
    save_strategy: Optional[str] = field(default="steps")
    save_steps: int = field(default=10000)
    eval_strategy: Optional[str] = field(default="steps")
    eval_steps: int = field(default=10000)
    num_train_epochs: int = field(default=1)
    add_special_token_for_lm: bool = field(default=False)
    restore_from: str = field(default="", metadata={"help": "merge_lora.py输出的safetensors路径"})
    overwrite_output_dir: bool = field(default=True)
    logging_steps: int = field(default=200)
    deepspeed: str = field(default=None)
    bf16: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=1)
    optim: str = field(default="adamw_torch")
    per_device_train_batch_size: int = field(default=1)
    lr_scheduler_type: str = field(default="cosine")
    learning_rate: float = field(default=1e-5)
    gradient_checkpointing: bool = field(default=True)
    warmup_ratio: float = field(default=0.1)
    weight_decay: float = field(default=0.01)
    seed: int = field(default=42)
    per_device_eval_batch_size: int = field(default=1)
    dataloader_num_workers: int = field(default=0)
    dataloader_pin_memory: bool = field(default=True)
    use_torch_compile: bool = field(default=False)
    torch_compile_dynamic: bool = field(default=True)


class SECO(torch.nn.Module):
    def __init__(self, model_args, training_args, device=None):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_args.model_name_or_path
        dtype = torch.bfloat16 if training_args.bf16 else torch.float16

        self.encoder = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, trust_remote_code=True
        )
        self.decoder = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vocab_size  = self.encoder.config.vocab_size
        self.bos_id      = self.tokenizer.bos_token_id
        self.eos_id      = self.tokenizer.eos_token_id
        self.eos_token_ids = self.eos_id
        self.dim         = self.encoder.config.hidden_size
        self.compress_ratio = model_args.compress_ratio
        self.cos         = nn.CosineSimilarity(dim=-1)
        self.decoder_dtype = self.decoder.get_input_embeddings().weight.dtype

        self._load_checkpoint()
        self.eval()
        self.to(self._device)

    def _load_checkpoint(self):
        restore_from = self.training_args.restore_from
        if not restore_from:
            print("[警告] restore_from 为空，使用原始预训练权重")
            return
        if not os.path.exists(restore_from):
            raise FileNotFoundError(f"找不到权重: {restore_from}")

        print(f"加载合并后权重: {restore_from}")
        raw_sd = load_file(restore_from, device="cpu")

        missing, unexpected = self.load_state_dict(raw_sd, strict=False)

        real_missing = [k for k in missing if "lm_head" not in k]
        print(f"加载完成 | 正常缺失(tied lm_head): {len(missing)-len(real_missing)} | "
              f"真正缺失: {len(real_missing)} | 冗余: {len(unexpected)}")
        if real_missing:
            print("  真正缺失的key（前10条）:")
            for k in real_missing[:10]:
                print(f"    {k}")
        if unexpected:
            print("  冗余key（前10条）:")
            for k in unexpected[:10]:
                print(f"    {k}")

    def tokens_to_embeddings(self, token_ids):
        return self.encoder.get_input_embeddings()(token_ids)

    def _compress_one(self, hidden_states, prompt_mask, attention_mask):
        valid_mask  = attention_mask.bool()
        prompt_mask = prompt_mask.bool() & valid_mask
        input_mask  = (~prompt_mask) & valid_mask

        prompt_tokens = hidden_states[prompt_mask]
        input_tokens  = hidden_states[input_mask]

        if input_tokens.size(0) == 0:
            return (hidden_states[valid_mask],
                    torch.ones(valid_mask.sum(), device=hidden_states.device, dtype=torch.bool))

        K = max(2, math.ceil(input_tokens.size(0) / self.compress_ratio))
        prompt_repr = (prompt_tokens.mean(0, keepdim=True)
                       if prompt_tokens.size(0) > 0
                       else input_tokens.mean(0, keepdim=True))

        sim_to_prompt = self.cos(input_tokens, prompt_repr)
        K = min(K, input_tokens.size(0))
        _, center_idx = torch.topk(sim_to_prompt, K, largest=True)
        centers = input_tokens[center_idx]

        non_center_mask = torch.ones(input_tokens.size(0), dtype=torch.bool,
                                     device=hidden_states.device)
        non_center_mask[center_idx] = False
        non_centers = input_tokens[non_center_mask]

        if non_centers.size(0) > 0:
            sim_to_centers = self.cos(non_centers.unsqueeze(1), centers.unsqueeze(0))
            best_cluster   = sim_to_centers.argmax(dim=1)
            sim_prompt_non = sim_to_prompt[non_center_mask].unsqueeze(1)
            scores         = sim_to_centers * sim_prompt_non

            clustered = []
            for k in range(K):
                members = [centers[k:k+1]]
                weights = [sim_to_prompt[center_idx[k]].unsqueeze(0)]
                idx = (best_cluster == k).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    members.append(non_centers[idx])
                    weights.append(scores[idx, k])
                vecs   = torch.cat(members, dim=0)
                weight = F.softmax(torch.cat(weights), dim=0).unsqueeze(-1)
                clustered.append((vecs * weight).sum(0, keepdim=True))
            compressed = torch.cat(clustered, dim=0)
        else:
            compressed = centers

        combined = torch.cat([compressed, prompt_tokens], dim=0)
        return combined, torch.ones(combined.size(0), device=hidden_states.device, dtype=torch.bool)

    def forward(self, input_ids=None, prompt_mask=None, attention_mask=None, labels=None):
        inputs_embeds = self.tokens_to_embeddings(input_ids)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        enc_out     = self.encoder.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = enc_out.last_hidden_state

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        comp_embeds, comp_mask = self._compress_one(
            hidden_states[0], prompt_mask[0], attention_mask[0]
        )
        comp_embeds = comp_embeds.unsqueeze(0).to(self.decoder_dtype)
        comp_mask   = comp_mask.unsqueeze(0)

        out = self.decoder.generate(
            inputs_embeds=comp_embeds,
            attention_mask=comp_mask.long(),
            max_new_tokens=10,
            min_new_tokens=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_token_ids,
            do_sample=False,
            use_cache=True,
            output_attentions=False,
            output_scores=False,
            return_dict_in_generate=False,
        )

        torch.cuda.synchronize()
        t3 = time.perf_counter()
        return out, t1, t2, t3

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.encoder.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        self.encoder.gradient_checkpointing_disable(*args, **kwargs)