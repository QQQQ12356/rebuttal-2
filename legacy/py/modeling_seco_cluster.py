import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from dataclasses import dataclass, field
from typing import Optional
from peft import get_peft_model
from safetensors.torch import load_file
from icecream import ic as pprint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    compress_ratio: int = field(default=16, metadata={"help": "compression ratio for the clustering"})
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_alpha: int = field(default=32, metadata={"help": "lora alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    train: bool = field(default=False, metadata={"help": "if true, the model ckpt will be initialized for training; else, it's for inference"})
    max_new_tokens: int = field(default=20, metadata={"help": "Maximum tokens generated during inference."})
    repetition_penalty: float = field(default=1.1, metadata={"help": "Repetition penalty for generation."})
    temperature: float = field(default=0.0, metadata={"help": "Sampling temperature."})
    top_p: float = field(default=0.9, metadata={"help": "Top-p for generation when sampling is enabled."})
    do_sample: bool = field(default=False, metadata={"help": "Whether to use sampling during generation."})
    
    
@dataclass
class DataArguments:
    train_file: str = field(default="/path/to/train/file", metadata={"help": "Path to the training data."})
    test_file: str = field(default="/path/to/test/file", metadata={"help": "Path to the test data."})
    debug_data: bool = field(default=False, metadata={"help": "Enable debug dataset"})
    test_sample_index: int = field(default=0, metadata={"help": "Index of the sample to test in the dataset."})
    train_samples: int = field(default=24000, metadata={"help": "Maximum number of training examples."})
    eval_samples: int = field(default=500, metadata={"help": "Maximum number of evaluation examples."})
    shuffle_eval: bool = field(default=False, metadata={"help": "Randomly sample the evaluation set."})
    eval_output_dir: str = field(default="./eval_result", metadata={"help": "Directory used to save inference results."})
    
@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=28000,
        metadata={"help": "Maximum sequence length."},
    )
    report_to: Optional[str] = field(default="wandb")
    project_name: Optional[str] = field(default="cluster")
    max_steps: int = field(default=10000, metadata={"help": "Maximum number of training steps."})
    save_strategy: Optional[str] = field(default="steps")
    save_steps: int = field(default=10000, metadata={"help": "Interval between two checkpoints saved."})
    eval_strategy: Optional[str] = field(default="steps")
    eval_steps: int = field(default=200000, metadata={"help": "Interval between two evaluations."})
    num_train_epochs: int = field(default=1)
    add_special_token_for_lm: bool = field(default=False)
    restore_from: str = field(default="", metadata={"help": "checkpoint to restore from"})
    overwrite_output_dir: bool = field(default=True)
    logging_steps: int = field(default=100)
    deepspeed: str = field(default=None)
    bf16: bool = field(default=True, metadata={"help": "Use bfloat16"})
    gradient_accumulation_steps: int = field(default=1)
    optim: str = field(default="adamw_torch")
    per_device_train_batch_size: int = field(default=1)
    lr_scheduler_type: str = field(default="cosine")
    learning_rate: float = field(default=1e-5)
    gradient_checkpointing: bool = field(default=True)
    warmup_ratio: float = field(default=0.1)
    weight_decay: float = field(default=0.01)
    seed: int = field(default=42)

def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.2f}")

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

class SECO(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        self.encoder = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            trust_remote_code=True
        )
        self.training_mode = model_args.train
        self.encoder = get_peft_model(self.encoder, lora_config)
        self.decoder = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            trust_remote_code=True
        )
        self.decoder = get_peft_model(self.decoder, lora_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.encoder.config.vocab_size
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.dim = self.encoder.config.hidden_size
        self.compress_ratio = model_args.compress_ratio
        self.cos = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        if self.training_mode:
            self.init()

    def init(self):
        print_trainable_parameters(self)
        if self.training_args.restore_from is not None and self.training_args.restore_from != "":
            print(f"Loading from the pretrained checkpoint: {self.training_args.restore_from}...")
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict, strict=False)
            print(f"Finished loading from {self.training_args.restore_from}")
        print("Enabling gradient checkpointing...")
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def tokens_to_embeddings(self, token_ids):
        base_embeds = self.encoder.get_base_model().model.embed_tokens(token_ids)
        return base_embeds

    def _compress_one(self, hidden_states, prompt_mask, attention_mask):
        valid_mask = attention_mask.bool()
        prompt_mask = prompt_mask.bool() & valid_mask
        input_mask = (~prompt_mask) & valid_mask

        prompt_tokens = hidden_states[prompt_mask]
        input_tokens = hidden_states[input_mask]

        if input_tokens.size(0) == 0:
            return hidden_states[valid_mask], torch.ones(valid_mask.sum(), device=hidden_states.device, dtype=torch.bool)

        K = max(2, math.ceil(input_tokens.size(0) / self.compress_ratio))
        if prompt_tokens.size(0) > 0:
            prompt_repr = prompt_tokens.mean(dim=0, keepdim=True)
        else:
            prompt_repr = input_tokens.mean(dim=0, keepdim=True)
        
        sim_to_prompt = self.cos(input_tokens, prompt_repr)
        K = min(K, input_tokens.size(0))
        _, center_indices = torch.topk(sim_to_prompt, K, largest=True)
        cluster_centers = input_tokens[center_indices]
        non_center_mask = torch.ones(input_tokens.size(0), dtype=torch.bool, device=hidden_states.device)
        non_center_mask[center_indices] = False
        non_center_tokens = input_tokens[non_center_mask]

        if non_center_tokens.size(0) > 0:
            # sim_to_centers = self.cos(non_center_tokens.unsqueeze(1), cluster_centers.unsqueeze(0))
            # sim_prompt_non = sim_to_prompt[non_center_mask].unsqueeze(1)
            # scores = sim_to_centers * sim_prompt_non
            # best_cluster = scores.argmax(dim=1)
            sim_to_centers = self.cos(non_center_tokens.unsqueeze(1), cluster_centers.unsqueeze(0))
            best_cluster = sim_to_centers.argmax(dim=1)
            sim_prompt_non = sim_to_prompt[non_center_mask].unsqueeze(1)
            scores = sim_to_centers * sim_prompt_non
            # pprint(best_cluster)
            # pprint(scores.shape)
            clustered = []
            for k in range(K):
                members = [cluster_centers[k:k+1]]
                weights = [sim_to_prompt[center_indices[k]].unsqueeze(0)]

                idx = (best_cluster == k).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    members.append(non_center_tokens[idx])
                    weights.append(scores[idx, k])

                vecs = torch.cat(members, dim=0)
                weight = F.softmax(torch.cat(weights), dim=0).unsqueeze(-1)
                clustered.append((vecs * weight).sum(dim=0, keepdim=True))
            compressed_input = torch.cat(clustered, dim=0)
        else:
            compressed_input = cluster_centers

        combined = torch.cat([compressed_input, prompt_tokens], dim=0)
        return combined, torch.ones(combined.size(0), device=hidden_states.device, dtype=torch.bool)

    def _compress_batch(self, hidden_states, prompt_mask, attention_mask):
        batch_compressed = []
        batch_masks = []

        for b in range(hidden_states.size(0)):
            _compressed, mask = self._compress_one(
                hidden_states[b],
                prompt_mask[b],
                attention_mask[b]
            )
            batch_compressed.append(_compressed)
            batch_masks.append(mask)

        max_len = max(x.size(0) for x in batch_compressed)
        d_model = hidden_states.size(-1)

        padded_embeds = []
        padded_masks = []

        for emb, mask in zip(batch_compressed, batch_masks):
            pad_len = max_len - emb.size(0)
            if pad_len > 0:
                emb = torch.cat([emb, emb.new_zeros(pad_len, d_model)], dim=0)
                mask = torch.cat([mask, torch.zeros(pad_len, device=hidden_states.device, dtype=torch.bool)], dim=0)
            padded_embeds.append(emb)
            padded_masks.append(mask)

        return torch.stack(padded_embeds, dim=0), torch.stack(padded_masks, dim=0)

    def forward(
        self,
        input_ids=None,
        prompt_mask=None,
        attention_mask=None,
        labels=None,
    ):
        inputs_embeds = self.tokens_to_embeddings(input_ids)

        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states[-1]
        compressed_embeds, compressed_mask = self._compress_batch(
            hidden_states, prompt_mask, attention_mask
        )
        if labels is not None:
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = self.tokenizer.pad_token_id
            label_embeds = self.decoder.get_input_embeddings()(safe_labels)

            full_embeds = torch.cat([compressed_embeds, label_embeds], dim=1)
            label_mask = (labels != -100)
            full_mask = torch.cat([compressed_mask, label_mask], dim=1)
            ignore_labels = torch.full(compressed_mask.shape, -100, device=labels.device)
            full_labels = torch.cat([ignore_labels, labels], dim=1)
            return self.decoder(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                labels=full_labels,
                return_dict=True
            )
        else:
            compressed_attention_mask = compressed_mask.long()
            outputs = self.decoder.generate(
                inputs_embeds=compressed_embeds,
                attention_mask=compressed_attention_mask,
                max_new_tokens=15,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_id,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.3,
                use_cache=True
            )
            return outputs

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.encoder.gradient_checkpointing_enable(*args, **kwargs)
        # self.decoder.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        self.encoder.gradient_checkpointing_disable(*args, **kwargs)
        # self.decoder.gradient_checkpointing_disable(*args, **kwargs)