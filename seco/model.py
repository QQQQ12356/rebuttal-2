"""Two independent LoRA language models around the unchanged SeCo compressor."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compression import CompressionMixin
from .io import model_source, read_json, write_json
from .similarity import ChunkedCosine


class SECO(CompressionMixin, torch.nn.Module):
    accepts_loss_kwargs = False

    def __init__(self, model_config: dict, *, merged_dir: str | Path | None = None):
        super().__init__()
        self.model_config = model_config.copy()
        self.compress_ratio = model_config["compress_ratio"]
        self.cos = ChunkedCosine(model_config.get("similarity_chunk_tokens", 16))
        merged = Path(merged_dir) if merged_dir else None
        tokenizer_path = merged / "tokenizer" if merged else model_config["name_or_path"]
        common = {"local_files_only": True} if merged else {"revision": model_config["revision"]}
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True, **common)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define EOS")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {"torch_dtype": getattr(torch, model_config["dtype"]),
                  "attn_implementation": model_config["attn_implementation"], **common}
        for name in ("encoder", "decoder"):
            source = str(merged / name) if merged else model_config["name_or_path"]
            language_model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
            if not merged:
                lora = LoraConfig(r=model_config["lora_r"], lora_alpha=model_config["lora_alpha"],
                                  lora_dropout=model_config["lora_dropout"], bias="none",
                                  task_type="CAUSAL_LM", target_modules="all-linear")
                language_model = get_peft_model(language_model, lora)
            language_model.config.use_cache = False
            setattr(self, name, language_model)
        self.config = self.decoder.config
        self.merged = bool(merged)

    def load_adapters(self, checkpoint: str | Path) -> None:
        """Fail closed on missing, wrong-rank or unrelated adapter checkpoints."""
        checkpoint = Path(checkpoint)
        path = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
        state = load_file(str(path), device="cpu")
        expected = {key for key in self.state_dict() if ".lora_" in key}
        if set(state) != expected:
            raise ValueError(f"Adapter keys mismatch: missing={sorted(expected - set(state))[:5]}, "
                             f"unexpected={sorted(set(state) - expected)[:5]}")
        self.load_state_dict(state, strict=False)  # Frozen base parameters are intentionally absent.

    def merge_for_inference(self) -> None:
        if not self.merged:
            self.encoder = self.encoder.merge_and_unload(safe_merge=True)
            self.decoder = self.decoder.merge_and_unload(safe_merge=True)
            self.merged = True
        self.decoder.config.use_cache = True

    def save_merged(self, directory: Path, metadata: dict) -> None:
        if not self.merged:
            raise ValueError("Merge adapters before exporting")
        directory.mkdir(parents=True, exist_ok=False)
        for name in ("encoder", "decoder"):
            getattr(self, name).save_pretrained(directory / name, safe_serialization=True,
                                               max_shard_size="4GB")
        self.tokenizer.save_pretrained(directory / "tokenizer")
        write_json(directory / "artifact.json", {"format": "seco-merged-v1", **metadata})

    def _encode_compress(self, input_ids, prompt_mask, attention_mask):
        encoder = self.encoder.get_base_model() if hasattr(self.encoder, "get_base_model") else self.encoder
        # Calling the backbone keeps LoRA layers active while skipping the vocabulary
        # head and the allocation of every hidden layer. No encoder KV cache is needed.
        hidden = encoder.model(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False, output_hidden_states=False,
                               return_dict=True).last_hidden_state
        compressed, mask = self._compress_batch(hidden, prompt_mask, attention_mask)
        return compressed.to(self.decoder.get_input_embeddings().weight.dtype), mask

    @staticmethod
    def _left_pad(compressed, mask):
        # First answer token must follow the final *valid* compressed token during
        # both teacher forcing and batched generation. Right-padding loses this.
        lengths = mask.sum(dim=1)
        positions = torch.arange(mask.size(1), device=mask.device)[None, :]
        indices = (positions - (mask.size(1) - lengths[:, None])).clamp_min(0)
        left_mask = positions >= mask.size(1) - lengths[:, None]
        embeds = compressed.gather(1, indices[..., None].expand(-1, -1, compressed.size(-1)))
        return embeds.masked_fill(~left_mask[..., None], 0), left_mask

    def forward(self, input_ids=None, prompt_mask=None, attention_mask=None, labels=None):
        if labels is None:
            raise ValueError("Training forward requires labels; call generate_batch for inference")
        if self.training and self.model_config["ratio_mode"] == "random":
            choices = self.model_config["random_ratios"]
            # Trainer persists torch CPU RNG state; no external RNG needs restoring.
            self.compress_ratio = choices[torch.randint(len(choices), ()).item()]
        else:
            self.compress_ratio = self.model_config["compress_ratio"]
        compressed, mask = self._encode_compress(input_ids, prompt_mask, attention_mask)
        compressed, mask = self._left_pad(compressed, mask)
        safe_labels = labels.masked_fill(labels == -100, self.tokenizer.pad_token_id)
        label_embeds = self.decoder.get_input_embeddings()(safe_labels)
        full_mask = torch.cat([mask, labels != -100], dim=1)
        position_ids = (full_mask.long().cumsum(-1) - 1).clamp_min(0)
        output = self.decoder(inputs_embeds=torch.cat([compressed, label_embeds], dim=1),
                              attention_mask=full_mask, position_ids=position_ids,
                              labels=torch.cat([torch.full_like(mask, -100, dtype=torch.long), labels], dim=1),
                              use_cache=False, return_dict=True)
        # Trainer needs the loss only. Do not retain full decoder logits in callbacks.
        return {"loss": output.loss}

    @torch.inference_mode()
    def generate_batch(self, batch: dict, *, ratio: float, generation: dict):
        if not math.isfinite(ratio) or ratio < 1:
            raise ValueError("Inference compression ratio must be finite and >= 1")
        self.compress_ratio = ratio
        compressed, mask = self._encode_compress(**batch)
        compressed, mask = self._left_pad(compressed, mask)
        return self.decoder.generate(inputs_embeds=compressed, attention_mask=mask.long(),
                                     pad_token_id=self.tokenizer.pad_token_id,
                                     eos_token_id=self.tokenizer.eos_token_id,
                                     use_cache=True, **generation)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        kwargs = gradient_checkpointing_kwargs or {"use_reentrant": False}
        if kwargs.get("use_reentrant", False):
            raise ValueError("Use non-reentrant checkpointing with frozen base models")
        for model in (self.encoder, self.decoder):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)

    def gradient_checkpointing_disable(self):
        for model in (self.encoder, self.decoder):
            model.gradient_checkpointing_disable()


def load_checkpoint_model(checkpoint: str | Path, *, model_path: str | None = None,
                          attention: str | None = None) -> tuple[SECO, dict]:
    checkpoint = Path(checkpoint)
    metadata = read_json(checkpoint / "experiment.json")
    if metadata.get("format") != "seco-adapters-v1":
        raise ValueError("Unsupported experiment format")
    config = metadata["config"]["model"].copy()
    if model_path:
        config["name_or_path"] = model_source(model_path)
    if attention:
        config["attn_implementation"] = attention
    if (checkpoint / "artifact.json").is_file():
        if read_json(checkpoint / "artifact.json")["format"] != "seco-merged-v1":
            raise ValueError("Unsupported merged model format")
        model = SECO(config, merged_dir=checkpoint)
    else:
        model = SECO(config)
        model.load_adapters(checkpoint)
    return model, metadata
