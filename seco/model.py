"""Shared language model lifecycle and the SeCo compressor."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .compression import CompressionMixin
from .experiment import check_method, validate_ratios
from .io import model_source, read_json, write_json
from .similarity import ChunkedCosine


class CompressionModel(torch.nn.Module):
    accepts_loss_kwargs = False
    method = "seco"
    frozen_decoder = False
    uses_fusion = False
    model_parts = ("encoder", "decoder")

    def __init__(self, model_config: dict, *, merged_dir: str | Path | None = None):
        super().__init__()
        self.model_config = model_config.copy()
        self.compress_ratio = model_config["compress_ratio"]
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
        if self.uses_fusion and model_config.get("fusion_layers", 1):
            self.model_parts += ("fusion",)
        for name in self.model_parts:
            source = str(merged / name) if merged else model_config["name_or_path"]
            depth = {"encoder": model_config.get("encoder_layers"),
                     "fusion": model_config.get("fusion_layers", 1)}.get(name)
            part_kwargs = kwargs.copy()
            if depth and not merged:
                config = AutoConfig.from_pretrained(source, **common)
                if depth > config.num_hidden_layers:
                    raise ValueError(f"{name} depth exceeds backbone depth")
                config.num_hidden_layers = depth
                part_kwargs["config"] = config
            language_model = AutoModelForCausalLM.from_pretrained(source, **part_kwargs)
            if name == "decoder" and self.frozen_decoder:
                language_model.requires_grad_(False)
            elif not merged:
                lora = LoraConfig(r=model_config["lora_r"], lora_alpha=model_config["lora_alpha"],
                                  lora_dropout=model_config["lora_dropout"], bias="none",
                                  task_type="CAUSAL_LM",
                                  target_modules=model_config.get("lora_target_modules", "all-linear"))
                language_model = get_peft_model(language_model, lora)
            language_model.config.use_cache = False
            setattr(self, name, language_model)
        self.config = self.decoder.config
        self.merged = bool(merged)
        self.method_parameters = torch.nn.ParameterDict()

    def checkpoint_keys(self):
        return {key for key, _ in self.named_parameters()
                if ".lora_" in key or key.startswith("method_parameters.")}

    def load_method_parameters(self, path):
        expected = {f"method_parameters.{key}" for key in self.method_parameters}
        if expected:
            state = load_file(str(path), device="cpu")
            if set(state) != expected:
                raise ValueError("Method parameter keys mismatch")
            self.load_state_dict(state, strict=False)

    def load_adapters(self, checkpoint: str | Path) -> None:
        """Fail closed on missing, wrong-rank or unrelated adapter checkpoints."""
        checkpoint = Path(checkpoint)
        path = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
        state = load_file(str(path), device="cpu")
        expected = self.checkpoint_keys()
        if set(state) != expected:
            raise ValueError(f"Adapter keys mismatch: missing={sorted(expected - set(state))[:5]}, "
                             f"unexpected={sorted(set(state) - expected)[:5]}")
        self.load_state_dict(state, strict=False)  # Frozen base parameters are intentionally absent.

    def merge_for_inference(self) -> None:
        if not self.merged:
            for name in self.model_parts:
                part = getattr(self, name)
                if hasattr(part, "merge_and_unload"):
                    setattr(self, name, part.merge_and_unload(safe_merge=True))
            self.merged = True
        self.decoder.config.use_cache = True

    def save_merged(self, directory: Path, metadata: dict) -> None:
        if not self.merged:
            raise ValueError("Merge adapters before exporting")
        directory.mkdir(parents=True, exist_ok=False)
        for name in self.model_parts:
            getattr(self, name).save_pretrained(directory / name, safe_serialization=True,
                                               max_shard_size="4GB")
        self.tokenizer.save_pretrained(directory / "tokenizer")
        if self.method_parameters:
            save_file({f"method_parameters.{key}": value.detach().cpu().contiguous()
                       for key, value in self.method_parameters.items()}, str(directory / "method.safetensors"))
        write_json(directory / "artifact.json", {"format": "seco-merged-v1", **metadata})

    @staticmethod
    def hidden(model, **inputs):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        return base.model(**inputs, use_cache=False, output_hidden_states=False,
                          return_dict=True).last_hidden_state

    def _training_prefix(self, input_ids, prompt_mask, attention_mask, labels):
        compressed, mask = self._encode_compress(input_ids, prompt_mask, attention_mask)
        return compressed, mask, None

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
        self._select_training_ratio()
        compressed, mask, auxiliary_loss = self._training_prefix(input_ids, prompt_mask, attention_mask, labels)
        loss = self._answer_loss(compressed, mask, labels)
        return {"loss": loss if auxiliary_loss is None else loss + auxiliary_loss}

    def _select_training_ratio(self):
        if self.training and self.model_config["ratio_mode"] == "random":
            choices = self.model_config["random_ratios"]
            # Trainer persists torch CPU RNG state; no external RNG needs restoring.
            self.compress_ratio = choices[torch.randint(len(choices), ()).item()]
        else:
            self.compress_ratio = self.model_config["compress_ratio"]

    def _answer_loss(self, compressed, mask, labels):
        if compressed.size(1) + labels.size(1) > self.config.max_position_embeddings:
            raise ValueError("Compressed prefix and labels exceed the decoder position limit")
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
        return output.loss

    @torch.inference_mode()
    def generate_batch(self, batch: dict, *, ratio: float, generation: dict, group_stats: list | None = None):
        """Generate tokens and optionally append one SECO grouping record per input."""
        if not math.isfinite(ratio) or ratio < 1:
            raise ValueError("Inference compression ratio must be finite and >= 1")
        validate_ratios(self.model_config, [ratio])
        self.compress_ratio = ratio
        compression_options = {} if group_stats is None else {"group_stats": group_stats}
        compressed, mask = self._encode_compress(**batch, **compression_options)
        if compressed.size(1) + generation.get("max_new_tokens", 0) > self.config.max_position_embeddings:
            raise ValueError("Compressed prefix and generation exceed the decoder position limit")
        compressed, mask = self._left_pad(compressed, mask)
        return self.decoder.generate(inputs_embeds=compressed, attention_mask=mask.long(),
                                     pad_token_id=self.tokenizer.pad_token_id,
                                     eos_token_id=self.tokenizer.eos_token_id,
                                     use_cache=True, **generation)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        kwargs = gradient_checkpointing_kwargs or {"use_reentrant": False}
        if kwargs.get("use_reentrant", False):
            raise ValueError("Use non-reentrant checkpointing with frozen base models")
        for model in (getattr(self, name) for name in self.model_parts):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)

    def gradient_checkpointing_disable(self):
        for model in (getattr(self, name) for name in self.model_parts):
            model.gradient_checkpointing_disable()


class SECO(CompressionMixin, CompressionModel):
    def __init__(self, model_config, **kwargs):
        super().__init__(model_config, **kwargs)
        self.cos = ChunkedCosine(model_config.get("similarity_chunk_tokens", 16))

    def _encode_compress(self, input_ids, prompt_mask, attention_mask, *, group_stats=None):
        hidden = self.hidden(self.encoder, input_ids=input_ids, attention_mask=attention_mask)
        compressed, mask = self._compress_batch(hidden, prompt_mask, attention_mask, group_stats=group_stats)
        return compressed.to(self.decoder.get_input_embeddings().weight.dtype), mask


def build_model(config, **kwargs):
    method = config.get("method", "seco")
    if method == "seco":
        return SECO(config, **kwargs)
    if method == "comi":
        from comi.model import COMI as method_class
    elif method == "icae":
        from icae.model import ICAE as method_class
    elif method == "ram":
        from ram.model import RAM as method_class
    else:
        raise ValueError(f"Unknown method: {method}")
    return method_class(config, **kwargs)


def load_checkpoint_model(checkpoint: str | Path, *, model_path: str | None = None,
                          attention: str | None = None, dtype: str | None = None) -> tuple[CompressionModel, dict]:
    checkpoint = Path(checkpoint)
    metadata = read_json(checkpoint / "experiment.json")
    if metadata.get("format") != "seco-adapters-v1":
        raise ValueError("Unsupported experiment format")
    config = metadata["config"]["model"].copy()
    check_method(config)
    if model_path:
        config["name_or_path"] = model_source(model_path)
    if attention:
        config["attn_implementation"] = attention
    if dtype:
        config["dtype"] = dtype
    if (checkpoint / "artifact.json").is_file():
        if read_json(checkpoint / "artifact.json")["format"] != "seco-merged-v1":
            raise ValueError("Unsupported merged model format")
        model = build_model(config, merged_dir=checkpoint)
        model.load_method_parameters(checkpoint / "method.safetensors")
    else:
        model = build_model(config)
        model.load_adapters(checkpoint)
    return model, metadata
