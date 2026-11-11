"""ICAE learned memory tokens with an adapted encoder and frozen decoder."""
import math

import torch

from seco.prefix import ContextCompressionModel, pad_prefixes


class ICAE(ContextCompressionModel):
    method = "icae"
    frozen_decoder = True

    def __init__(self, model_config, **kwargs):
        super().__init__(model_config, **kwargs)
        dim = self.config.hidden_size
        dtype = self.encoder.get_input_embeddings().weight.dtype
        slots = model_config.get("memory_slots", 32)
        self.pretraining = model_config.get("pretraining")
        parameters = {"memory": (slots, True),
                      "lm_marker": (1, not self.pretraining or self.pretraining["lm_weight"] > 0)}
        if self.pretraining or model_config.get("keep_ae_marker"):
            parameters["ae_marker"] = (1, bool(self.pretraining and self.pretraining["ae_weight"] > 0))
        for name, (count, trainable) in parameters.items():
            value = torch.empty(count, dim, dtype=dtype).normal_(std=self.config.initializer_range)
            self.method_parameters[name] = torch.nn.Parameter(value, requires_grad=trainable)

    def compress_context(self, context_ids, prompt_ids):
        return torch.cat([self.encode_memory(context_ids), self.method_parameters["lm_marker"]])

    def encode_memory(self, context_ids):
        slots = self.method_parameters["memory"]
        chunk_size = len(slots) * int(self.compress_ratio)
        embed = self.encoder.get_input_embeddings()
        memory = []
        for chunk in context_ids.split(chunk_size):
            if len(chunk):
                count = (len(slots) if self.model_config.get("full_tail_slots")
                         else math.ceil(len(chunk) / self.compress_ratio))
                inputs = torch.cat([embed(chunk), slots[:count]], dim=0)
                if len(inputs) > self.encoder.config.max_position_embeddings:
                    raise ValueError("ICAE chunk plus memory slots exceeds the encoder position limit")
                memory.append(self.hidden(self.encoder, inputs_embeds=inputs[None])[0, -count:])
        return torch.cat(memory) if memory else slots[:0]

    def forward(self, input_ids=None, prompt_mask=None, attention_mask=None, labels=None, ae_labels=None):
        if not self.pretraining:
            return super().forward(input_ids, prompt_mask, attention_mask, labels)
        if labels is None or ae_labels is None:
            raise ValueError("ICAE pretraining requires both AE and LM targets")
        self._select_training_ratio()
        memories = [self.encode_memory(ids[mask.bool()]) for ids, mask in zip(input_ids, attention_mask)]
        losses = []
        for objective, targets in (("ae", ae_labels), ("lm", labels)):
            weight = self.pretraining[f"{objective}_weight"]
            if weight > 0:
                marker = self.method_parameters[f"{objective}_marker"]
                embeds, mask = pad_prefixes([torch.cat([memory, marker]) for memory in memories])
                losses.append(weight * self._answer_loss(embeds, mask, targets))
        return {"loss": sum(losses) / sum(self.pretraining.values())}
