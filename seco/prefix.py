"""Shared batching for methods that compress context separately from the prompt."""
import torch
from torch.nn.utils.rnn import pad_sequence

from .model import CompressionModel


def pad_prefixes(prefixes):
    lengths = torch.tensor([prefix.size(0) for prefix in prefixes], device=prefixes[0].device)
    padded = pad_sequence(prefixes, batch_first=True)
    mask = torch.arange(padded.size(1), device=padded.device) < lengths[:, None]
    return padded, mask


class ContextCompressionModel(CompressionModel):
    def _encode_compress(self, input_ids, prompt_mask, attention_mask):
        padded, mask, _ = self._prefix(input_ids, prompt_mask, attention_mask)
        return padded, mask

    def _training_prefix(self, input_ids, prompt_mask, attention_mask, labels):
        return self._prefix(input_ids, prompt_mask, attention_mask, labels)

    def compress_training_context(self, context_ids, prompt_ids, labels):
        return self.compress_context(context_ids, prompt_ids), None

    def _prefix(self, input_ids, prompt_mask, attention_mask, labels=None):
        embed = self.decoder.get_input_embeddings()
        prefixes = []
        auxiliary_losses = []
        for index, (ids, prompt, valid) in enumerate(zip(input_ids, prompt_mask.bool(), attention_mask.bool())):
            context_ids = ids[valid & ~prompt]
            prompt_ids = ids[valid & prompt]
            if labels is None:
                memory = self.compress_context(context_ids, prompt_ids)
            else:
                memory, auxiliary_loss = self.compress_training_context(context_ids, prompt_ids, labels[index])
                if auxiliary_loss is not None:
                    auxiliary_losses.append(auxiliary_loss)
            prefixes.append(torch.cat([memory.to(embed.weight.dtype), embed(prompt_ids)], dim=0))
        padded, mask = pad_prefixes(prefixes)
        return padded, mask, torch.stack(auxiliary_losses).mean() if auxiliary_losses else None

    def fuse(self, memory):
        if hasattr(self, "fusion"):
            return self.hidden(self.fusion, inputs_embeds=memory.unsqueeze(0))[0]
        return memory
