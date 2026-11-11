"""RAM query-based close reading and parallel segment skimming."""
import math

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from seco.prefix import ContextCompressionModel


class RAM(ContextCompressionModel):
    method = "ram"
    uses_fusion = True

    def __init__(self, model_config, **kwargs):
        super().__init__(model_config, **kwargs)
        dtype = self.encoder.get_input_embeddings().weight.dtype
        alignment = torch.nn.Linear(self.config.hidden_size, self.config.hidden_size, dtype=dtype)
        self.method_parameters.update({"weight": alignment.weight, "bias": alignment.bias})

    def compress_context(self, context_ids, prompt_ids):
        return self._compress(context_ids, prompt_ids)[0]

    def compress_training_context(self, context_ids, prompt_ids, labels):
        memory, scores = self._compress(context_ids, prompt_ids)
        weight = self.model_config.get("contrastive_loss_weight", 0)
        if not self.training or not weight or scores is None:
            return memory, None
        # The prepared training labels end with EOS and mask all padding.
        answer = labels[labels != -100][:-1]
        if not 0 < len(answer) <= len(context_ids):
            return memory, None
        matches = (context_ids.unfold(0, len(answer), 1) == answer).all(dim=-1).nonzero().flatten()
        if not len(matches):
            return memory, None
        width = min(len(context_ids), math.ceil(self.compress_ratio))
        starts = matches // width
        ends = (matches + len(answer) - 1) // width
        boundaries = torch.zeros(len(scores) + 1, dtype=torch.long, device=scores.device)
        ones = torch.ones_like(starts)
        boundaries.scatter_add_(0, starts, ones)
        boundaries.scatter_add_(0, ends + 1, -ones)
        positive = boundaries.cumsum(dim=0)[:-1] > 0
        logits = scores / self.model_config.get("contrastive_temperature", 0.1)
        loss = logits.logsumexp(dim=0) - logits[positive].logsumexp(dim=0)
        return memory, weight * loss

    def _compress(self, context_ids, prompt_ids):
        if not len(context_ids):
            return self.decoder.get_input_embeddings()(context_ids), None
        width = min(len(context_ids), math.ceil(self.compress_ratio))
        segments = context_ids.split(width)
        query_ids = prompt_ids if len(prompt_ids) else segments[0]
        query = self.hidden(self.encoder, input_ids=query_ids[None])[0].float().mean(dim=0)
        batch_size = self.model_config.get("segment_batch_size", 256)
        representations, pooled = [], []
        for start in range(0, len(segments), batch_size):
            batch = segments[start:start + batch_size]
            ids = pad_sequence(batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            lengths = torch.tensor([len(segment) for segment in batch], device=ids.device)
            mask = torch.arange(ids.size(1), device=ids.device) < lengths[:, None]
            hidden = self.hidden(self.encoder, input_ids=ids, attention_mask=mask)
            means = hidden.float().masked_fill(~mask[..., None], 0).sum(dim=1) / lengths[:, None]
            representations.append(F.cosine_similarity(means, query, dim=-1))
            relevance = F.cosine_similarity(hidden.float(), query, dim=-1).masked_fill(~mask, -torch.inf)
            weights = torch.softmax(relevance, dim=-1).to(hidden.dtype)
            pooled.append((weights[..., None] * hidden).sum(dim=1))
        scores = torch.cat(representations)
        keep_count = int(len(segments) * self.model_config.get("keep_ratio", 0.05))
        keep_count = min(keep_count, len(segments) - 1)
        keep = scores.argsort(descending=True, stable=True)[:keep_count]
        skim = torch.ones(len(segments), device=scores.device, dtype=torch.bool)
        skim[keep] = False
        skimming_memory = F.linear(self.fuse(torch.cat(pooled)[skim]),
                                  self.method_parameters["weight"], self.method_parameters["bias"])
        embed = self.decoder.get_input_embeddings()
        # Insert skim vectors and untouched close-reading tokens in source order.
        lengths = torch.full_like(skim, width, dtype=torch.long)
        lengths[-1] = len(segments[-1])
        output_lengths = torch.where(skim, 1, lengths)
        offsets = output_lengths.cumsum(dim=0) - output_lengths
        keep = keep.sort().values
        positions = torch.arange(width, device=scores.device)
        valid = positions < lengths[keep, None]
        token_indices = (keep[:, None] * width + positions)[valid]
        output_indices = (offsets[keep, None] + positions)[valid]
        memory = skimming_memory.new_zeros((len(skimming_memory) + len(token_indices), skimming_memory.size(-1)))
        memory = memory.index_copy(0, offsets[skim], skimming_memory)
        memory = memory.index_copy(0, output_indices, embed(context_ids[token_indices]).to(memory.dtype))
        return memory, scores
