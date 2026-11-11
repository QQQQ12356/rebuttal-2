"""COMI coarse group reallocation, fine-grained MIG pooling and memory fusion."""
import math

import torch
from torch.nn import functional as F

from seco.prefix import ContextCompressionModel


def max_redundancy(normalized, block_size=128):
    """Maximum off-diagonal cosine similarity with bounded temporary storage."""
    size = len(normalized)
    if size == 1:
        return normalized.new_zeros(1)
    columns = torch.arange(size, device=normalized.device)
    scores = []
    for start in range(0, size, block_size):
        block = normalized[start:start + block_size]
        diagonal = columns[start:start + len(block), None] == columns
        similarities = (block @ normalized.T).masked_fill(diagonal, -torch.inf)
        scores.append(similarities.max(dim=-1).values)
    return torch.cat(scores)


class COMI(ContextCompressionModel):
    method = "comi"
    uses_fusion = True

    def pool(self, hidden, query):
        size = len(hidden)
        groups = math.ceil(size / self.compress_ratio)
        normalized = F.normalize(hidden.float(), dim=-1)
        relevance = normalized @ F.normalize(query.float(), dim=-1)
        with torch.no_grad():
            # Preserve the slot budget and give every contiguous group at least one token.
            width = min(size, math.ceil(self.compress_ratio))
            nominal = F.pad(relevance, (0, (-size) % width), value=-torch.inf).view(-1, width)
            representatives = nominal.argmax(dim=1) + torch.arange(groups, device=hidden.device) * width
            coarse = relevance[representatives] - max_redundancy(normalized[representatives])
            quotas = (size - groups) * torch.softmax(-coarse, dim=0)
            counts = quotas.floor().long()
            remainder = size - groups - int(counts.sum())
            counts[(quotas - counts).argsort(descending=True, stable=True)[:remainder]] += 1
            sizes = counts + 1
            starts = sizes.cumsum(dim=0) - sizes
        pooled = []
        for lengths, offsets in zip(sizes.split(32), starts.split(32)):
            positions = torch.arange(int(lengths.max()), device=hidden.device)
            valid = positions < lengths[:, None]
            indices = (offsets[:, None] + positions).clamp_max(size - 1)
            vectors = normalized[indices]
            candidates = valid[:, None, :] & (positions[:, None] != positions)
            similarity = vectors @ vectors.transpose(-1, -2)
            redundancy = similarity.masked_fill(~candidates, -torch.inf).max(dim=-1).values
            redundancy = torch.where(lengths[:, None] > 1, redundancy, 0)
            mig = (relevance[indices] - redundancy).masked_fill(~valid, -torch.inf)
            weights = torch.softmax(mig, dim=-1).to(hidden.dtype)
            pooled.append((weights[..., None] * hidden[indices]).sum(dim=1))
        return torch.cat(pooled)

    def compress_context(self, context_ids, prompt_ids):
        memory = []
        for segment in context_ids.split(self.model_config.get("segment_size", 4096)):
            hidden = self.hidden(self.encoder, input_ids=torch.cat([segment, prompt_ids])[None])[0]
            if len(segment) == 0:
                return hidden[:0]
            query = hidden[len(segment):] if len(prompt_ids) else hidden
            memory.append(self.fuse(self.pool(hidden[:len(segment)], query.mean(dim=0))))
        return torch.cat(memory)
