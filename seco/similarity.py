"""Bound temporary cosine broadcast memory without changing SeCo's formulas."""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


class ChunkedCosine(torch.nn.Module):
    def __init__(self, chunk_tokens: int):
        super().__init__()
        self.chunk_tokens = chunk_tokens
        self.cosine = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, left, right):
        if self.chunk_tokens <= 0 or left.ndim != 3 or left.size(0) <= self.chunk_tokens:
            return self.cosine(left, right)
        parts = []
        for chunk in left.split(self.chunk_tokens, dim=0):
            if torch.is_grad_enabled() and (chunk.requires_grad or right.requires_grad):
                # Retaining every N×K×D broadcast intermediate defeats chunking
                # during training. Recompute only these cosine intermediates.
                parts.append(checkpoint(self.cosine, chunk, right, use_reentrant=False))
            else:
                parts.append(self.cosine(chunk, right))
        return torch.cat(parts, dim=0)
