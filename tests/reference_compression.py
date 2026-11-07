"""Original SeCo compression, copied verbatim; see docs/benchmark/07_architecture.md."""
import math
import torch
import torch.nn.functional as F


class CompressionMixin:
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

