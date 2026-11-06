"""Small CPU tensors and random tiny Qwen3 models; no pretrained downloads."""
import unittest

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from reference_compression import CompressionMixin as OriginalCompression
from seco.compression import CompressionMixin
from seco.data import Collator
from seco.model import SECO
from seco.similarity import ChunkedCosine


class Compressor(CompressionMixin):
    def __init__(self, ratio, chunk):
        self.compress_ratio, self.cos = ratio, ChunkedCosine(chunk)


class Reference(OriginalCompression):
    def __init__(self, ratio):
        self.compress_ratio, self.cos = ratio, torch.nn.CosineSimilarity(dim=-1)


class ModelTests(unittest.TestCase):
    def test_original_compression_outputs_and_gradients(self):
        for ratio in (1, 4, 16, 32):
            torch.manual_seed(7)
            inputs = torch.randn(3, 19, 8, requires_grad=True)
            valid = torch.arange(19)[None, :] < torch.tensor([19, 11, 5])[:, None]
            prompt = valid & (torch.arange(19)[None, :] >= torch.tensor([15, 9, 0])[:, None])
            original, expected_mask = Reference(ratio)._compress_batch(inputs, prompt, valid)
            candidate, actual_mask = Compressor(ratio, 2)._compress_batch(inputs, prompt, valid)
            torch.testing.assert_close(candidate, original, rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.equal(actual_mask, expected_mask))
            expected_grad = torch.autograd.grad(original.square().sum(), inputs, retain_graph=True)[0]
            actual_grad = torch.autograd.grad(candidate.square().sum(), inputs)[0]
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-5)

    def test_real_eos_is_not_masked_as_padding(self):
        batch = Collator(2)([{"input_ids": [1, 2, 3], "prompt_mask": [0, 0, 1], "labels": [4, 2]},
                             {"input_ids": [2], "prompt_mask": [1], "labels": [2]}])
        self.assertTrue(batch["attention_mask"][0, 1])
        self.assertFalse(batch["attention_mask"][1, 1])
        self.assertEqual(batch["labels"][1, 1].item(), -100)

    def test_unequal_batches_left_pad_before_first_answer(self):
        embeds = torch.tensor([[[1.0], [2.0], [0.0]], [[3.0], [4.0], [5.0]]])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        padded, left_mask = SECO._left_pad(embeds, mask)
        self.assertEqual(padded[:, :, 0].tolist(), [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        self.assertEqual(left_mask.tolist(), [[False, True, True], [True, True, True]])

    def test_tiny_qwen_lora_has_encoder_and_decoder_gradients(self):
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=8, max_position_embeddings=64, pad_token_id=0, eos_token_id=2)
        model = SECO.__new__(SECO)
        torch.nn.Module.__init__(model)
        model.model_config = {"ratio_mode": "random", "random_ratios": [2, 4], "compress_ratio": 4}
        model.cos = ChunkedCosine(2)
        from types import SimpleNamespace
        model.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
        for name in ("encoder", "decoder"):
            lora = LoraConfig(r=2, lora_alpha=4, target_modules="all-linear", task_type="CAUSAL_LM")
            setattr(model, name, get_peft_model(Qwen3ForCausalLM(config), lora))
        model.config = config
        model.merged = False
        model.gradient_checkpointing_enable()
        model.train()
        batch = Collator(0)([{"input_ids": [3, 4, 5, 6, 7], "prompt_mask": [0, 0, 0, 1, 1], "labels": [8, 2]},
                             {"input_ids": [3, 4, 7], "prompt_mask": [0, 0, 1], "labels": [9, 2]}])
        loss = model(**batch)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for branch in (model.encoder, model.decoder):
            self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                for parameter in branch.parameters() if parameter.requires_grad))
        # Make the adapters nonzero; zero-initialized LoRA alone cannot test merging.
        torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05).step()
        model.eval()
        expected_loss = model(**batch)["loss"].detach()
        batch.pop("labels")
        result = model.generate_batch(batch, ratio=4, generation={"max_new_tokens": 2, "do_sample": False})
        self.assertEqual(result.size(0), 2)
        model.merge_for_inference()
        merged_result = model.generate_batch(batch, ratio=4, generation={"max_new_tokens": 2, "do_sample": False})
        self.assertTrue(torch.equal(result, merged_result))
        batch["labels"] = torch.tensor([[8, 2], [9, 2]])
        torch.testing.assert_close(model(**batch)["loss"], expected_loss, rtol=1e-5, atol=1e-6)
