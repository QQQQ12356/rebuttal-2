"""Small CPU tensors and randomly initialized models; no pretrained downloads."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from icae.pretraining import TASK, PretrainingCollator, split_document
from icae.stages import initialization_checkpoint
from reference_compression import CompressionMixin as OriginalCompression
from seco.compression import CompressionMixin
from seco.data import Collator
from seco.io import write_json
from seco.model import SECO, build_model, load_checkpoint_model
from seco.similarity import ChunkedCosine


class Compressor(CompressionMixin):
    def __init__(self, ratio, chunk):
        self.compress_ratio, self.cos = ratio, ChunkedCosine(chunk)


class Reference(OriginalCompression):
    def __init__(self, ratio):
        self.compress_ratio, self.cos = ratio, torch.nn.CosineSimilarity(dim=-1)


class ModelTests(unittest.TestCase):
    def test_method_weights_survive_adapter_and_merged_exports(self):
        with TemporaryDirectory() as temporary, patch.dict(os.environ, {"COMPRESSION_METHOD": ""}):
            root = Path(temporary)
            backbone = root / "llama"
            config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                 max_position_embeddings=64, pad_token_id=0, eos_token_id=2)
            LlamaForCausalLM(config).save_pretrained(backbone)
            tokens = {"<pad>": 0, "<unk>": 1, "<eos>": 2, **{str(i): i for i in range(3, 32)}}
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(tokens, unk_token="<unk>")),
                                                pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
            tokenizer.save_pretrained(backbone)
            batch = Collator(0)([
                {"input_ids": [3, 4, 5, 6, 7, 8, 9, 10], "prompt_mask": [0] * 6 + [1] * 2, "labels": [3, 4, 2]},
                {"input_ids": [5, 6, 7, 8], "prompt_mask": [0] * 3 + [1], "labels": [5, 2]},
            ])
            methods = {
                "comi": {"encoder_layers": 1, "segment_size": 4, "fusion_layers": 1},
                "icae": {"memory_slots": 2},
                "ram": {"encoder_layers": 1, "segment_batch_size": 2, "fusion_layers": 1,
                        "keep_ratio": 0.5, "contrastive_loss_weight": 1.0},
            }
            for method, settings in methods.items():
                with self.subTest(method=method):
                    config = {"method": method, "name_or_path": str(backbone), "revision": "main",
                              "dtype": "float32", "attn_implementation": "eager", "ratio_mode": "fixed",
                              "compress_ratio": 2, "lora_r": 2, "lora_alpha": 4, "lora_dropout": 0.0, **settings}
                    if method == "icae":
                        config.update(full_tail_slots=True, keep_ae_marker=True)
                        pretrain = build_model({**config, "pretraining": {"ae_weight": 1, "lm_weight": 1}})
                        document = split_document([3, 4, 5, 6, 7], bos_id=1, eos_id=2)
                        self.assertEqual(document, {"input_ids": [1, 3, 4], "ae_labels": [1, 3, 4, 2],
                                                    "labels": [5, 6, 7, 2]})
                        pretrain_batch = PretrainingCollator(0)([document])
                        pretrain(**pretrain_batch)["loss"].backward()
                        for name in ("ae_marker", "lm_marker", "memory"):
                            self.assertGreater(pretrain.method_parameters[name].grad.abs().sum().item(), 0)
                        checkpoint = root / "pretrain"
                        checkpoint.mkdir()
                        save_file({key: value for key, value in pretrain.state_dict().items()
                                   if key in pretrain.checkpoint_keys()}, checkpoint / "model.safetensors")
                        write_json(checkpoint / "experiment.json", {
                            "format": "seco-adapters-v1", "task": TASK, "tokenizer_fingerprint": "test",
                            "config": {"model": pretrain.model_config},
                        })
                        initialization = initialization_checkpoint(checkpoint, config, "test")
                    model = build_model(config)
                    if method == "icae":
                        model.load_adapters(initialization["checkpoint"])
                        torch.testing.assert_close(model.method_parameters["ae_marker"],
                                                   pretrain.method_parameters["ae_marker"])
                        self.assertFalse(model.method_parameters["ae_marker"].requires_grad)
                    loss = model(**batch)["loss"]
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for parameter in (param for param in model.method_parameters.values() if param.requires_grad):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01).step()
                    model.eval()
                    with torch.no_grad():
                        expected = model(**batch)["loss"]
                    metadata = {"format": "seco-adapters-v1", "config": {"model": config}}
                    adapters = root / method
                    adapters.mkdir()
                    save_file({key: value for key, value in model.state_dict().items()
                               if key in model.checkpoint_keys()}, adapters / "model.safetensors")
                    write_json(adapters / "experiment.json", metadata)
                    restored, _ = load_checkpoint_model(adapters)
                    merged = root / f"{method}_merged"
                    restored.eval().merge_for_inference()
                    restored.save_merged(merged, {})
                    write_json(merged / "experiment.json", metadata)
                    reloaded, _ = load_checkpoint_model(merged)
                    with torch.no_grad():
                        torch.testing.assert_close(reloaded.eval()(**batch)["loss"], expected)

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
