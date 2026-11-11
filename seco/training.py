"""Trainer integration for compact method weights and native DDP/ZeRO state."""
from __future__ import annotations

from pathlib import Path

from safetensors.torch import save_file
from transformers import Trainer

from .io import write_json


class SeCoTrainer(Trainer):
    def __init__(self, *args, experiment: dict, **kwargs):
        self.experiment = experiment
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def _save(self, output_dir=None, state_dict=None):
        # Trainer/Accelerate gathers ZeRO-3 parameters before calling this method.
        # Native DeepSpeed checkpoint state is still saved by Trainer on all ranks.
        output = Path(output_dir or self.args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        state = self.model.state_dict() if state_dict is None else state_dict
        expected = self.model.checkpoint_keys()
        adapters = {key: value.detach().cpu().contiguous().clone()
                    for key, value in state.items() if key in expected}
        if not adapters or set(adapters) != expected or any(v.numel() == 0 for v in adapters.values()):
            raise ValueError("Incomplete adapter state; verify ZeRO-3 gather-on-save is enabled")
        save_file(adapters, str(output / "model.safetensors"))
        write_json(output / "experiment.json", self.experiment)
        self.model.tokenizer.save_pretrained(output / "tokenizer")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # DDP/one-GPU adapter restoration. DeepSpeed's own loader additionally
        # restores partitioned model, optimizer, scheduler and RNG state.
        (model if model is not None else self.model).load_adapters(resume_from_checkpoint)
