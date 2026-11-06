# training_utils.py

import os
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.trainer_utils import get_last_checkpoint
from transformers import Trainer
from datasets import Dataset


class InstructFTTokenizeFunction:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        all_input_ids = []
        all_prompt_masks = []
        all_labels = []
        eos_token = self.tokenizer.eos_token
        for i, p, a in zip(examples["input"], examples["prompt"], examples["answer"]):
            input_ids = self.tokenizer.encode(f"{i}", add_special_tokens=False)
            prompt_ids = self.tokenizer.encode(f"{p}", add_special_tokens=False)
            if isinstance(a, list):
                answer_ids = self.tokenizer.encode(f"{a[0]}{eos_token}", add_special_tokens=True)
            else:
                answer_ids = self.tokenizer.encode(f"{a}{eos_token}", add_special_tokens=True)
            combined_input = input_ids + prompt_ids
            p_mask = [0] * len(input_ids) + [1] * len(prompt_ids)
            all_input_ids.append(torch.tensor(combined_input[:]))
            all_prompt_masks.append(torch.tensor(p_mask[:]))
            all_labels.append(torch.tensor(answer_ids))

        return {
            "input_ids": all_input_ids,
            "prompt_mask": all_prompt_masks,
            "labels": all_labels
        }


class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]
        prompt_mask = [torch.tensor(f["prompt_mask"]) for f in features]

        batch_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        batch_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        batch_prompt_mask = pad_sequence(prompt_mask, batch_first=True, padding_value=0)
        attention_mask = (batch_input_ids != self.pad_token_id).bool()

        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "prompt_mask": batch_prompt_mask,
            "attention_mask": attention_mask,
        }


def train_model(model, train_dataset, eval_dataset, training_args, tokenizer):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(training_args)

    data_collator = DataCollatorForDynamicPadding(tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        print(f"Loaded from the checkpoint: {checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)