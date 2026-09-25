"""
dataset_loader.py — PY-V
Loads the train/val JSONL datasets and turns each record into model inputs
for LoRA training:
  - prompt = the same mode template inference uses (record's metadata.task),
    via prompt_builder, so training and inference never drift apart — in V's
    template or, for chat brains, re-wrapped in the brain's own chat format
    (prompt_format "native_chat", exactly as format_for_model() does at inference)
  - answer = output + end-of-text token, so the model learns to stop
  - loss on the answer only: prompt tokens are labelled -100
  - records longer than max_seq_length are dropped, never cut (a cut answer
    has no end-of-text token and would teach the model not to stop)
All paths and lengths come from configs/config.yaml via config_loader.
"""

import json

from datasets import Dataset, DatasetDict

from model.training.config_loader import CFG
from inference.engine.prompt_builder import build_training_prompt, to_native_chat

IGNORE_INDEX = -100   # label value the loss skips


def _read_jsonl(path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode_example(tokenizer, record: dict, max_length: int, prompt_format: str = "template"):
    """Token ids + labels for one record, or None if it is too long."""
    mode   = record.get("metadata", {}).get("task", "generate")
    text   = build_training_prompt(mode, record["instruction"])
    if prompt_format == "native_chat":
        text = to_native_chat(text, tokenizer)
    # Prompt tokenized exactly like the generator does at inference; the answer
    # never gets special tokens (a brain that adds a start token would get one
    # mid-sequence otherwise)
    prompt = tokenizer(text)["input_ids"]
    answer = tokenizer(record["output"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

    if len(prompt) + len(answer) > max_length:
        return None
    return {
        "input_ids":      prompt + answer,
        "attention_mask": [1] * (len(prompt) + len(answer)),
        "labels":         [IGNORE_INDEX] * len(prompt) + answer,
    }


def get_tokenized_dataset(tokenizer, prompt_format: str = None) -> DatasetDict:
    """Train and val splits from the paths in config.yaml, ready for the Trainer.
    prompt_format defaults to config model.prompt_format."""
    prompt_format = prompt_format or CFG.model.prompt_format
    max_length    = CFG.training.max_seq_length
    splits        = {}

    for split, path in (("train", CFG.paths.dataset), ("val", CFG.paths.val_dataset)):
        records = _read_jsonl(path)
        encoded = [e for e in (encode_example(tokenizer, r, max_length, prompt_format) for r in records) if e]
        print(f"  {split}: {len(encoded)} examples from {path} "
              f"({len(records) - len(encoded)} longer than {max_length} tokens dropped)")
        splits[split] = Dataset.from_list(encoded)

    return DatasetDict(splits)
