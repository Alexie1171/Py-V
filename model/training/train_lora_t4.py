"""
train_lora_t4.py — PY-V (model/training/)
LoRA training on a Google Colab T4, run from the Colab runner notebook.
Starts fresh from plain Phi-2 (4-bit). Data, prompt format, end-of-text
token and answer-only loss come from dataset_loader.py; hyperparameters
from the `training` section of configs/config.yaml. Batch settings are
T4-specific and kept here.

Checkpoints go to --output-dir every save_steps; a rerun resumes from the
newest one there (point it at Drive so a Colab disconnect loses little).

Usage (from repo root):
    python -m model.training.train_lora_t4 --output-dir /content/drive/MyDrive/PY-V/model_v2/lora
"""

import argparse
import dataclasses
import os
import time

import torch

from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from model.training.config_loader import CFG
from model.utils.model_loader import load_model
from model.training.dataset_loader import IGNORE_INDEX, get_tokenized_dataset

torch.backends.cuda.matmul.allow_tf32 = True

start_time = time.time()


# =========================
# LOGGING CALLBACK
# =========================
class TrainingLogger(TrainerCallback):
    def __init__(self):
        self.step_times = []
        self.last_step_time = None
        self.best_loss = float("inf")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        now = time.time()
        step = state.global_step
        elapsed = now - start_time

        if "loss" in logs:
            loss = logs["loss"]

            if self.last_step_time:
                self.step_times.append(now - self.last_step_time)
                if len(self.step_times) > 20:
                    self.step_times.pop(0)

            avg_step = sum(self.step_times) / len(self.step_times) if self.step_times else 0
            eta = avg_step * (state.max_steps - step) / args.logging_steps

            is_best = ""
            if loss < self.best_loss:
                self.best_loss = loss
                is_best = " ← best"

            print(
                f"Step {step}/{state.max_steps} | "
                f"Loss: {loss:.4f}{is_best} | "
                f"Elapsed: {elapsed/60:.1f}min | "
                f"ETA: {eta/60:.1f}min",
                flush=True,
            )

            self.last_step_time = now

        if "eval_loss" in logs:
            print(f"EVAL @ step {step} | Eval loss: {logs['eval_loss']:.4f}", flush=True)


# =========================
# MODEL
# =========================
def load_base_model():
    model, tokenizer = load_model()
    tokenizer.padding_side = "right"

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def apply_lora(model):
    t = CFG.training
    config = LoraConfig(
        r=t.lora_r,
        lora_alpha=t.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        lora_dropout=t.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


# =========================
# TRAINING
# =========================
def _length_grouping() -> dict:
    """Batch similar-length examples (less padding, faster). transformers 5
    replaced `group_by_length=True` with `train_sampling_strategy`; Colab has
    5.x, the laptop 4.57."""
    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    if "train_sampling_strategy" in fields:
        return {"train_sampling_strategy": "group_by_length"}
    return {"group_by_length": True}


def train(output_dir: str):
    t = CFG.training

    model, tokenizer = load_base_model()
    model = apply_lora(model)
    model.print_trainable_parameters()

    tokenized = get_tokenized_dataset(tokenizer)

    # Pads input_ids with the pad token and labels with IGNORE_INDEX, so padding
    # never hides the end-of-text token even though pad == eos for Phi-2
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,

        # ⚡ SPEED OPTIMIZED FOR T4 (effective batch 16)
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,

        learning_rate=t.learning_rate,
        num_train_epochs=t.epochs,
        warmup_steps=t.warmup_steps,

        logging_steps=10,

        save_steps=t.save_steps,
        eval_strategy="steps",
        eval_steps=t.eval_steps,
        save_total_limit=2,

        # PeftModel hides the base model's arguments — name the labels, or
        # no eval loss is computed
        label_names=["labels"],

        fp16=True,
        optim="paged_adamw_8bit",

        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        **_length_grouping(),

        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=data_collator,
        callbacks=[TrainingLogger()],
    )

    checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    print(f"Resuming from {checkpoint}" if checkpoint else "No checkpoint - starting fresh from plain Phi-2")

    trainer.train(resume_from_checkpoint=checkpoint)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(CFG.paths.model_output),
                        help="where checkpoints and the final adapter are saved")
    train(parser.parse_args().output_dir)
