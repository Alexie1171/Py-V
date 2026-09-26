"""
train_lora_t4.py — PY-V (model/training/)
LoRA training on one T4 (Kaggle or Colab), started by scripts/gpu_pipeline.py.
Starts fresh from the plain base model in config (4-bit). Data, prompt format, end-of-text
token and answer-only loss come from dataset_loader.py; hyperparameters
from the `training` section of configs/config.yaml. Batch size and gradient
checkpointing are measured on the GPU at start (pick_batch_setup): the
fastest setup whose worst-case batch fits, effective batch always 16.

Checkpoints go to --output-dir every save_steps; a rerun resumes from the
newest one there (point it at saved storage so a stopped run loses little).
Uses one GPU only: on a 2-GPU machine it takes GPU 0 unless CUDA_VISIBLE_DEVICES
says otherwise (the pipeline gives each training its own GPU).
At the end the adapter folder also gets v_adapter.json (brain, prompt format,
dataset, GPU setup, final losses — the inference loader reads the prompt
format from it) and training_log.json (every logged loss / eval loss).

Usage (from repo root):
    python -m model.training.train_lora_t4 --output-dir /content/drive/MyDrive/PY-V/model_granite/lora
    python -m model.training.train_lora_t4 --model ibm-granite/granite-4.2-3b --prompt-format native_chat \
        --output-dir /content/drive/MyDrive/PY-V/model_granite-4.2-3b/lora
"""

import argparse
import dataclasses
import datetime
import json
import os
import time

# One GPU per training: with two visible GPUs (Kaggle T4 x2) the Trainer would
# wrap the 4-bit model in DataParallel. Set before torch first touches CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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
from model.utils.model_loader import ADAPTER_META, load_model, split_rules_source
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
        target_modules=t.lora_target_modules,
        lora_dropout=t.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


# =========================
# GPU SETUP — use as much of the T4 as fits
# =========================
EFFECTIVE_BATCH = 16     # examples per optimizer step — kept whatever fits on the GPU at once
GPU_BUDGET      = 0.85   # a worst-case batch may use at most 85% of GPU memory (rest: fragmentation, eval)


def _probe(model, batch_size: int, seq_len: int):
    """Peak GPU memory (GB) of a worst-case batch (every example at max length):
    forward + backward. None if it ran out of memory."""
    ids  = torch.randint(1000, 5000, (batch_size, seq_len), device=model.device)
    peak = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 2**30
    except torch.cuda.OutOfMemoryError:
        pass
    model.zero_grad(set_to_none=True)
    del ids
    torch.cuda.empty_cache()
    return peak


def pick_batch_setup(model, seq_len: int) -> tuple:
    """Fastest setup that fits: gradient checkpointing off if possible (~30%
    less compute per step), then the biggest batch per forward pass.
    Returns (batch_size, gradient_checkpointing)."""
    # Hugging Face only checkpoints in training mode, and a freshly loaded model
    # is in eval mode: measured that way, checkpointing never saved anything
    # (Kaggle run 1: "not even one fits" on a 15 GB T4)
    model.train()
    budget = GPU_BUDGET * torch.cuda.get_device_properties(0).total_memory / 2**30
    for checkpointing in (False, True):
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        for batch_size in (16, 8, 4, 2, 1):
            peak = _probe(model, batch_size, seq_len)
            fits = peak is not None and peak < budget
            print(f"  try batch {batch_size:>2}, gradient checkpointing {'on ' if checkpointing else 'off'}: "
                  f"{f'peak {peak:.1f} GB' if peak is not None else 'out of memory'} "
                  f"(limit {budget:.1f} GB) -> {'fits' if fits else 'too big'}", flush=True)
            if fits:
                return batch_size, checkpointing
    raise RuntimeError("Not even one max-length example fits on this GPU")


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

    batch_size, checkpointing = pick_batch_setup(model, t.max_seq_length)
    accumulation = max(1, EFFECTIVE_BATCH // batch_size)
    total_gb     = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU setup ({torch.cuda.get_device_name(0)}, {total_gb:.0f} GB): batch {batch_size} x "
          f"accumulation {accumulation} = {batch_size * accumulation} examples per step, "
          f"gradient checkpointing {'on' if checkpointing else 'off'}", flush=True)

    # Pads input_ids with the pad token and labels with IGNORE_INDEX, so padding
    # never hides the end-of-text token even when pad == eos (Phi-2, Granite)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,

        # Measured by pick_batch_setup() — as much of the GPU as fits, effective batch 16
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        gradient_checkpointing=checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},

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
    print(f"Resuming from {checkpoint}" if checkpoint else f"No checkpoint - starting fresh from plain {CFG.model.name}")

    result = trainer.train(resume_from_checkpoint=checkpoint)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    history = trainer.state.log_history
    evals   = [h["eval_loss"] for h in history if "eval_loss" in h]
    meta = {
        "base_model":     CFG.model.name,
        "prompt_format":  CFG.model.prompt_format,
        "split_rules_from": split_rules_source(),   # the inference loader uses the same rules
        "dataset":        str(CFG.paths.dataset),
        "train_examples": len(tokenized["train"]),
        "val_examples":   len(tokenized["val"]),
        "steps":          trainer.state.global_step,
        "train_loss":     round(result.training_loss, 4),
        "eval_loss":      [round(e, 4) for e in evals],
        "gpu":            f"{torch.cuda.get_device_name(0)}, batch {batch_size} x accumulation {accumulation}, "
                          f"gradient checkpointing {'on' if checkpointing else 'off'}",
        "minutes":        round((time.time() - start_time) / 60, 1),
        "training":       dataclasses.asdict(CFG.training),
        "date":           datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(output_dir, ADAPTER_META), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=1)
    print(f"Adapter saved to {output_dir} (prompt format {CFG.model.prompt_format}, "
          f"final check-set loss {evals[-1] if evals else 'n/a'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(CFG.paths.model_output),
                        help="where checkpoints and the final adapter are saved")
    parser.add_argument("--model", default=CFG.model.name,
                        help="brain to train (default: config model.name)")
    parser.add_argument("--prompt-format", choices=["template", "native_chat"], default=CFG.model.prompt_format,
                        help="prompt format to train with (default: config model.prompt_format)")
    parser.add_argument("--split-rules-from", default=CFG.model.split_rules_from,
                        help="take the text-splitting rules from this brain's tokenizer.json "
                             "(default: config model.split_rules_from, else the brain's own)")
    args = parser.parse_args()
    CFG.model.name             = args.model
    CFG.model.prompt_format    = args.prompt_format
    CFG.model.split_rules_from = args.split_rules_from
    train(args.output_dir)
