import glob
import os
import time

import torch
torch.backends.cuda.matmul.allow_tf32 = True

from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from model.training.config_loader import CFG
from model.utils.model_loader import load_model
from model.training.dataset_loader import IGNORE_INDEX, get_tokenized_dataset

start_time = time.time()


# =========================
# LOGGING CALLBACK
# =========================

class TrainingLogger(TrainerCallback):

    def __init__(self):
        self.step_times  = []
        self.last_step_time = None
        self.best_loss   = float("inf")

    def on_train_begin(self, args, state, control, **kwargs):
        print("\n" + "=" * 60)
        print("  PY-V LoRA TRAINING STARTED")
        print(f"  Total steps : {state.max_steps}")
        print(f"  Epochs      : {args.num_train_epochs}")
        print(f"  Batch size  : {args.per_device_train_batch_size}")
        print(f"  Grad accum  : {args.gradient_accumulation_steps}")
        print(f"  LR          : {args.learning_rate}")
        print("=" * 60 + "\n")
        self.last_step_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        now     = time.time()
        elapsed = now - start_time
        step    = state.global_step

        # ── Train loss ───────────────────────────────────────────
        if "loss" in logs:
            loss = logs["loss"]
            lr   = logs.get("learning_rate", 0)

            # Rolling ETA over last 20 steps
            if self.last_step_time:
                self.step_times.append(now - self.last_step_time)
                if len(self.step_times) > 20:
                    self.step_times.pop(0)

            avg_step = sum(self.step_times) / len(self.step_times) if self.step_times else 0
            eta_sec  = avg_step * (state.max_steps - step)
            eta_str  = f"{eta_sec/3600:.1f}h" if eta_sec >= 3600 else f"{eta_sec/60:.0f}min"

            is_best = ""
            if loss < self.best_loss:
                self.best_loss = loss
                is_best = " ← best"

            print(
                f"  Step {step:>4}/{state.max_steps} | "
                f"Loss: {loss:.4f}{is_best} | "
                f"LR: {lr:.2e} | "
                f"Elapsed: {elapsed/60:.1f}min | "
                f"ETA: {eta_str}"
            )
            self.last_step_time = now

        # ── Eval loss ────────────────────────────────────────────
        if "eval_loss" in logs:
            print(f"\n{'─' * 60}")
            print(f"  EVAL @ step {step} | Eval Loss: {logs['eval_loss']:.4f}")
            print(f"{'─' * 60}\n")

    def on_save(self, args, state, control, **kwargs):
        print(
            f"\n  ✔ Checkpoint saved at step {state.global_step} "
            f"→ {args.output_dir}/checkpoint-{state.global_step}\n"
        )

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("  PY-V LoRA TRAINING COMPLETE")
        print(f"  Total time  : {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)")
        print(f"  Best loss   : {self.best_loss:.4f}")
        print(f"  Final step  : {state.global_step}")
        print(f"  Saved to    : {args.output_dir}")
        print("=" * 60 + "\n")


# =========================
# 1. LOAD MODEL (4-bit)
# =========================

def load_base_model():
    model, tokenizer = load_model()
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


# =========================
# 2. APPLY LoRA
# =========================

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
# 3. CHECKPOINT PROMPT
# =========================

def resolve_checkpoint() -> str | None:
    """
    Auto-detect the latest checkpoint in model/lora/.
    If one exists, ask the user whether to resume from it.
    Returns the checkpoint path to resume from, or None to start fresh.
    """
    checkpoints = sorted(
        glob.glob("model/lora/checkpoint-*"),
        key=os.path.getmtime,
    )

    if not checkpoints:
        print("\n  No checkpoints found. Starting fresh.\n")
        return None

    latest = checkpoints[-1]
    print(f"\n  Checkpoint found: '{latest}'")
    answer = input("  Resume from it? (y/n): ").strip().lower()

    if answer == "y":
        print(f"  Resuming from {latest}\n")
        return latest
    else:
        print("  Starting fresh.\n")
        return None


# =========================
# 4. TRAINING PIPELINE
# =========================

def train():
    t = CFG.training

    model, tokenizer = load_base_model()
    model = apply_lora(model)
    model.print_trainable_parameters()

    tokenized = get_tokenized_dataset(tokenizer)

    # Labels padded with IGNORE_INDEX, so padding never hides the end-of-text
    # token even when pad == eos (Phi-2, Granite)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=IGNORE_INDEX,
    )

    training_args = TrainingArguments(
        output_dir="model/lora",
        per_device_train_batch_size=t.batch_size,
        per_device_eval_batch_size=t.batch_size,
        gradient_accumulation_steps=t.gradient_accumulation,
        learning_rate=t.learning_rate,
        num_train_epochs=t.epochs,
        warmup_steps=t.warmup_steps,
        logging_steps=1,
        logging_strategy="steps",
        logging_first_step=True,
        save_steps=t.save_steps,
        eval_strategy="steps",
        eval_steps=t.eval_steps,
        save_total_limit=2,
        label_names=["labels"],   # PeftModel hides them — needed for eval loss
        fp16=True,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[TrainingLogger()],
    )

    # Ask user whether to resume from checkpoint
    resume = resolve_checkpoint()
    trainer.train(resume_from_checkpoint=resume)

    model.save_pretrained("model/lora")
    tokenizer.save_pretrained("model/lora")


if __name__ == "__main__":
    train()