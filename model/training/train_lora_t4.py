import glob
import os
import time
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from model.utils.model_loader import load_model
from model.training.dataset_loader import get_formatted_dataset

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
            eta = avg_step * (state.max_steps - step)

            is_best = ""
            if loss < self.best_loss:
                self.best_loss = loss
                is_best = " ← best"

            print(
                f"Step {step}/{state.max_steps} | "
                f"Loss: {loss:.4f}{is_best} | "
                f"Elapsed: {elapsed/60:.1f}min | "
                f"ETA: {eta/60:.1f}min"
            )

            self.last_step_time = now


# =========================
# DATA
# =========================
def load_data():
    return get_formatted_dataset()


def tokenize(tokenizer, example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=384,
        padding=False,
    )


# =========================
# MODEL
# =========================
def load_base_model():
    model, tokenizer = load_model()

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def apply_lora(model):
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


# =========================
# CHECKPOINT RESUME
# =========================
def get_latest_checkpoint():
    checkpoints = sorted(
        glob.glob("model/lora/checkpoint-*"),
        key=os.path.getmtime,
    )
    return checkpoints[-1] if checkpoints else None


# =========================
# TRAINING
# =========================
def train():

    dataset = load_data()

    model, tokenizer = load_base_model()
    model = apply_lora(model)

    model.print_trainable_parameters()

    tokenized = dataset.map(
        lambda x: tokenize(tokenizer, x),
        batched=True,
        num_proc=2,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir="/content/drive/MyDrive/PY-V/model/lora",

        # ⚡ SPEED OPTIMIZED FOR T4
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,

        learning_rate=5e-5,
        num_train_epochs=1,
        warmup_steps=10,

        logging_steps=10,

        # 💾 SAVE EVERY 50 STEPS (YOUR REQUEST)
        save_steps=50,
        eval_steps=50,

        save_total_limit=2,

        fp16=True,
        optim="paged_adamw_8bit",

        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        group_by_length=True,

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

    checkpoint = get_latest_checkpoint()

    trainer.train(resume_from_checkpoint=checkpoint)

    model.save_pretrained("/content/drive/MyDrive/PY-V/model/lora")
    tokenizer.save_pretrained("/content/drive/MyDrive/PY-V/model/lora")


if __name__ == "__main__":
    train()