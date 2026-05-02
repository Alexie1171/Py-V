import torch
torch.backends.cuda.matmul.allow_tf32 = True
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
import os
import time
start_time = time.time()

from model.utils.model_loader import load_model
from model.training.dataset_loader import get_formatted_dataset
from transformers import TrainerCallback

class LossPrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = state.global_step
            loss = logs["loss"]
            elapsed = time.time() - start_time
            print(f"Step {step} | Loss: {loss:.4f} | Time: {elapsed/60:.2f} min")

# =========================
# 1. LOAD DATASET
# =========================
def load_data():
    dataset = get_formatted_dataset()
    return dataset


# =========================
# 2. TOKENIZATION
# =========================
def tokenize_function(tokenizer, example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=384,
        padding=False
    )

    return tokens

# =========================
# 3. LOAD MODEL (4-bit)
# =========================
def load_base_model():
    model, tokenizer = load_model()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # IMPORTANT: prepare for LoRA training
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


# =========================
# 4. APPLY LoRA
# =========================
def apply_lora(model):
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "dense"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    return model


# =========================
# 5. TRAINING PIPELINE
# =========================
def train():
    dataset = load_data()

    model, tokenizer = load_base_model()
    model = apply_lora(model)
    model.print_trainable_parameters()

    # Tokenize dataset
    tokenized = dataset.map(
        lambda x: tokenize_function(tokenizer, x),
        batched=True
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
        )

    # Training config (QUALITY MODE)
    training_args = TrainingArguments(
        output_dir="model/lora",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=1,
        logging_strategy="steps",
        logging_first_step=True,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=2,
        fp16=True,
        report_to="none",
        optim="paged_adamw_8bit"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LossPrinterCallback()]
    )

    trainer.train()

    # Save final model
    model.save_pretrained("model/lora")
    tokenizer.save_pretrained("model/lora")

    print("TRAINING COMPLETE ✔")


if __name__ == "__main__":
    train()