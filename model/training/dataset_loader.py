"""
dataset_loader.py — PY-V
Loads train/val JSONL datasets and formats them for LoRA training.
All paths come from configs/config.yaml via config_loader.
"""

from datasets import load_dataset

from model.training.config_loader import CFG


def load_py_v_dataset():
    """Load train and val splits from paths defined in config.yaml."""
    train_path = str(CFG.paths.dataset)
    val_path   = str(CFG.paths.val_dataset)

    dataset = load_dataset(
        "json",
        data_files={
            "train": train_path,
            "val":   val_path,
        }
    )
    return dataset


def format_example(example: dict) -> dict:
    """
    Format a single sample into the Phi-2 Instruct prompt template.
    Uses prompt_builder so the format stays consistent between
    training and inference.
    """
    from inference.engine.prompt_builder import build_training_prompt
    return {"text": build_training_prompt(example["instruction"], example["output"])}


def get_formatted_dataset():
    """Load dataset and apply prompt formatting. Ready for tokenization."""
    dataset = load_py_v_dataset()
    dataset = dataset.map(
        format_example,
        remove_columns=dataset["train"].column_names,
    )
    return dataset