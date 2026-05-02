"""
config_loader.py — PY-V
Loads configs/config.yaml and exposes typed dataclasses for
model, training, and path config. All other modules import from here
instead of hardcoding values.
"""

import os
import yaml
from dataclasses import dataclass
from pathlib import Path

# ─── Config Root ──────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _load_raw() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found at: {_CONFIG_PATH}")
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Typed Sections ───────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    name:       str
    max_tokens: int


@dataclass
class TrainingConfig:
    batch_size:             int
    gradient_accumulation:  int
    epochs:                 int
    learning_rate:          float
    warmup_steps:           int
    save_steps:             int
    eval_steps:             int
    lora_r:                 int
    lora_alpha:             int
    lora_dropout:           float
    max_seq_length:         int


@dataclass
class PathsConfig:
    raw_data:       Path
    processed_data: Path
    dataset:        Path
    model_output:   Path
    val_dataset:    Path


@dataclass
class AppConfig:
    model:    ModelConfig
    training: TrainingConfig
    paths:    PathsConfig


# ─── Parser ───────────────────────────────────────────────────────────────────

def load_config() -> AppConfig:
    """
    Parse configs/config.yaml into a typed AppConfig object.
    Call this once at module startup and pass the config around.
    """
    raw = _load_raw()

    model_cfg = ModelConfig(
        name       = raw["model"]["name"],
        max_tokens = raw["model"]["max_tokens"],
    )

    t = raw.get("training", {})
    training_cfg = TrainingConfig(
        batch_size            = t.get("batch_size",            1),
        gradient_accumulation = t.get("gradient_accumulation", 16),
        epochs                = t.get("epochs",                3),
        learning_rate         = t.get("learning_rate",         2e-4),
        warmup_steps          = t.get("warmup_steps",          10),
        save_steps            = t.get("save_steps",            50),
        eval_steps            = t.get("eval_steps",            50),
        lora_r                = t.get("lora_r",                8),
        lora_alpha            = t.get("lora_alpha",            32),
        lora_dropout          = t.get("lora_dropout",          0.05),
        max_seq_length        = t.get("max_seq_length",        384),
    )

    p = raw.get("paths", {})
    paths_cfg = PathsConfig(
        raw_data       = Path(p.get("raw_data",       "./data/raw")),
        processed_data = Path(p.get("processed_data", "./data/processed")),
        dataset        = Path(p.get("dataset",        "./data/datasets/train.jsonl")),
        val_dataset    = Path(p.get("val_dataset",    "./data/datasets/val.jsonl")),
        model_output   = Path(p.get("model_output",   "./model/lora")),
    )

    return AppConfig(model=model_cfg, training=training_cfg, paths=paths_cfg)


# ─── Module-level singleton ───────────────────────────────────────────────────
# Import this directly: `from model.training.config_loader import CFG`

CFG: AppConfig = load_config()