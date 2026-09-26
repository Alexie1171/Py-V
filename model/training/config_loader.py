"""
config_loader.py — PY-V
Loads configs/config.yaml and exposes typed dataclasses for
model, training, path, and RAG config. All other modules import from here
instead of hardcoding values.
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
    name:             str
    max_tokens:       int
    prompt_format:    str             # "template" or "native_chat"
    split_rules_from: Optional[str]   # whose tokenizer.json splits the text (None = the brain's own)


@dataclass
class GenerationModeConfig:
    repetition_penalty:   float
    no_repeat_ngram_size: int


@dataclass
class GenerationConfig:
    modes: dict   # mode name (or "default") → GenerationModeConfig

    def for_mode(self, mode: str) -> GenerationModeConfig:
        return self.modes.get(mode, self.modes["default"])


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
    lora_target_modules:    list    # layer names LoRA trains — depend on the brain
    max_seq_length:         int


@dataclass
class PathsConfig:
    raw_data:       Path
    processed_data: Path
    dataset:        Path
    model_output:   Path
    val_dataset:    Path


@dataclass
class RAGConfig:
    enabled:      bool
    index_path:   Path
    top_k:        int
    active_modes: list


@dataclass
class MemoryConfig:
    enabled:           bool
    db_path:           Path
    top_k:             int
    code_top_k:        int
    max_prompt_tokens: int
    history_turns:     int
    active_code_modes: list
    semantic_search:   bool


@dataclass
class DatasetV2Config:
    output_dir:  Path
    sample_size: int
    sources:     dict   # source name → its settings (hf_id, config, split, fetch, ...)
    build:       dict   # step 5 mix settings; build["output_dir"] is a Path


@dataclass
class IntentConfig:
    brain_for_unclear: bool   # a message the word rules can't place clearly → the brain picks the mode


@dataclass
class EvaluationConfig:
    dataset:         str
    config:          str
    split:           str
    num_problems:    int
    timeout_seconds: int
    output_dir:      Path


@dataclass
class AppConfig:
    model:      ModelConfig
    generation: GenerationConfig
    training:   TrainingConfig
    paths:      PathsConfig
    rag:        RAGConfig
    memory:     MemoryConfig
    intent:     IntentConfig
    dataset_v2: DatasetV2Config
    evaluation: EvaluationConfig


# ─── Parser ───────────────────────────────────────────────────────────────────

def load_config() -> AppConfig:
    """
    Parse configs/config.yaml into a typed AppConfig object.
    Call this once at module startup and pass the config around.
    """
    raw = _load_raw()

    model_cfg = ModelConfig(
        name          = raw["model"]["name"],
        max_tokens    = raw["model"]["max_tokens"],
        prompt_format = raw["model"].get("prompt_format", "template"),
        split_rules_from = raw["model"].get("split_rules_from"),
    )
    if model_cfg.prompt_format not in ("template", "native_chat"):
        raise ValueError(f"model.prompt_format must be 'template' or 'native_chat', not {model_cfg.prompt_format!r}")

    g = raw.get("generation", {})
    g.setdefault("default", {})
    generation_cfg = GenerationConfig(modes={
        mode: GenerationModeConfig(
            repetition_penalty   = s.get("repetition_penalty",   1.0),
            no_repeat_ngram_size = s.get("no_repeat_ngram_size", 0),
        )
        for mode, s in g.items()
    })

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
        lora_target_modules   = t["lora_target_modules"],
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

    r = raw.get("rag", {})
    rag_cfg = RAGConfig(
        enabled      = r.get("enabled",      False),
        index_path   = Path(r.get("index_path", "./retrieval/index")),
        top_k        = r.get("top_k",        3),
        active_modes = r.get("active_modes", ["generate", "debug", "refactor"]),
    )

    m = raw.get("memory", {})
    memory_cfg = MemoryConfig(
        enabled           = m.get("enabled",           False),
        db_path           = Path(m.get("db_path",      "./data/memory/v_memory.db")),
        top_k             = m.get("top_k",             5),
        code_top_k        = m.get("code_top_k",        2),
        max_prompt_tokens = m.get("max_prompt_tokens", 300),
        history_turns     = m.get("history_turns",     6),
        active_code_modes = m.get("active_code_modes", ["generate", "debug", "refactor"]),
        semantic_search   = m.get("semantic_search",   True),
    )

    intent_cfg = IntentConfig(brain_for_unclear=raw.get("intent", {}).get("brain_for_unclear", False))

    d = raw.get("dataset_v2", {})
    build = dict(d.get("build", {}))
    build["output_dir"] = Path(build.get("output_dir", "./data/datasets/v2"))
    dataset_v2_cfg = DatasetV2Config(
        output_dir  = Path(d.get("output_dir", "./data/raw/v2")),
        sample_size = d.get("sample_size", 20),
        sources     = d.get("sources", {}),
        build       = build,
    )

    e = raw.get("evaluation", {})
    evaluation_cfg = EvaluationConfig(
        dataset         = e.get("dataset",         "google-research-datasets/mbpp"),
        config          = e.get("config",          "sanitized"),
        split           = e.get("split",           "test"),
        num_problems    = e.get("num_problems",    100),
        timeout_seconds = e.get("timeout_seconds", 10),
        output_dir      = Path(e.get("output_dir", "./experiments/outputs")),
    )

    return AppConfig(
        model      = model_cfg,
        generation = generation_cfg,
        training   = training_cfg,
        paths      = paths_cfg,
        rag        = rag_cfg,
        memory     = memory_cfg,
        intent     = intent_cfg,
        dataset_v2 = dataset_v2_cfg,
        evaluation = evaluation_cfg,
    )


# ─── Module-level singleton ───────────────────────────────────────────────────
# Import this directly: `from model.training.config_loader import CFG`

CFG: AppConfig = load_config()