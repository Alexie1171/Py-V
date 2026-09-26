"""
model_loader.py — PY-V (model/utils/)
Shared base model loader. Reads model name and quant settings from
configs/config.yaml via config_loader. Both inference/engine and
model/training import from here — no duplication.
"""

import json

import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model.training.config_loader import CFG

# Written next to every adapter by train_lora_t4.py: brain, prompt format, text-splitting
# rules, dataset, GPU setup, losses. The inference loader reads format and rules from it.
ADAPTER_META = "v_adapter.json"

SPLIT_PARTS = ("normalizer", "pre_tokenizer", "decoder")   # how text is cut up / put back together


def split_rules_source(model_name: str = None, split_rules_from: str = None) -> str:
    """Whose tokenizer.json gives the splitting rules: split_rules_from, else
    config model.split_rules_from, else the brain itself."""
    return split_rules_from or CFG.model.split_rules_from or model_name or CFG.model.name


def load_tokenizer(model_name: str = None, split_rules_from: str = None):
    """
    The brain's tokenizer, cutting text up the same way in every transformers
    version. Some versions (5.0.0 — Kaggle, 2026-09) rebuild a "GPT2Tokenizer"
    from vocab + merges with GPT-2's splitting rule instead of the brain's
    tokenizer.json: Granite 4.1 base then got text in pieces it never learned
    (MBPP 69 → 17). So normalizer, pre-tokenizer and decoder always come from a
    tokenizer.json — the brain's own, or that of split_rules_source() for a
    brain whose own file has a wrong rule (it must have the same merges).
    """
    model_name = model_name or CFG.model.name
    source     = split_rules_source(model_name, split_rules_from)
    tokenizer  = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    backend    = getattr(tokenizer, "backend_tokenizer", None)
    try:
        rules = Tokenizer.from_file(hf_hub_download(source, "tokenizer.json"))
    except Exception as error:
        if source != model_name:   # asked for another brain's rules — never silently skip them
            raise ValueError(f"No tokenizer.json for {source} - cannot take its splitting rules") from error
        print(f"Tokenizer: no tokenizer.json for {source} ({type(error).__name__}) - using transformers' own")
        return tokenizer
    if backend is None:
        return tokenizer

    wanted = json.loads(rules.to_str())
    if source != model_name:
        own = json.loads(Tokenizer.from_file(hf_hub_download(model_name, "tokenizer.json")).to_str())
        if own["model"].get("merges") != wanted["model"].get("merges"):
            raise ValueError(f"{source}'s splitting rules don't fit {model_name}: different merges")

    have    = json.loads(backend.to_str())
    changed = [part for part in SPLIT_PARTS if have.get(part) != wanted.get(part)]
    for part in changed:
        setattr(backend, part, getattr(rules, part))
    if changed:
        print(f"Tokenizer: {', '.join(changed)} taken from {source}'s tokenizer.json")
    return tokenizer


def load_model(split_rules_from: str = None):
    """
    Load the brain defined in config.yaml (model.name) with 4-bit
    quantization. Safe for GTX 1650 4GB VRAM.

    Returns:
        model:     quantized AutoModelForCausalLM; model.v_prompt_format is the
                   prompt format it gets (config model.prompt_format — the
                   inference loader overrides it from an adapter's v_adapter.json),
                   model.v_split_rules whose splitting rules its tokenizer uses
        tokenizer: matching tokenizer (load_tokenizer — splitting rules of
                   split_rules_source(), see there) with pad_token set
    """
    model_name = CFG.model.name

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = load_tokenizer(model_name, split_rules_from)
    tokenizer.pad_token = tokenizer.eos_token

    # Whole model on the first visible GPU. "auto" would split it across both
    # GPUs of a 2-GPU machine (Kaggle T4 x2) — slower, and train_lora_t4 sizes
    # its batch from GPU 0 only. The GPU pipeline gives every job its own GPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.v_prompt_format = CFG.model.prompt_format
    model.v_split_rules   = split_rules_source(model_name, split_rules_from)   # recorded in test summaries

    return model, tokenizer