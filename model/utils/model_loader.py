"""
model_loader.py — PY-V (model/utils/)
Shared base model loader. Reads model name and quant settings from
configs/config.yaml via config_loader. Both inference/engine and
model/training import from here — no duplication.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model.training.config_loader import CFG


def load_model():
    """
    Load Phi-2 (or any model defined in config.yaml) with 4-bit
    quantization. Safe for GTX 1650 4GB VRAM.

    Returns:
        model:     quantized AutoModelForCausalLM
        tokenizer: matching AutoTokenizer with pad_token set
    """
    model_name = CFG.model.name

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    return model, tokenizer