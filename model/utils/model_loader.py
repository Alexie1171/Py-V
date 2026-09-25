"""
model_loader.py — PY-V (model/utils/)
Shared base model loader. Reads model name and quant settings from
configs/config.yaml via config_loader. Both inference/engine and
model/training import from here — no duplication.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model.training.config_loader import CFG

# Written next to every adapter by train_lora_t4.py: brain, prompt format, dataset,
# GPU setup, losses. The inference loader reads the prompt format from it.
ADAPTER_META = "v_adapter.json"


def load_model():
    """
    Load the brain defined in config.yaml (model.name) with 4-bit
    quantization. Safe for GTX 1650 4GB VRAM.

    Returns:
        model:     quantized AutoModelForCausalLM; model.v_prompt_format is the
                   prompt format it gets (config model.prompt_format — the
                   inference loader overrides it from an adapter's v_adapter.json)
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
    model.v_prompt_format = CFG.model.prompt_format

    return model, tokenizer