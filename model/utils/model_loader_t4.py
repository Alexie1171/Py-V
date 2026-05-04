"""
model_loader.py — PY-V (model/utils/)

Universal model loader for:
- GTX 1650 (local 4GB VRAM)
- Google Colab T4 GPU

Fixes:
- Phi pad_token_id crash
- config initialization bug
- tokenizer padding issues
- 4-bit quant compatibility
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)

from model.training.config_loader import CFG


def load_model():
    model_name = CFG.model.name

    # =========================
    # TOKENIZER
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Ensure padding is always safe
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =========================
    # LOAD CONFIG FIRST (CRITICAL FIX)
    # =========================
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # FIX: Phi / HF models missing pad_token_id
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id

    # =========================
    # QUANTIZATION (GPU ONLY)
    # =========================
    is_cuda = torch.cuda.is_available()

    bnb_config = None
    if is_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # =========================
    # MODEL LOAD (CONFIG PATCHED BEFORE INIT)
    # =========================
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,  # 🔥 CRITICAL FIX
        quantization_config=bnb_config if is_cuda else None,
        device_map="auto" if is_cuda else None,
        torch_dtype=torch.float16 if is_cuda else torch.float32,
        trust_remote_code=True,
    )

    return model, tokenizer