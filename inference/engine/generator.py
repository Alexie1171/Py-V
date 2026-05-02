"""
generator.py — PY-V (inference/engine/)
Code generation logic. Builds prompts via prompt_builder,
runs inference, strips the prompt prefix from output.
Optimized for GTX 1650 4GB VRAM.
"""

import torch
from inference.engine.prompt_builder import build_inference_prompt, extract_output, max_new_tokens


def generate_code(
    model,
    tokenizer,
    instruction: str,
    max_tokens:  int  = None,
    temperature: float = 0.2,
) -> str:
    """
    Generate Python code from a natural-language instruction.

    Args:
        model:       Loaded (quantized) causal LM
        tokenizer:   Matching tokenizer
        instruction: Natural language prompt, e.g. "Write a function to sort a list"
        max_tokens:  Override max new tokens (defaults to config value)
        temperature: Sampling temperature — lower = more deterministic

    Returns:
        Generated code string with prompt prefix stripped.
    """
    prompt = build_inference_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = max_tokens or max_new_tokens(),
            do_sample      = temperature > 0,
            temperature    = temperature if temperature > 0 else 1.0,
            pad_token_id   = tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_ids    = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True)

    return extract_output(raw_output)