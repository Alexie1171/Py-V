"""
prompt_builder.py — PY-V (inference/engine/)
Central prompt formatting for Phi-2. Both the training dataset loader
and the inference generator must use these functions so the model sees
the exact same format at train time and inference time.

Phi-2 Instruct format:
    Instruct: <instruction>
    Output: <code>
"""

from model.training.config_loader import CFG


# ─── Templates ────────────────────────────────────────────────────────────────

_INSTRUCT_PREFIX = "Instruct"
_OUTPUT_PREFIX   = "Output"


def build_inference_prompt(instruction: str) -> str:
    """
    Build a prompt for inference — no output tail so the model completes it.

    Example output:
        Instruct: Write a Python function to reverse a string.
        Output:
    """
    instruction = instruction.strip()
    return f"{_INSTRUCT_PREFIX}: {instruction}\n{_OUTPUT_PREFIX}:"


def build_training_prompt(instruction: str, output: str) -> str:
    """
    Build a full prompt+completion string for training.
    The model learns to produce <output> given <instruction>.

    Example output:
        Instruct: Write a Python function to reverse a string.
        Output:
        def reverse_string(s):
            return s[::-1]
    """
    instruction = instruction.strip()
    output      = output.strip()
    return (
        f"{_INSTRUCT_PREFIX}: {instruction}\n"
        f"{_OUTPUT_PREFIX}:\n"
        f"{output}"
    )


def extract_output(full_text: str) -> str:
    """
    Strip the prompt prefix from generated text, returning only the
    model's output. Used in the generator after decoding.

    If the output marker isn't found, returns the full text as-is
    (safe fallback).
    """
    marker = f"{_OUTPUT_PREFIX}:"
    idx    = full_text.find(marker)
    if idx == -1:
        return full_text.strip()
    return full_text[idx + len(marker):].strip()


# ─── Token budget helper ──────────────────────────────────────────────────────

def max_new_tokens() -> int:
    """Return max_tokens from config for use in generation calls."""
    return CFG.model.max_tokens