import torch
import re
from inference.engine.prompt_builder import max_new_tokens


# ─── Artifact patterns that leak from training data into prose responses ──────

_ARTIFACT_PATTERNS = [
    r"\nExercise:.*",
    r"\nTask:.*",
    r"\[\.{3}\]",
    r"\[\.\.\.\]",
]


def _strip_artifacts(text: str) -> str:
    """Remove training-data suffix patterns that bleed into prose responses."""
    for pattern in _ARTIFACT_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.MULTILINE)
    return text.strip()


# ─── Code filter for explain/chat modes ──────────────────────────────────────

def remove_code_if_not_allowed(text: str, mode: str) -> str:
    if mode not in ["chat", "explain"]:
        return text.strip()

    # Remove fenced code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Remove triple-quoted blocks
    text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
    text = re.sub(r"'''.*?'''", "", text, flags=re.DOTALL)

    lines   = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned.append(line)
            continue

        # Block comment lines — code artifacts leaking into prose
        if stripped.startswith("#"):
            continue

        # Block unambiguous code lines
        if any([
            stripped.startswith("def "),
            stripped.startswith("class "),
            stripped.startswith("return "),
            stripped.startswith("print("),
            stripped.startswith("if __name__"),
            stripped.startswith("else:"),
            stripped.startswith("elif "),
            stripped.startswith("for "),
            stripped.startswith("while "),
            re.match(r"^(import|from)\s+\w+", stripped),
        ]):
            continue

        # Block indented lines that look like code
        if (line.startswith("    ") or line.startswith("\t")) and any([
            stripped.startswith("return "),
            stripped.startswith("if "),
            stripped.startswith("else:"),
            stripped.startswith("elif "),
            stripped.startswith("for "),
            stripped.startswith("while "),
            "=" in stripped and not stripped.endswith("."),
        ]):
            continue

        cleaned.append(line)

    result = "\n".join(cleaned).strip()

    # If barely anything survived, the output was entirely code
    if len(result) < 20:
        return ""

    return result


# ─── Core generation ─────────────────────────────────────────────────────────

def _run_generation(model, tokenizer, prompt, max_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens       = max_tokens or max_new_tokens(),
            do_sample            = temperature > 0,
            temperature          = temperature if temperature > 0 else 1.0,
            repetition_penalty   = 1.1,
            no_repeat_ngram_size = 4,
            eos_token_id         = tokenizer.eos_token_id,
            pad_token_id         = tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ─── Stop words ──────────────────────────────────────────────────────────────

_STOP_WORDS = [
    "User:",
    "Assistant:",
    "User :",
    "Assistant :",
    "\nUser",
    "\nAssistant",
    "### Instruction",
    "### Answer",
    "###",
    "\nExercise:",
    "\nTask:",
]


def _apply_stop_words(text: str) -> str:
    cut = min(
        (text.find(stop) for stop in _STOP_WORDS if text.find(stop) != -1),
        default=len(text)
    )
    return text[:cut]


# ─── Public interface ─────────────────────────────────────────────────────────

def generate_from_prompt(
    model,
    tokenizer,
    prompt:      str,
    mode:        str   = None,
    max_tokens:  int   = None,
    temperature: float = 0.2,
) -> str:

    if mode in ["explain", "chat"]:
        temperature = min(temperature, 0.3)

    text = _run_generation(model, tokenizer, prompt, max_tokens, temperature)
    text = _apply_stop_words(text)
    text = remove_code_if_not_allowed(text, mode)
    text = _strip_artifacts(text)

    # Retry at higher temperature if output is empty
    if not text.strip() and mode in ["chat", "explain"]:
        text = _run_generation(model, tokenizer, prompt, max_tokens, 0.5)
        text = _apply_stop_words(text)
        text = remove_code_if_not_allowed(text, mode)
        text = _strip_artifacts(text)

    return text.strip()


def generate_code(
    model,
    tokenizer,
    instruction: str,
    max_tokens:  int   = None,
    temperature: float = 0.2,
) -> str:
    """
    Stateless code generation for the /generate API endpoint.
    No session context or RAG — bare inference prompt only.
    """
    from inference.engine.prompt_builder import build_inference_prompt
    prompt = build_inference_prompt(instruction)
    return generate_from_prompt(
        model, tokenizer, prompt,
        mode=        "generate",
        max_tokens=  max_tokens,
        temperature= temperature,
    )