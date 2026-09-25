import torch
import re
from model.training.config_loader import CFG
from inference.engine.prompt_builder import max_new_tokens
from inference.engine.prompt_templates import TEMPLATES


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


# ─── Prompt echo filter ──────────────────────────────────────────────────────
# The model sometimes copies the template's instruction sentences into its
# answer (e.g. into a docstring). Only the fixed template text is matched —
# never the user's input, which the answer may legitimately repeat.

def _instruction_lines() -> list:
    lines = []
    for template in TEMPLATES.values():
        for line in template.split("\n"):
            stripped = line.strip()
            if len(stripped) >= 20 and "{" not in stripped and not stripped.startswith("###"):
                lines.append(stripped)
    return lines


_INSTRUCTION_LINES = _instruction_lines()


def _strip_prompt_echo(text: str) -> str:
    """Drop answer lines that repeat (part of) a template instruction line."""
    kept       = []
    skip_blank = False

    for line in text.split("\n"):
        stripped = line.strip()

        if len(stripped) >= 20 and any(stripped in inst for inst in _INSTRUCTION_LINES):
            # '"""' + echo + blank line → keep the quotes, drop the blank too
            skip_blank = bool(kept) and kept[-1].strip() in ('"""', "'''")
            continue

        if skip_blank and not stripped:
            skip_blank = False
            continue

        skip_blank = False
        kept.append(line)

    return "\n".join(kept)


# ─── Code filter for explain/chat modes ──────────────────────────────────────

def remove_code_if_not_allowed(text: str, mode: str) -> str:
    if mode not in ["chat", "explain"]:
        return text.strip()

    # Remove fenced code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Remove triple-quoted blocks
    text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
    text = re.sub(r"'''.*?'''", "", text, flags=re.DOTALL)

    # Remove unpaired leftovers (e.g. a stray opening """ before prose)
    text = re.sub(r'"""|\'\'\'|```', "", text)

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

    # Drop a dangling lead-in whose code was cut ("Here's a simple example:")
    lines = result.split("\n")
    if lines and lines[-1].rstrip().endswith(":"):
        result = "\n".join(lines[:-1]).strip()

    # If barely anything survived, the output was entirely code
    if len(result) < 20:
        return ""

    return result


# ─── Stop words ──────────────────────────────────────────────────────────────
# Passed to model.generate() as stop_strings so generation halts as soon as
# one appears (instead of running to max_new_tokens), then cut from the text
# by _apply_stop_words().

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
    "\nQuestion:",
    "\nAnswer:",
    # Echoes of the prompt's own RAG / history blocks (see prompt_builder.py)
    "Relevant examples from codebase:",
    "Recent context:",
    "\nINSTRUCTION:",
    "\nOUTPUT:",
    "\n[1]\n",
    "\n[2]\n",
    "\n[3]\n",
]

# Code modes: the v1 model was trained without an end-of-text token, so after a
# finished answer it keeps writing tests and demo calls. Stop at those. Test
# code starts after a blank line — a single "\n" also matched the prompt's
# last newline and killed answers whose function is named test_* (MBPP 19).
_CODE_STOP_WORDS = [
    "\n\ndef test_",
    "\n\nclass Test",
    "\n\n@pytest",
    "\nif __name__",
    "\n\nprint(",
]

# Prose modes: code starting means the model has drifted — stop there and let
# remove_code_if_not_allowed() and the retry handle the rest.
_PROSE_STOP_WORDS = [
    "```",
    "\ndef ",
    "\nclass ",
    "\nimport ",
    "\nfrom ",
    ">>>",
]


def _stop_words_for(mode: str) -> list:
    if mode in ["generate", "debug", "refactor"]:
        return _STOP_WORDS + _CODE_STOP_WORDS
    if mode in ["chat", "explain"]:
        return _STOP_WORDS + _PROSE_STOP_WORDS
    return _STOP_WORDS


def _apply_stop_words(text: str, mode: str = None) -> str:
    stops = _stop_words_for(mode)
    cut = min(
        (text.find(stop) for stop in stops if text.find(stop) != -1),
        default=len(text)
    )
    return text[:cut]


# ─── Core generation ─────────────────────────────────────────────────────────

# Repeat penalties come from CFG.generation per mode: prose modes keep them
# against loops; code modes turn them off — code must repeat names and the
# user's code (with them on it produced is_palindrom, find_volume for find_Volume).

def _run_generation(model, tokenizer, prompt, max_tokens, temperature, mode=None):
    inputs   = tokenizer(prompt, return_tensors="pt").to(model.device)
    settings = CFG.generation.for_mode(mode)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens       = max_tokens or max_new_tokens(),
            do_sample            = temperature > 0,
            temperature          = temperature if temperature > 0 else 1.0,
            repetition_penalty   = settings.repetition_penalty,
            no_repeat_ngram_size = settings.no_repeat_ngram_size,
            stop_strings         = _stop_words_for(mode),
            tokenizer            = tokenizer,
            eos_token_id         = tokenizer.eos_token_id,
            pad_token_id         = tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


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

    text = _run_generation(model, tokenizer, prompt, max_tokens, temperature, mode)
    text = _apply_stop_words(text, mode)
    text = _strip_prompt_echo(text)
    text = remove_code_if_not_allowed(text, mode)
    text = _strip_artifacts(text)

    # Retry at higher temperature if output is empty
    if not text.strip() and mode in ["chat", "explain"]:
        text = _run_generation(model, tokenizer, prompt, max_tokens, 0.5, mode)
        text = _apply_stop_words(text, mode)
        text = _strip_prompt_echo(text)
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