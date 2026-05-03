import torch
from inference.engine.prompt_builder import max_new_tokens
import re

def remove_code_if_not_allowed(text: str, mode: str) -> str:
    if mode not in ["chat", "explain"]:
        return text.strip()

    # Remove fenced code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Remove triple-quoted blocks
    text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
    text = re.sub(r"'''.*?'''", "", text, flags=re.DOTALL)

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines that follow code removal
        if not stripped:
            cleaned.append(line)
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

        # Block indented lines only if they look like code
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

    # If barely anything survived, the whole output was code — return empty
    result = "\n".join(cleaned).strip()
    if len(result) < 20:
        return ""

    return result

def _run_generation(model, tokenizer, prompt, max_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens or max_new_tokens(),
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            repetition_penalty=1.1,        # reduced from 1.2
            no_repeat_ngram_size=4,        # increased from 3 — less aggressive
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def generate_from_prompt(
    model,
    tokenizer,
    prompt: str,
    mode: str = None,
    max_tokens: int = None,
    temperature: float = 0.2,
) -> str:

    if mode in ["explain", "chat"]:
        temperature = min(temperature, 0.3)

    stop_words = [
        "User:",
        "Assistant:",
        "User :",
        "Assistant :",
        "\nUser",
        "\nAssistant",
        "### Instruction",
        "### Answer",
        "###"
    ]

    def apply_stop_words(text: str) -> str:
        cut = min(
            (text.find(stop) for stop in stop_words if text.find(stop) != -1),
            default=len(text)
        )
        return text[:cut]

    text = _run_generation(model, tokenizer, prompt, max_tokens, temperature)
    text = apply_stop_words(text)
    text = remove_code_if_not_allowed(text, mode)

    if not text.strip() and mode in ["chat", "explain"]:
        text = _run_generation(model, tokenizer, prompt, max_tokens, 0.5)
        text = apply_stop_words(text)
        text = remove_code_if_not_allowed(text, mode)

    return text.strip()