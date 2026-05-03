import torch
from inference.engine.prompt_builder import max_new_tokens
import re


def remove_code_if_not_allowed(text: str, mode: str):

    if mode in ["chat", "explain"]:

        # remove full code blocks
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

        lines = text.split("\n")
        cleaned = []

        for line in lines:
            stripped = line.strip()

            # HARD BLOCK patterns
            if any([
                stripped.startswith("def "),
                stripped.startswith("class "),
                stripped.startswith("import "),
                stripped.startswith("print("),
                "=" in stripped and "(" in stripped,
            ]):
                continue

            cleaned.append(line)

        text = "\n".join(cleaned)

    return text.strip()


def _run_generation(model, tokenizer, prompt, max_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens or max_new_tokens(),
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
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

    # 🔹 Mode-based steering
    if mode in ["explain", "chat"]:
        temperature = min(temperature, 0.3)

        if mode == "explain":
            prompt += "\nExplain only in plain English. No code examples."
        elif mode == "chat":
            prompt += "\nRespond in natural language only. No programming syntax."

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

    # 🔹 First generation
    text = _run_generation(model, tokenizer, prompt, max_tokens, temperature)

    # 🔹 Stop-word cleanup
    for stop in stop_words:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
            break

    # 🔹 Remove code if not allowed
    text = remove_code_if_not_allowed(text, mode)

    # 🔥 Retry if output is empty (NO HARDCODING)
    if not text.strip() and mode in ["chat", "explain"]:

        retry_prompt = prompt + "\nIMPORTANT: Answer only in plain English text. Do not use code."

        text = _run_generation(model, tokenizer, retry_prompt, max_tokens, 0.2)

        for stop in stop_words:
            idx = text.find(stop)
            if idx != -1:
                text = text[:idx]
                break

        text = remove_code_if_not_allowed(text, mode)

    return text.strip()