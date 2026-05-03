import torch
from inference.engine.prompt_builder import max_new_tokens


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

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens or max_new_tokens(),
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    for stop in stop_words:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
            break

    return text.strip()