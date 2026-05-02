import torch


def generate_code(model, tokenizer, prompt: str, max_tokens: int = 100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.2
    )

    return tokenizer.decode(outputs[0])