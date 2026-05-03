from model.training.config_loader import CFG
from inference.engine.prompt_templates import TEMPLATES

def build_prompt(mode: str, user_input: str, context: dict) -> str:
    template = TEMPLATES.get(mode, TEMPLATES["chat"])
    
    # Don't inject history for explain/chat — it biases Phi-2 toward code
    if mode in ["explain", "chat"]:
        formatted_context = ""
    else:
        formatted_context = format_context(context)

    return template.format(
        user_input=user_input,
        context=formatted_context,
    )


def build_training_prompt(instruction: str, output: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Answer:\n{output}"

def build_inference_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Answer:\n"


def format_context(ctx: dict) -> str:
    if not ctx or not ctx.get("history"):
        return ""

    formatted = ["Recent context:"]
    for h in ctx.get("history", [])[-3:]:
        role = h.get("role")
        content = (h.get("content") or "").strip()[:120]
        if role == "user":
            formatted.append(f"User previously asked: {content}")
        elif role == "assistant":
            formatted.append(f"Assistant responded: {content}")

    return "\n".join(formatted)


def max_new_tokens() -> int:
    return CFG.model.max_tokens