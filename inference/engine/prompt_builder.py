from model.training.config_loader import CFG
from inference.engine.prompt_templates import TEMPLATES


def build_prompt(mode: str, user_input: str, context: dict) -> str:
    template = TEMPLATES.get(mode, TEMPLATES["chat"])
    formatted_context = format_context(context)

    return template.format(
        user_input=user_input,
        context=formatted_context
    )


def format_context(ctx: dict) -> str:
    """
    Clean context format that avoids pattern continuation bias.
    """

    if not ctx or not ctx.get("history"):
        return ""

    formatted = ["Previous context (DO NOT continue patterns from this. It is only for reference):"]

    for h in ctx.get("history", [])[-3:]:
        role = h.get("role")
        content = (h.get("content") or "").strip()[:120]

        if role == "user":
            formatted.append(f"- User asked: {content}")
        elif role == "assistant":
            formatted.append(f"- Assistant responded: {content}")

    formatted.append("DO NOT imitate formatting style from context.")
    return "\n".join(formatted)


def max_new_tokens() -> int:
    return CFG.model.max_tokens