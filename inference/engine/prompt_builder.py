from model.training.config_loader import CFG
from inference.engine.prompt_templates import TEMPLATES

# Modes that receive RAG context — kept in sync with config, but also
# checked here so prompt_builder stays self-contained.
_RAG_MODES = {"generate", "debug", "refactor"}


def build_prompt(
    mode:               str,
    user_input:         str,
    context:            dict,
    retrieved_chunks:   list = None,
) -> str:
    """
    Build the final prompt string for the model.

    retrieved_chunks is a list of dicts returned by Retriever.search(),
    each with a 'metadata' key containing at least a 'content' field.
    Only injected for modes in _RAG_MODES.
    """
    template = TEMPLATES.get(mode, TEMPLATES["chat"])

    # History context — off for all modes until Phase 11 (long-term memory).
    # explain/chat never got it (code bias); code modes answered the previous
    # question instead of the new one. format_context() is kept for Phase 11.
    formatted_context = ""

    # RAG context — only injected for relevant modes
    if mode in _RAG_MODES and retrieved_chunks:
        formatted_retrieved = format_retrieved_context(retrieved_chunks)
    else:
        formatted_retrieved = ""

    # explain and chat templates don't have a {retrieved_context} slot,
    # so we only pass it for the modes that need it.
    if mode in _RAG_MODES:
        return template.format(
            user_input         = user_input,
            context            = formatted_context,
            retrieved_context  = formatted_retrieved,
        )
    else:
        return template.format(
            user_input = user_input,
            context    = formatted_context,
        )


def build_training_prompt(mode: str, instruction: str) -> str:
    """
    Prompt part of a training example: exactly what inference sends for this
    mode (no history, no RAG). The answer + end-of-text token are appended by
    model/training/dataset_loader.py.
    """
    return build_prompt(mode, instruction, {})


def build_inference_prompt(instruction: str) -> str:
    """Prompt for the stateless /generate endpoint — the generate template, as in training."""
    return build_prompt("generate", instruction, {})


def format_retrieved_context(chunks: list) -> str:
    """
    Format retrieved RAG chunks into a concise context block.
    Each chunk contributes its content snippet, capped to keep
    prompt length manageable on GTX 1650.
    """
    if not chunks:
        return ""

    MAX_CHARS_PER_CHUNK = 300
    MAX_TOTAL_CHARS     = 800

    lines   = ["Relevant examples from codebase:"]
    total   = 0

    for i, chunk in enumerate(chunks):
        # Content is stored at metadata["content"] by the indexer
        content = (
            chunk.get("metadata", {}).get("content", "")
            or chunk.get("content", "")
        ).strip()

        if not content:
            continue

        snippet = content[:MAX_CHARS_PER_CHUNK]
        if len(content) > MAX_CHARS_PER_CHUNK:
            snippet += "..."

        entry = f"\n[{i+1}]\n{snippet}"

        if total + len(entry) > MAX_TOTAL_CHARS:
            break

        lines.append(entry)
        total += len(entry)

    if len(lines) == 1:   # only the header, nothing added
        return ""

    return "\n".join(lines) + "\n\n"


def format_context(ctx: dict) -> str:
    """
    Last two user questions only. Previous assistant answers are left out:
    the model copied them instead of answering the new question.
    """
    if not ctx or not ctx.get("history"):
        return ""

    questions = [
        (h.get("content") or "").strip()[:120]
        for h in ctx["history"]
        if h.get("role") == "user"
    ][-2:]

    if not questions:
        return ""

    formatted = ["Recent context:"]
    for q in questions:
        formatted.append(f"User previously asked: {q}")

    return "\n".join(formatted)


def max_new_tokens() -> int:
    return CFG.model.max_tokens