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
    memories:           dict = None,
) -> str:
    """
    Build the final prompt string for the model.

    retrieved_chunks is a list of dicts returned by Retriever.search(),
    each with a 'metadata' key containing at least a 'content' field.
    Only injected for modes in _RAG_MODES.

    memories is MemoryManager.recall() output (Phase 11) — formatted into the
    template's existing {context} slot, so the templates (and the adapters
    trained on them) stay unchanged.
    """
    template = TEMPLATES.get(mode, TEMPLATES["chat"])

    # Raw chat history stays off (the model answered the previous question
    # instead of the new one); the context slot carries short memory facts,
    # plus earlier code in code modes (Phase 11).
    formatted_context = format_memories(memories, mode) if memories else ""

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


def format_memories(memories: dict, mode: str) -> str:
    """
    Memory block for the prompt's {context} slot: short facts in every mode,
    earlier code answers only in the code modes (never in explain/chat — same
    code-bias reason as RAG). Stays within CFG.memory.max_prompt_tokens
    (~4 characters per token); best matches first.
    """
    budget = CFG.memory.max_prompt_tokens * 4
    lines  = []

    facts = memories.get("facts") or []
    if facts:
        lines.append("Things V remembers about the user and the project:")
        for hit in facts:
            line = f"- {hit.text}"
            if sum(len(l) + 1 for l in lines) + len(line) > budget:
                break
            lines.append(line)
        if len(lines) == 1:
            lines = []

    code = (memories.get("code") or []) if mode in CFG.memory.active_code_modes else []
    for hit in code:
        block = f"Code V wrote earlier for a similar request:\n{_code_part(hit.text)}"
        if sum(len(l) + 1 for l in lines) + len(block) > budget:
            break
        lines.append(block)

    return ("\n".join(lines) + "\n") if lines else ""


def _code_part(text: str) -> str:
    """The first fenced block of an earlier answer, else the answer itself."""
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            body = parts[1].split("\n", 1)
            return "```python\n" + (body[1] if len(body) > 1 else body[0]).strip() + "\n```"
    return text.strip()


def build_training_prompt(mode: str, instruction: str) -> str:
    """
    Prompt part of a training example: exactly what inference sends for this
    mode (no history, no RAG). The answer + end-of-text token are appended by
    model/training/dataset_loader.py.
    """
    return build_prompt(mode, instruction, {})


def to_native_chat(prompt: str, tokenizer) -> str:
    """
    Re-wrap a template prompt in the model's own chat format, for chat-tuned
    brains (e.g. Granite 4.2) — thinking mode off. The "### Instruction:" /
    "### Answer:" markers are dropped and the instruction text becomes the
    user message. Base models have no chat template: the prompt is returned
    unchanged.
    """
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    body = prompt.removeprefix("### Instruction:\n").rsplit("### Answer:", 1)[0].strip()
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": body}],
        tokenize              = False,
        add_generation_prompt = True,
        enable_thinking       = False,
    )


def format_for_model(prompt: str, model, tokenizer) -> str:
    """
    The prompt exactly as the loaded model receives it: V's template, or
    re-wrapped in the brain's own chat format when the model was loaded with
    prompt format "native_chat" (set by the loaders as model.v_prompt_format —
    from the adapter's v_adapter.json, else config model.prompt_format).
    """
    if getattr(model, "v_prompt_format", CFG.model.prompt_format) == "native_chat":
        return to_native_chat(prompt, tokenizer)
    return prompt


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