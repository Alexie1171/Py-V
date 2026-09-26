import re

from model.training.config_loader import CFG
from inference.engine.prompt_templates import (TEMPLATES, OTHER_LANGUAGE_TEMPLATES, INTENT_TEMPLATE, V_PERSONA,
                                               V_MACHINE, V_MACHINE_BUSY, V_LOOKUP, V_STUDIES, V_NOTES,
                                               STUDY_NOTE_TEMPLATE)

# Modes that receive RAG context — kept in sync with config, but also
# checked here so prompt_builder stays self-contained.
_RAG_MODES = {"generate", "debug", "refactor"}

_CODE_MODES    = {"generate", "debug", "refactor"}
HISTORY_CHARS  = 400   # per earlier message in chat mode (~100 tokens) — the laptop GPU slows down past ~1,500 prompt tokens


def build_prompt(
    mode:               str,
    user_input:         str,
    context:            dict,
    retrieved_chunks:   list = None,
    memories:           dict = None,
    language:           dict = None,
    file_note:          str  = "",
) -> str:
    """
    Build the final prompt string for the model.

    retrieved_chunks is a list of dicts returned by Retriever.search(),
    each with a 'metadata' key containing at least a 'content' field.
    Only injected for modes in _RAG_MODES.

    memories is MemoryManager.recall() output (Phase 11) — formatted into the
    template's existing {context} slot, so the templates (and the adapters
    trained on them) stay unchanged.

    language is language_detector.detect_language() output: another language
    than Python gets OTHER_LANGUAGE_TEMPLATES — memory facts only (earlier
    code is Python), no RAG (the index is Python).

    file_note = file_context.note() — which file the user works in (chat
    panel, Phase 10.2), ahead of the memory block in the {context} slot. The
    file's code itself, when it goes in, is already in user_input.
    """
    if language and mode in OTHER_LANGUAGE_TEMPLATES:
        return OTHER_LANGUAGE_TEMPLATES[mode].format(
            language   = language["name"],
            tag        = language["tag"],
            user_input = user_input,
            context    = file_note + (format_memories(memories, "explain") if memories else ""),
        )

    template = TEMPLATES.get(mode, TEMPLATES["chat"])

    # Raw chat history stays off (the model answered the previous question
    # instead of the new one); the context slot carries short memory facts,
    # plus earlier code in code modes (Phase 11).
    formatted_context = file_note + (format_memories(memories, mode) if memories else "")

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
    study notes (Phase 13, "notes": learning store rows) in every mode,
    earlier code answers only in the code modes (never in explain/chat — same
    code-bias reason as RAG). Stays within CFG.memory.max_prompt_tokens
    (~4 characters per token) — notes get their own CFG.learning.notes_chars;
    best matches first.
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

    notes = memories.get("notes") or []
    if notes:
        room, block = CFG.learning.notes_chars, []
        for n in notes:
            text = f"({n['topic']}, {n['subtopic']}) " + " ".join(l.lstrip("- ").strip() for l in n["text"].split("\n"))
            if block and sum(len(b) + 1 for b in block) + len(text) > room:
                break
            block.append(text[:room])
        lines.append(V_NOTES.format(notes="\n".join(f"- {b}" for b in block)))

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


def uses_chat_format(model, tokenizer) -> bool:
    """
    True when the loaded brain takes prompts in its own chat format: prompt
    format "native_chat" (set by the loaders as model.v_prompt_format — from the
    adapter's v_adapter.json, else config model.prompt_format) and a tokenizer
    with a chat template.
    """
    return (getattr(model, "v_prompt_format", CFG.model.prompt_format) == "native_chat"
            and bool(getattr(tokenizer, "chat_template", None)))


def format_for_model(prompt: str, model, tokenizer) -> str:
    """
    The prompt exactly as the loaded model receives it: V's template, or
    re-wrapped in the brain's own chat format (uses_chat_format).
    """
    if uses_chat_format(model, tokenizer):
        return to_native_chat(prompt, tokenizer)
    return prompt


def build_chat_prompt(user_input: str, context: dict, memories: dict, tokenizer,
                      history_turns: int = None, machine: str = "", editor: str = "", extra: str = "") -> str:
    """
    Chat mode for brains with their own chat format, as a real conversation:
    V's persona (+ what she knows about the computer, + remembered facts) as
    the system message, the last turns of this chat, then the message exactly
    as the user wrote it. Wrapping it in "Answer the following question using
    only plain English..." made V answer like homework. Returned in the brain's
    format — generate_from_prompt(..., formatted=True). Brains without a chat
    format use build_prompt("chat", ...). history_turns None = config;
    machine = format_machine() output; editor = file_context.describe() (the
    file open in the chat panel's editor — its name only, no code); extra =
    Phase 13 blocks (what she found online — format_lookup(); what she has
    studied — format_studies()), last, closest to the question.
    """
    system = V_PERSONA
    facts  = format_memories(memories, "chat") if memories else ""
    for block in (machine, editor, facts, extra):
        if block:
            system += "\n\n" + block.strip()

    messages = [{"role": "system", "content": system}]
    messages += chat_history(context, history_turns)
    messages.append({"role": "user", "content": user_input})
    return tokenizer.apply_chat_template(
        messages,
        tokenize              = False,
        add_generation_prompt = True,
        enable_thinking       = False,
    )


def chat_history(context: dict, turns: int = None) -> list:
    """
    The last `turns` messages of this chat as chat turns (chat mode only;
    None = CFG.memory.history_turns — fewer when the machine is busy). No code goes in, same reason as memory and RAG: a turn
    from a code mode keeps just the first line of the request, and V's code
    answer becomes a short note; code blocks elsewhere become "(code left
    out)". Each message is cut to HISTORY_CHARS.
    """
    turns  = CFG.memory.history_turns if turns is None else turns
    recent = ((context or {}).get("history") or [])[-turns:] if turns > 0 else []
    out    = []
    for turn in recent:
        role, text = turn.get("role"), (turn.get("content") or "").strip()
        if role not in ("user", "assistant") or not text:
            continue
        if turn.get("mode") in _CODE_MODES:
            if role == "assistant":
                text = "(I answered with code here, left out of this chat.)"
            else:
                first, _, rest = text.partition("\n")
                text = first.strip() + (" ..." if rest.strip() else "")
        else:
            text = re.sub(r"```.*?```", "(code left out)", text, flags=re.DOTALL)
        if len(text) > HISTORY_CHARS:
            text = text[:HISTORY_CHARS].rsplit(" ", 1)[0] + " ..."
        out.append({"role": role, "content": text})

    while out and out[0]["role"] == "assistant":   # a conversation starts with the user
        out.pop(0)
    return out


def format_lookup(found: str, sources: list) -> str:
    """What a web lookup found, for the chat system message (V_LOOKUP)."""
    names = ", ".join(s.get("title") or s.get("url", "") for s in sources) or "the web"
    return V_LOOKUP.format(found=found.strip(), sources=names)


def format_studies(topics: list, limit: int = 5) -> str:
    """What she has studied (learning store topics), for the chat system message (V_STUDIES)."""
    parts = []
    for t in topics[:limit]:
        part = f"{t['topic']} ({t['minutes']:g} min, {t.get('notes', 0)} notes"
        if t["covered"]:
            part += "; covered: " + ", ".join(t["covered"][:5])
        if t["next"]:
            part += "; not yet: " + ", ".join(t["next"][:3])
        parts.append(part + ")")
    return V_STUDIES.format(topics="; ".join(parts)) if parts else ""


def build_study_prompt(topic: str, subtopic: str, title: str, excerpt: str, covered: list, tokenizer) -> str:
    """One study note (learning/study.py): V's persona + STUDY_NOTE_TEMPLATE, in the brain's own chat
    format when it has one (pass formatted=True), else V's template format."""
    body = STUDY_NOTE_TEMPLATE.format(topic=topic, subtopic=subtopic, title=title or "a web page",
                                      covered=", ".join(covered) if covered else "nothing yet", excerpt=excerpt.strip())
    if not getattr(tokenizer, "chat_template", None):
        return f"### Instruction:\n{body}\n\n### Answer:\n"
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": V_PERSONA}, {"role": "user", "content": body}],
        tokenize              = False,
        add_generation_prompt = True,
        enable_thinking       = False,
    )


def format_machine(specs: dict, snap) -> str:
    """
    What V knows about the computer for the chat system message: its parts
    (machine.Machine.specs) and this answer's live numbers (machine.Snapshot).
    """
    parts = []
    if specs.get("gpu"):
        parts.append(f"{specs['gpu']} graphics card ({specs['gpu_gb']:.1f} GB)")
    parts.append(f"{specs['ram_gb']:.1f} GB of usable RAM")
    parts.append(f"{specs['cpu']} ({specs['cores']} cores, {specs['threads']} threads)")

    usage = [f"{snap.ram_free_gb:.1f} GB of RAM free"]
    if snap.gpu_free_gb is not None:
        usage.append(f"{snap.gpu_free_gb:.1f} GB of graphics memory free for V")
    usage.append(f"CPU at {snap.cpu_percent:.0f}%")

    return V_MACHINE.format(specs=", ".join(parts), usage=", ".join(usage),
                            busy=V_MACHINE_BUSY if snap.level != "free" else "")


def build_inference_prompt(instruction: str) -> str:
    """Prompt for the stateless /generate endpoint — the generate template, as in training."""
    return build_prompt("generate", instruction, {})


INTENT_MAX_CHARS = 1500   # of the message — enough to see what is asked, keeps the question quick


def build_intent_prompt(message: str) -> str:
    """Question to the brain: which mode does this message need? (intent_classifier.py)"""
    text = message if len(message) <= INTENT_MAX_CHARS else message[:INTENT_MAX_CHARS] + "\n..."
    return INTENT_TEMPLATE.format(message=text)


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