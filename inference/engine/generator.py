import torch
import re
import threading
from contextlib import nullcontext
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from model.training.config_loader import CFG
from inference.engine.prompt_builder import max_new_tokens, format_for_model
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


# ─── Emojis ──────────────────────────────────────────────────────────────────
# Owner: no emojis. The persona says so; this catches the ones the brain adds
# anyway (chat / explain answers only — code is left alone).

_EMOJI = re.compile(r"[ \t]*[\U0001F000-\U0001FAFF☀-➿⭐⭕️‍]+")


def _strip_emojis(text: str) -> str:
    return _EMOJI.sub("", text)


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
    original = text.strip()

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

    # If barely anything survived code removal, the output was entirely code.
    # A short answer with nothing removed stays ("I'm V." — she answers only what is asked)
    if len(result) < 20 and result != original:
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
    # Echoes of the prompt's own RAG / history / memory blocks (see prompt_builder.py)
    "Relevant examples from codebase:",
    "Recent context:",
    "Things V remembers",
    "Code V wrote earlier",
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

def adapter_for_mode(model, mode):
    """Context in which the model answers a question of this mode: the LoRA
    adapter switched off when its v_adapter.json limits it to other modes
    ("use_in_modes" → model.v_adapter_modes, None = every mode). The Granite
    chat adapter helps fix / improve / long files but writes worse new code
    (MBPP 76 → 62), so generate / explain / chat run on the plain brain."""
    modes = getattr(model, "v_adapter_modes", None)
    if modes is None or mode in modes or not hasattr(model, "disable_adapter"):
        return nullcontext()
    return model.disable_adapter()


_SENTENCE_END = re.compile(r"""[.!?]["')\]]?(?=\s|$)""")


def _drop_unfinished(text: str) -> str:
    """A prose answer that ran out of tokens: cut after its last full sentence
    (kept as is when it has none)."""
    ends = list(_SENTENCE_END.finditer(text))
    return text[:ends[-1].end()] if ends else text


# One answer on the brain at a time: a second one at once would fill the 4 GB
# laptop GPU (the API serves requests in parallel threads; the panel streams).
BRAIN_LOCK = threading.Lock()


class _StopWhenSet(StoppingCriteria):
    """Stops the brain after its current word once the event is set (the Stop
    button, or the user closed the panel)."""

    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)


def _run_generation(model, tokenizer, prompt, max_tokens, temperature, mode=None, formatted=False,
                    streamer=None, stop=None):
    if not formatted:
        prompt = format_for_model(prompt, model, tokenizer)
    inputs   = tokenizer(prompt, return_tensors="pt").to(model.device)
    settings = CFG.generation.for_mode(mode)
    limit    = max_tokens or max_new_tokens()
    extra    = {}
    if streamer is not None:
        extra["streamer"] = streamer
    if stop is not None:
        extra["stopping_criteria"] = StoppingCriteriaList([_StopWhenSet(stop)])

    with BRAIN_LOCK, torch.inference_mode(), adapter_for_mode(model, mode):
        output_ids = model.generate(
            **inputs,
            max_new_tokens       = limit,
            do_sample            = temperature > 0,
            temperature          = temperature if temperature > 0 else 1.0,
            repetition_penalty   = settings.repetition_penalty,
            no_repeat_ngram_size = settings.no_repeat_ngram_size,
            stop_strings         = _stop_words_for(mode),
            tokenizer            = tokenizer,
            eos_token_id         = tokenizer.eos_token_id,
            pad_token_id         = tokenizer.eos_token_id,
            **extra,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    text    = tokenizer.decode(new_ids, skip_special_tokens=True)
    if mode in ("chat", "explain") and len(new_ids) >= limit:   # cut off by the token limit
        text = _drop_unfinished(text)
    return text


# ─── Public interface ─────────────────────────────────────────────────────────

def generate_from_prompt(
    model,
    tokenizer,
    prompt:      str,
    mode:        str   = None,
    max_tokens:  int   = None,
    temperature: float = None,
    formatted:   bool  = False,
) -> str:
    """
    temperature None = the mode's setting (config generation.<mode>.temperature).
    formatted = the prompt is already in the brain's own chat format
    (prompt_builder.build_chat_prompt) — V's templates are re-wrapped otherwise.
    """
    if temperature is None:
        temperature = CFG.generation.for_mode(mode).temperature

    text = _clean(_run_generation(model, tokenizer, prompt, max_tokens, temperature, mode, formatted), mode)

    # Retry at higher temperature if output is empty
    if not text.strip() and mode in ["chat", "explain"]:
        text = _clean(_run_generation(model, tokenizer, prompt, max_tokens, max(temperature, 0.5), mode, formatted), mode)

    return text.strip()


def stream_from_prompt(
    model,
    tokenizer,
    prompt:      str,
    mode:        str   = None,
    max_tokens:  int   = None,
    temperature: float = None,
    formatted:   bool  = False,
    stop:        threading.Event = None,
):
    """
    generate_from_prompt, streamed (the chat panel): yields ("piece", text) as
    the brain writes, then ("answer", text) — the cleaned answer, the same one
    generate_from_prompt gives. Raw pieces can hold bits the cleanup removes
    (a stop word, code in chat), so the caller replaces the streamed text with
    the answer. stop: set it and the brain stops after its current word; closing
    this generator early (the user left) stops it too.
    """
    if temperature is None:
        temperature = CFG.generation.for_mode(mode).temperature
    stop     = stop or threading.Event()
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    result   = {}

    def work():
        try:
            result["text"] = _run_generation(model, tokenizer, prompt, max_tokens, temperature, mode,
                                             formatted, streamer, stop)
        except Exception as e:   # handed to the caller below; ends the stream so it can't hang
            result["error"] = e
            streamer.end()

    thread   = threading.Thread(target=work, daemon=True)
    finished = False
    thread.start()
    try:
        for piece in streamer:
            if mode in ["chat", "explain"]:
                piece = _strip_emojis(piece)
            if piece:
                yield "piece", piece
        finished = True
    finally:
        if not finished:
            stop.set()
        thread.join()

    if "error" in result:
        raise result["error"]
    text = _clean(result["text"], mode)
    if not text.strip() and mode in ["chat", "explain"] and not stop.is_set():
        text = _clean(_run_generation(model, tokenizer, prompt, max_tokens, max(temperature, 0.5), mode,
                                      formatted, stop=stop), mode)
    yield "answer", text.strip()


def _clean(text: str, mode: str) -> str:
    text = _apply_stop_words(text, mode)
    text = _strip_prompt_echo(text)
    text = remove_code_if_not_allowed(text, mode)
    if mode in ["chat", "explain"]:
        text = _strip_emojis(text)
    return _strip_artifacts(text)


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