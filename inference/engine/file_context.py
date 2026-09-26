"""
file_context.py — PY-V (inference/engine/)
The file open in the editor, for the chat panel (Phase 10.2): which part of it
goes into the prompt, and how the brain sees it.

The panel sends the open file with every message (name, VS Code's languageId,
its text, the selection, the cursor line) unless the user turned it off. What
goes in (not in chat mode — there only one line about the file, describe()):
  - the selection — selecting code is how the user points at it. In explain
    mode only when the message is about it ("what does this do", a word or
    name from it, a bare "explain"): "what is a decorator?" with some code
    still selected is a general question, and unrelated context derails the
    3B brain (the 2026-09-26 memory lesson). Not when the message brings its
    own code and doesn't point at the file
  - no selection: the file, when the message is about it ("fix this file",
    "what does this function do", "why does parse_config fail", its name) or
    when fix / improve has no code of its own (the code must come from somewhere)
  - otherwise nothing — code modes get a note with the file's name
A long file goes in pieces: top-level blocks (functions, classes, ...; a big
class splits into its methods) scored by the cursor, the words of the message
and the imports at the top, best first until the budget — machine WorkLimits
.file_chars, smaller when the computer is busy (Phase 12). Improve answers
rewrite what they got, so there only the code at the cursor (and pieces the
message names) goes in, no more than fits in one answer.
Left-out lines show as "... (lines 41-80 left out)".
The code goes after the message in a fenced block — the way pasted code
arrives and the way the long-file fix examples were trained ("Here is the
whole file:").
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from inference.engine.language_detector import file_language

# The message points at the open file
_ABOUT_FILE = re.compile(
    r"\b(this|that|these|those|my|the|current|open|above|below|selected|highlighted)\s+"
    r"(files?|code|script|module|functions?|methods?|class(es)?|program|snippet|lines?|loops?|bugs?|errors?|"
    r"parts?|block|tests?|component|query|queries)\b"
    r"|\b(in|from|of) here\b|\bwhat does this\b|\bthis one\b|\b(this|it)\s*[?.!]*\s*$", re.IGNORECASE)

# Loose pointing, for explain: "why is this slow?", "what does it return?"
_POINTS = re.compile(r"\b(this|these|it)\b", re.IGNORECASE)
# Names that look like code in the message: parse_config, parseConfig, add(), `total`
_CODE_NAME = re.compile(r"`([^`\n]+)`|\b([A-Za-z_]\w*)(?=\()|\b(\w*[a-z][A-Z]\w*|\w+_\w+)\b")

_CLOSER = re.compile(r"^([)\]}]|end\b|else\b|elif\b|except\b|finally\b|catch\b)")
_WORD   = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_COMMON = set("""
the and for with this that these those what does from into your you are can how why when where which there
their then than have has had not but all any about code file files function functions class method methods
line lines make write fix improve explain please want need show give tell help just also only here some more
most very it its is was were be been does did doing done use used using get set add run work works working
""".split())

CODE_BUDGET_CHARS_PER_TOKEN = 3   # improve answers rewrite the code: attach no more than fits in one answer
TINY_LINES = 3                    # a piece this short (class header, divider) joins the next one


@dataclass
class OpenFile:
    """The open file as the panel sends it (inference/api/schemas.OpenFile)."""
    name:           str
    language_id:    str = ""
    content:        str = ""
    first_line:     int = 1     # line number of content's first line (a huge file comes as a window)
    selection:      str = ""
    selection_line: int = 0     # first selected line, 1-based; 0 = nothing selected
    cursor_line:    int = 0     # 1-based; 0 = unknown

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["OpenFile"]:
        if not data or not data.get("name"):
            return None
        fields = {k: data[k] for k in cls.__dataclass_fields__ if data.get(k) is not None}
        return cls(**fields)

    @property
    def base_name(self) -> str:
        return re.split(r"[\\/]", self.name)[-1]


@dataclass
class Attached:
    """What of the file went into the prompt."""
    block: str   # appended to the message: intro line + fenced code
    label: str   # for the panel: "app.py, lines 10-24"


def about_file(message: str, file: OpenFile) -> bool:
    """
    The message points at the open file: "fix this function", "explain this",
    "what's wrong in app.py", or a code name that is in the file ("why does
    parse_config fail", "what does add() return").
    """
    if _ABOUT_FILE.search(message):
        return True
    if file.base_name and file.base_name.lower() in message.lower():
        return True
    text = file.selection + "\n" + file.content
    for match in _CODE_NAME.finditer(message):
        name = next(g for g in match.groups() if g).strip()
        if len(name) > 1 and re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", text):
            return True
    return False


def _about_text(message: str, text: str) -> bool:
    """Explain mode: the message is about this text — "this" / "it", a word from it, or
    too short to have a subject of its own ("explain", "hmm?", "why")."""
    if _POINTS.search(message) or len(message.split()) < 3:
        return True
    words = {w.lower() for w in _WORD.findall(message) if w.lower() not in _COMMON}
    if not words:
        return True
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in words)


def language_of(file: Optional[OpenFile], attached: bool, mode: str) -> Optional[dict]:
    """
    The file's language when it should decide the answer's language: always
    when its text is in the prompt; else a programming language (not a format)
    in the code modes ("write a function that ..." in a .ts file → TypeScript).
    None = V's default, Python.
    """
    if file is None:
        return None
    lang = file_language(file.language_id)
    if lang is None or (lang["format"] and not attached) or (not attached and mode not in ("generate", "debug", "refactor")):
        return None
    return {"name": lang["name"], "tag": lang["tag"]}


def attach(file: Optional[OpenFile], message: str, mode: str, own_code: bool, budget: int,
           piece_lines: int = 40, answer_tokens: int = 0, tag: str = "") -> Optional[Attached]:
    """
    The part of the open file for this message's prompt, or None (see the
    module docstring for when). budget = characters (WorkLimits.file_chars);
    answer_tokens = the answer's token limit, which caps what improve mode gets.
    tag = the code block's language tag.
    """
    if file is None or mode == "chat":
        return None
    pointed = about_file(message, file)
    if mode == "refactor" and answer_tokens:
        budget = min(budget, answer_tokens * CODE_BUDGET_CHARS_PER_TOKEN)
    surroundings = mode != "refactor"   # improve rewrites all it gets: no neighbours, no imports
    if budget <= 0:
        return None

    selection = file.selection.strip()
    if mode == "explain":
        use_selection = selection and (pointed or (not own_code and _about_text(message, file.selection)))
    else:
        use_selection = selection and (pointed or not own_code)

    if use_selection:
        first = max(1, file.selection_line or 1)
        focus = file.cursor_line - first if file.cursor_line >= first else 0
        code, shown = _cut(file.selection, first, focus, message, budget, piece_lines, surroundings)
        last = first + file.selection.rstrip("\n").count("\n")
        where = f"line {first}" if last == first else f"lines {first}-{last}"
        intro = f"Selected code from {file.name} ({where})"
        label = f"{file.base_name}, {where}" + ("" if shown == "all" else " (parts)")
    elif file.content.strip() and (pointed or (mode in ("debug", "refactor") and not own_code)):
        first = max(1, file.first_line or 1)
        focus = file.cursor_line - first if file.cursor_line >= first else None
        code, shown = _cut(file.content, first, focus, message, budget, piece_lines, surroundings)
        if shown == "all" and first == 1:
            intro, label = f"Here is the whole file {file.name}", f"{file.base_name} (whole file)"
        else:
            intro, label = f"Parts of the file {file.name} (the rest is left out)", f"{file.base_name}, {shown}"
    else:
        return None

    return Attached(block=f"{intro}:\n\n```{tag}\n{code}\n```", label=label)


def note(file: Optional[OpenFile]) -> str:
    """Code modes, file not attached: where the user works — for the prompt's context slot."""
    return f"The user is working in {file.name}.\n" if file else ""


def describe(file: Optional[OpenFile]) -> str:
    """Chat mode: one line about the open file for V's system message."""
    if file is None:
        return ""
    lang  = file_language(file.language_id)
    kind  = "Python" if file.language_id == "python" else (lang["name"] if lang else "")
    where = f"{file.name} ({kind})" if kind else file.name
    if file.selection.strip() and file.selection_line:
        last = file.selection_line + file.selection.rstrip("\n").count("\n")
        where += f", lines {file.selection_line}-{last} selected"
    return f"Open in the user's editor: {where}. You don't see its code in chat; they can ask you to fix, improve or explain it."


# ─── Long files in pieces ─────────────────────────────────────────────────────

def _cut(text: str, first_line: int, focus: Optional[int], message: str, budget: int,
         piece_lines: int, surroundings: bool = True) -> Tuple[str, str]:
    """
    (code, what was shown): the whole text if it fits the budget, else the best
    pieces in file order with markers for the left-out lines. What was shown:
    "all", or "lines 1-12, 40-80" (file line numbers).
    """
    text  = text.replace("\r\n", "\n").rstrip("\n")
    if len(text) <= budget:
        return text, "all"

    lines  = text.split("\n")
    pieces = split_pieces(lines, piece_lines)
    chosen = pick_pieces(lines, pieces, message, focus, budget, surroundings)

    spans = []   # side-by-side pieces as one span
    for a, b in chosen:
        if spans and spans[-1][1] == a:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))

    out, shown, pos = [], [], 0
    for a, b in spans:
        if a > pos:
            out.append(_gap(pos, a, first_line))
        out.extend(lines[a:b])
        shown.append(f"{a + first_line}-{b - 1 + first_line}" if b - a > 1 else f"{a + first_line}")
        pos = b
    if pos < len(lines):
        out.append(_gap(pos, len(lines), first_line))
    return "\n".join(out), "lines " + ", ".join(shown)


def _gap(a: int, b: int, first_line: int) -> str:
    return f"... (lines {a + first_line}-{b - 1 + first_line} left out)" if b - a > 1 \
        else f"... (line {a + first_line} left out)"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def split_pieces(lines: List[str], max_lines: int) -> List[Tuple[int, int]]:
    """
    Blocks of the file as (start, end) line ranges, end exclusive: a new block
    starts after a blank line at the outermost indent (a function, a class, a
    group of imports); a block longer than max_lines splits the same way one
    indent deeper (a class into its methods), and is cut every max_lines when
    it can't. A tiny block (a class header, a divider comment) joins the next
    one. Works for any language that separates blocks with blank lines.
    """
    pieces = _split(lines, 0, len(lines), max(1, max_lines), 0)
    joined = []
    for a, b in pieces:
        if joined and joined[-1][1] - joined[-1][0] <= TINY_LINES and b - joined[-1][0] <= max_lines:
            joined[-1] = (joined[-1][0], b)
        else:
            joined.append((a, b))
    return joined


def _split(lines, a, b, max_lines, depth):
    if b - a <= max_lines:
        return [(a, b)]
    body = [i for i in range(a + 1, b) if lines[i].strip() and not _CLOSER.match(lines[i].lstrip())]
    if depth < 4 and body:
        level = min(_indent(lines[i]) for i in body)   # a closing "}" doesn't set the level
        cuts  = [i for i in body if _indent(lines[i]) == level and not lines[i - 1].strip()]
        if cuts:
            edges, out = [a] + cuts + [b], []
            for s, e in zip(edges, edges[1:]):
                out += _split(lines, s, e, max_lines, depth + 1)
            return out
    return [(s, min(b, s + max_lines)) for s in range(a, b, max_lines)]


def pick_pieces(lines: List[str], pieces: List[Tuple[int, int]], message: str, focus: Optional[int],
                budget: int, surroundings: bool = True) -> List[Tuple[int, int]]:
    """
    The pieces that go in, in file order: the one at the cursor first, then
    the ones that define or use words from the message, the top of the file
    (docstring, imports), and the cursor's two neighbours on each side — until
    the budget. surroundings False (improve): only the cursor's piece and the
    ones the message names. Pieces with none of these reasons stay out. A best
    piece bigger than the whole budget is cut around the cursor.
    """
    words   = {w for w in _WORD.findall(message) if w.lower() not in _COMMON}
    at      = next((n for n, (a, b) in enumerate(pieces) if focus is not None and a <= focus < b), None)
    ranked  = []
    for n, (a, b) in enumerate(pieces):
        head  = next((lines[i] for i in range(a, b) if lines[i].strip()), "")
        body  = "\n".join(lines[a:b])
        score = 0.0
        if at is not None and n == at:
            score += 10.0
        elif at is not None and surroundings and abs(n - at) <= 2:
            score += 1.5 / abs(n - at)
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", head):
                score += 4.0
            elif re.search(rf"\b{re.escape(w)}\b", body):
                score += 1.0
        if n == 0 and surroundings:
            score += 2.0
        if score > 0:
            ranked.append((-score, n))
    ranked.sort()
    if not ranked:   # no cursor, no names (improve): from the top
        ranked = [(0.0, n) for n in range(len(pieces))]

    chosen, used = [], 0
    for _, n in ranked:
        a, b = pieces[n]
        size = sum(len(lines[i]) + 1 for i in range(a, b)) + 40   # + a gap marker
        if used + size <= budget:
            chosen.append((a, b))
            used += size
        elif not chosen:   # the best piece alone is too big: the lines around the cursor
            chosen.append(_around(lines, a, b, focus, budget))
            used = budget
    return sorted(chosen)


def _around(lines, a, b, focus, budget):
    """Lines of a..b around focus (or from a) that fit the budget."""
    lo = hi = min(max(focus if focus is not None else a, a), b - 1)
    used = len(lines[lo]) + 1
    grew = True
    while grew:
        grew = False
        if hi + 1 < b and used + len(lines[hi + 1]) + 1 <= budget:
            hi   += 1
            used += len(lines[hi]) + 1
            grew  = True
        if lo - 1 >= a and used + len(lines[lo - 1]) + 1 <= budget:
            lo   -= 1
            used += len(lines[lo]) + 1
            grew  = True
    return (lo, hi + 1)
