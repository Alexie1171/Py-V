"""
file_update.py — PY-V (inference/engine/)
Where a code block from V's answer goes in the open file (chat panel, Phase
10.3). The panel's "Apply to file" sends the code + the file as it is now; this
works out the whole file with the change. The panel shows it as a diff and
writes it only when the user clicks Apply — never on its own.

In this order:
  1. selection — the answer read a selection that is still in the file → the
     code replaces it (re-indented to where the selection sits)
  2. whole file — the code is most of the file (at least WHOLE_LINES of its
     lines and WHOLE_NAMES of its definitions) → it replaces the file
  3. named blocks — the code defines functions / classes / methods the file
     already has → each replaces its namesake, decorators and all; new ones go
     after the last one replaced; imports the file lacks go to its imports.
     Python by syntax tree (exact); other languages by definition lines and
     braces / indentation
  4. otherwise → added after the cursor line
Example lines in the code block ("print(add(2, 3))" after a fixed function)
are left out in 3 — the diff shows what goes in.
"""

import ast
import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

WHOLE_LINES = 0.6   # the code has at least this share of the file's lines ...
WHOLE_NAMES = 0.6   # ... and of its top-level definitions → a whole-file answer

# A definition line in most languages: keyword + name, or a C-like signature
_DEF_KEYWORD = re.compile(
    r"^\s*(?:(?:export|public|private|protected|internal|static|async|abstract|final|override|virtual|inline|"
    r"pub(?:\([\w:]+\))?|default|extern|unsafe|const|open|sealed|data|suspend|local|partial)\s+)*"
    r"(?:function\*?|def|fn|func|fun|sub|proc|procedure|class|struct|interface|enum|impl|trait|module|object|"
    r"record|type|union|macro|defmacro|defn|defun|method)\s+(?:\([^)]*\)\s*)?([A-Za-z_$][\w$]*)")
_DEF_ASSIGN  = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?"
                          r"(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)")
_DEF_SIGNATURE = re.compile(r"^\s*(?:[\w:<>\[\],*&.]+\s+)+\**([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:const\s*)?"
                            r"(?:noexcept\s*)?(?:->\s*[\w:<>*&]+\s*)?\{?\s*$")
_NOT_NAMES = {"if", "for", "while", "switch", "catch", "return", "else", "new", "sizeof", "do", "try"}
# A line starting with one of these is a statement, never a signature ("return foo(x)", "else if (a)")
_STATEMENTS = {"return", "else", "if", "while", "for", "switch", "case", "throw", "new", "await", "yield", "print",
               "echo", "delete", "goto", "do", "not", "and", "or", "raise", "assert", "using", "import", "include"}


@dataclass
class Proposal:
    content: str                 # the whole file with the change
    how:     str                 # selection / whole / blocks / insert
    summary: str                 # for the panel: "replaces add() and mul()"
    names:   List[str] = field(default_factory=list)


def propose(content: str, code: str, language_id: str = "", selection: str = "", selection_line: int = 0,
            cursor_line: int = 0) -> Proposal:
    """The file with `code` put in (see the module docstring). Raises ValueError for an empty code block."""
    if not code.strip():
        raise ValueError("The code block is empty.")
    eol  = "\r\n" if "\r\n" in content else "\n"
    text = content.replace("\r\n", "\n")
    code = textwrap.dedent(code.replace("\r\n", "\n")).strip("\n")
    python = language_id == "python"

    result = (_into_selection(text, code, selection, selection_line)
              or _whole_file(text, code, python)
              or (_python_blocks(text, code) if python else None)
              or (_generic_blocks(text, code) if not python else None)
              or _insert(text, code, cursor_line))
    result.content = result.content.replace("\n", eol)
    return result


# ─── 1. the selection ─────────────────────────────────────────────────────────

def _into_selection(text: str, code: str, selection: str, selection_line: int) -> Optional[Proposal]:
    selection = (selection or "").replace("\r\n", "\n")
    if not selection.strip():
        return None
    starts, at = [], text.find(selection)
    while at >= 0 and len(starts) < 50:
        starts.append(at)
        at = text.find(selection, at + 1)
    if not starts:
        return None                           # the selected code changed since the answer
    # the copy nearest to where it was selected (it may have moved)
    start = min(starts, key=lambda s: abs(text.count("\n", 0, s) + 1 - selection_line)) if selection_line else starts[0]

    line_start = text.rfind("\n", 0, start) + 1
    indent     = _indent_at(text, line_start)
    body       = _reindent(code, indent)
    if start > line_start:                    # starts mid-line: that line's indent is already there
        body = body[len(indent):]
    if selection.endswith("\n"):
        body += "\n"
    first_line = text.count("\n", 0, start) + 1
    last       = first_line + selection.rstrip("\n").count("\n")
    where = f"line {first_line}" if last == first_line else f"lines {first_line}-{last}"
    return Proposal(text[:start] + body + text[start + len(selection):], "selection",
                    f"replaces your selection ({where})")


def _indent_at(text: str, line_start: int) -> str:
    end  = text.find("\n", line_start)
    line = text[line_start:end if end >= 0 else len(text)]
    return line[:len(line) - len(line.lstrip(" \t"))]


def _reindent(code: str, indent: str) -> str:
    return "\n".join(indent + line if line.strip() else line for line in code.split("\n"))


# ─── 2. the whole file ────────────────────────────────────────────────────────

def _whole_file(text: str, code: str, python: bool) -> Optional[Proposal]:
    file_lines = [l for l in text.split("\n") if l.strip()]
    code_lines = [l for l in code.split("\n") if l.strip()]
    if len(file_lines) < 3 or len(code_lines) < WHOLE_LINES * len(file_lines):
        return None
    file_names = set(_top_names(text, python))
    code_names = set(_top_names(code, python))
    if file_names:
        if len(file_names & code_names) < WHOLE_NAMES * len(file_names):
            return None
    elif code_lines[0].strip() != file_lines[0].strip():
        return None
    ending = "\n" if text.endswith("\n") else ""
    return Proposal(code + ending, "whole", "replaces the whole file", sorted(code_names))


def _top_names(text: str, python: bool) -> List[str]:
    if python:
        try:
            tree = ast.parse(text)
            return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        except SyntaxError:
            pass
    lines = text.split("\n")
    level = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return [name for i, name in _definitions(lines) if len(lines[i]) - len(lines[i].lstrip()) == level]


# ─── 3. named blocks ──────────────────────────────────────────────────────────

def _python_blocks(text: str, code: str) -> Optional[Proposal]:
    try:
        code_tree = ast.parse(code)
        file_tree = ast.parse(text)
    except SyntaxError:
        return _generic_blocks(text, code)
    new_defs = [n for n in code_tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if not new_defs:
        return None

    targets = {}   # name → (start line, end line, indent) in the file, 1-based inclusive
    for node in ast.walk(file_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            targets.setdefault(node.name, (start, node.end_lineno, node.col_offset))
    code_lines = code.split("\n")
    blocks = []
    for node in new_defs:
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        blocks.append((node.name, "\n".join(code_lines[start - 1:node.end_lineno])))
    have    = {ast.unparse(n) for n in file_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))}
    missing = ["\n".join(code_lines[n.lineno - 1:n.end_lineno]) for n in code_tree.body
               if isinstance(n, (ast.Import, ast.ImportFrom)) and ast.unparse(n) not in have]
    last_import = max((n.end_lineno for n in file_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))), default=0)
    if not last_import and file_tree.body and isinstance(file_tree.body[0], ast.Expr) \
            and isinstance(getattr(file_tree.body[0], "value", None), ast.Constant):
        last_import = file_tree.body[0].end_lineno        # no imports yet: after the module docstring
    return _replace_blocks(text, blocks, targets, missing, last_import)


def _generic_blocks(text: str, code: str) -> Optional[Proposal]:
    code_lines = code.split("\n")
    file_lines = text.split("\n")
    level  = min((len(l) - len(l.lstrip()) for l in code_lines if l.strip()), default=0)
    blocks = []
    for i, name in _definitions(code_lines):
        if len(code_lines[i]) - len(code_lines[i].lstrip()) != level:
            continue
        start = _with_leading(code_lines, i)
        end   = _block_end(code_lines, i)
        blocks.append((name, "\n".join(code_lines[start:end + 1])))
    if not blocks:
        return None
    targets = {}
    for i, name in _definitions(file_lines):
        if name not in targets:
            indent = len(file_lines[i]) - len(file_lines[i].lstrip())
            targets[name] = (_with_leading(file_lines, i) + 1, _block_end(file_lines, i) + 1, indent)
    return _replace_blocks(text, blocks, targets, [], 0)


def _replace_blocks(text: str, blocks: list, targets: dict, missing_imports: list, last_import: int) -> Optional[Proposal]:
    """blocks: [(name, code)]; targets: name → (start, end, indent) 1-based inclusive in the file."""
    found = [(name, body) for name, body in blocks if name in targets]
    if not found:
        return None
    lines   = text.split("\n")
    edits   = []   # (start, end, new lines) 0-based, end exclusive
    after, after_indent = 0, 0
    for name, body in found:
        start, end, indent = targets[name]
        edits.append((start - 1, end, _reindent(textwrap.dedent(body), " " * indent).split("\n")))
        if end > after:
            after, after_indent = end, indent
    extra = [(name, body) for name, body in blocks if name not in targets]
    if extra:   # new ones after the last one replaced, at its indent (methods stay in their class)
        gap, added = [""] * (2 if after_indent == 0 else 1), []
        for _, body in extra:
            added += gap + _reindent(textwrap.dedent(body), " " * after_indent).split("\n")
        edits.append((after, after, added))
    if missing_imports:
        new = "\n".join(missing_imports).split("\n")
        edits.append((last_import, last_import, new + ([""] if last_import == 0 else [])))
    # ranges must not overlap (a method and its class both named)
    edits.sort(key=lambda e: (e[0], e[1]))
    kept = []
    for edit in edits:
        if kept and edit[0] < kept[-1][1]:
            continue
        kept.append(edit)
    for start, end, new in sorted(kept, reverse=True):
        lines[start:end] = new

    names = [f"{name}()" for name, _ in found]
    summary = "replaces " + _join(names)
    if extra:
        summary += " and adds " + _join([f"{name}()" for name, _ in extra])
    if missing_imports:
        summary += f", adds {len(missing_imports)} import{'s' if len(missing_imports) > 1 else ''}"
    return Proposal("\n".join(lines), "blocks", summary, [n for n, _ in blocks])


def _definitions(lines: List[str]) -> List[Tuple[int, str]]:
    """(line index, name) of every definition line."""
    found = []
    for i, line in enumerate(lines):
        match = _DEF_KEYWORD.match(line) or _DEF_ASSIGN.match(line)
        if not match and (line.split() or [""])[0] not in _STATEMENTS:
            match = _DEF_SIGNATURE.match(line)
        if match and match.group(1) not in _NOT_NAMES and not line.rstrip().endswith(";"):
            found.append((i, match.group(1)))
    return found


def _with_leading(lines: List[str], i: int) -> int:
    """Start of a definition including its decorators / annotations / doc comments right above it."""
    start = i
    while start > 0 and re.match(r"^\s*(@|#\[|///|/\*\*|\*|//)", lines[start - 1]):
        start -= 1
    return start


def _block_end(lines: List[str], i: int) -> int:
    """Last line (0-based) of the block defined at line i: brace matching when it opens with "{",
    else indentation (Python-like; a closing "end" at the same indent included)."""
    indent = len(lines[i]) - len(lines[i].lstrip())
    opens  = i if "{" in _no_strings(lines[i]) else \
        i + 1 if i + 1 < len(lines) and lines[i + 1].strip().startswith("{") else None   # Allman style
    if opens is not None:
        depth = 0
        for k in range(opens, len(lines)):
            clean  = _no_strings(lines[k])
            depth += clean.count("{") - clean.count("}")
            if depth <= 0 and "}" in clean:
                return k
        return len(lines) - 1
    end = i
    for k in range(i + 1, len(lines)):
        if not lines[k].strip():
            continue
        if len(lines[k]) - len(lines[k].lstrip()) <= indent:
            if re.match(r"^\s*(end\b|\}|\)|fi\b|done\b|esac\b)", lines[k]):
                end = k
            break
        end = k
    return end


def _no_strings(line: str) -> str:
    return re.sub(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|//.*$|#.*$", "", line)


def _join(items: List[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ─── 4. at the cursor ─────────────────────────────────────────────────────────

def _insert(text: str, code: str, cursor_line: int) -> Proposal:
    lines = text.split("\n")
    at    = min(max(cursor_line, 0), len(lines))
    if at == 0 and not text.strip():
        return Proposal(code + "\n", "insert", "fills the empty file")
    block = ([""] if at and lines[at - 1].strip() else []) + code.split("\n") + [""]
    lines[at:at] = block
    return Proposal("\n".join(lines), "insert", f"adds it after line {at}" if at else "adds it at the top")
