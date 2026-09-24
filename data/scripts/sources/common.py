"""
common.py — PY-V Data Pipeline v2 (data/scripts/sources/)
Shared helpers for turning Hugging Face rows into PY-V training records:
record builder, fenced-code extraction, demo-code trimming, docstring
removal. No source-specific logic here.
"""

import ast
import hashlib
import re

# ```lang\n ... ```
_FENCE = re.compile(r"```[ \t]*([\w+#-]*)[ \t]*\n(.*?)```", re.DOTALL)

# A code block plus the lead-in line right before it ("Here's how:")
_FENCE_WITH_LEAD_IN = re.compile(
    r"(?m)(?:^[^\n]*:[ \t]*\n(?:[ \t]*\n)?)?```[^\n]*\n(?s:.*?)```"
)

# Top-level lines that start demo / test code after the real answer
_DEMO_STARTS = (
    "# Example", "# example", "# Test", "# test", "# Sample", "# sample",
    "# Usage", "# usage", "if __name__", "print(", "assert ",
)


# Ways a user asks for "improve this code" (refactor records)
IMPROVE_TEMPLATES = [
    "Improve this code:\n\n{code}",
    "Can you clean up this function and make it more Pythonic?\n\n{code}",
    "Refactor this to be simpler and easier to read:\n\n{code}",
    "This works but looks clumsy. Please improve it:\n\n{code}",
]


def pick(options: list, key: str):
    """Stable choice from `options` based on `key` (same key → same choice)."""
    return options[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(options)]


def make_record(instruction: str, output: str, source: str, task: str,
                license: str, **extra) -> dict:
    """PY-V dataset format + task label (generate/debug/refactor/explain)."""
    return {
        "instruction": instruction.strip(),
        "output":      output.strip(),
        "metadata":    {"source": source, "task": task, "license": license, **extra},
    }


def code_blocks(text: str) -> list:
    """All fenced blocks as (language, code) pairs; language lower-cased."""
    return [(lang.lower(), code) for lang, code in _FENCE.findall(text)]


def first_python_block(text: str):
    """Code of the first ```python (or untagged) block, or None."""
    for lang, code in code_blocks(text):
        if lang in ("python", "py", "python3", ""):
            return code
    return None


def strip_code_blocks(text: str) -> str:
    """
    Remove fenced blocks and their "Here's how:" lead-in lines. Any other line
    left ending in ":" that isn't followed by a list is a lead-in to removed
    code too, so it goes as well.
    """
    text  = _FENCE_WITH_LEAD_IN.sub("", text)
    lines = text.split("\n")
    kept  = []

    for i, line in enumerate(lines):
        if line.rstrip().endswith(":"):
            following = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
            if not re.match(r"^([-*•]|\d+[.)])\s", following):
                continue
        kept.append(line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def trim_demo_code(code: str) -> str:
    """Cut example-usage / test code that follows the real answer."""
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if i > 0 and line.startswith(_DEMO_STARTS):
            return "\n".join(lines[:i]).rstrip()
    return code.rstrip()


def strip_docstring(code: str):
    """
    Source of a single function/class without its docstring.
    None if it doesn't parse, isn't a def/class, or is only a docstring.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    if not tree.body or not isinstance(
        tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        return None

    node  = tree.body[0]
    first = node.body[0]
    has_docstring = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    if not has_docstring:
        return code
    if len(node.body) == 1:
        return None

    lines = code.split("\n")
    del lines[first.lineno - 1:first.end_lineno]
    return "\n".join(lines)
