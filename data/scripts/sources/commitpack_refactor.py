"""
commitpack_refactor.py — PY-V Data Pipeline v2 (data/scripts/sources/)
CommitPackFT (Python) → "improve this code" (refactor) records from real
commits. Keeps commits whose subject says refactor / simplify / clean up /
improve / optimise / readability, that change exactly one function (same set
of functions before and after), where both versions are short and the change
is real — not only comments, docstrings or whitespace.
Instruction = request + old function; output = new function.
Nothing is executed. License: per row (CommitPackFT keeps permissive repos).
"""

import ast
import copy
import difflib
import re
import textwrap
from collections import Counter

from datasets import load_dataset

from data.scripts.cleaner import is_mostly_ascii
from data.scripts.sources.common import IMPROVE_TEMPLATES, make_record, pick

SOURCE = "commitpack_refactor"

_REFACTOR_SUBJECT = re.compile(
    r"\b(refactor\w*|simplif\w*|clean(ed|s|ing)?[ -]?up|cleaner|tidy|tidied|improv\w*|"
    r"optimi[sz](e|es|ed|ing|ation)|readab\w*|pythonic|streamlin\w*|redundan\w*)\b",
    re.I,
)
# Bug fixes and text-only changes are not "improve this code"
_SKIP_SUBJECT = re.compile(
    r"\b(revert\w*|merge\w*|bump\w*|release\w*|typos?|docs?|readme|comments?|tests?|"
    r"fix\w*|bugs?|messages?|help|wording|strings?|logging|log)\b",
    re.I,
)
_SKIP_FILE    = re.compile(r"(^|/)(tests?/|test_[^/]*$|setup\.py$|conftest\.py$|docs?/|migrations/)", re.I)


def _functions(code: str) -> dict:
    """Top-level functions and class methods: qualified name → (node, dedented source)."""
    tree  = ast.parse(code)
    lines = code.split("\n")
    found = {}

    def add(node, name):
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        found[name] = (node, textwrap.dedent("\n".join(lines[start - 1:node.end_lineno])))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, node.name)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(item, f"{node.name}.{item.name}")
    return found


class _BlankStrings(ast.NodeTransformer):
    def visit_Constant(self, node):
        return ast.Constant(value="") if isinstance(node.value, str) else node


def _shape(node, ignore_strings: bool = False) -> str:
    """
    AST of a function without its docstring — equal shapes mean no real change.
    With ignore_strings, all string literals count as equal (text-only edits).
    """
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if ignore_strings:
        body = [_BlankStrings().visit(copy.deepcopy(stmt)) for stmt in body]
    return "".join(ast.dump(stmt) for stmt in body) + ast.dump(node.args)


def iter_records(cfg: dict, stats):
    rows = load_dataset("json", data_files=cfg["data_files"], split="train", streaming=True)
    min_lines, max_lines = cfg["code_lines"]
    min_sim, max_sim     = cfg["similarity"]
    per_repo = Counter()

    for row in rows:
        stats["scanned"] += 1
        subject, repo = row["subject"], row["repos"].split(",")[0]

        if not _REFACTOR_SUBJECT.search(subject) or _SKIP_SUBJECT.search(subject):
            stats["rejected: not a refactor commit"] += 1
            continue
        if _SKIP_FILE.search(row["new_file"]):
            stats["rejected: test/setup/docs file"] += 1
            continue
        if per_repo[repo] >= cfg["max_per_repo"]:
            stats["rejected: repository cap reached"] += 1
            continue

        try:
            old_funcs, new_funcs = _functions(row["old_contents"]), _functions(row["new_contents"])
        except SyntaxError:
            stats["rejected: does not parse (Python 2?)"] += 1
            continue
        if set(old_funcs) != set(new_funcs):
            stats["rejected: functions added or removed"] += 1
            continue

        changed = [name for name in old_funcs if _shape(old_funcs[name][0]) != _shape(new_funcs[name][0])]
        if len(changed) != 1:
            stats["rejected: not exactly one function changed"] += 1
            continue

        name = changed[0]
        if _shape(old_funcs[name][0], ignore_strings=True) == _shape(new_funcs[name][0], ignore_strings=True):
            stats["rejected: only text/strings changed"] += 1
            continue

        old, new = old_funcs[name][1], new_funcs[name][1]
        if not all(min_lines <= len([l for l in c.split("\n") if l.strip()]) <= max_lines for c in (old, new)):
            stats["rejected: function too short/long"] += 1
            continue

        similarity = difflib.SequenceMatcher(None, old.split("\n"), new.split("\n")).ratio()
        if not min_sim <= similarity <= max_sim:
            stats["rejected: change too small/large"] += 1
            continue

        if not (is_mostly_ascii(old) and is_mostly_ascii(new)):
            stats["rejected: non-English text"] += 1
            continue

        per_repo[repo] += 1
        yield make_record(
            pick(IMPROVE_TEMPLATES, row["commit"]).format(code=old),
            new,
            SOURCE, "refactor", row["license"],
            repository=repo, commit=row["commit"], subject=subject, function=name,
        )
