"""
mutations.py — PY-V Data Pipeline v2 (data/scripts/sources/)
Small, realistic bugs for "fix the error" training examples.

Each operator finds places in working code where a typical mistake fits and
returns the broken code plus a one-sentence explanation of the fix. Edits are
spliced into the original text using AST positions, so formatting and comments
stay exactly as they were — the fixed code is the original, untouched.
Pure text/AST work: nothing here runs code.
"""

import ast
import builtins
from dataclasses import dataclass


@dataclass
class Mutation:
    kind:        str
    buggy:       str
    explanation: str


class SourceText:
    """Original code + mapping from AST positions to text offsets."""

    def __init__(self, code: str):
        self.code   = code
        self.lines  = code.splitlines(keepends=True)
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))

    def offset(self, lineno: int, col: int) -> int:
        # AST columns are UTF-8 byte offsets
        line = self.lines[lineno - 1]
        return self.starts[lineno - 1] + len(line.encode("utf-8")[:col].decode("utf-8", "ignore"))

    def span(self, node) -> tuple:
        return (self.offset(node.lineno, node.col_offset),
                self.offset(node.end_lineno, node.end_col_offset))

    def text(self, node) -> str:
        start, end = self.span(node)
        return self.code[start:end]

    def replace(self, start: int, end: int, new: str) -> str:
        return self.code[:start] + new + self.code[end:]


def _one_line(node) -> bool:
    return node.lineno == node.end_lineno


def _swap_in_gap(src, node, left, right, old, new):
    """Replace operator `old` between two child nodes; returns (buggy, buggy_expr) or None."""
    _, gap_start = src.span(left)
    gap_end, _   = src.span(right)
    gap = src.code[gap_start:gap_end]
    if gap.strip() != old:
        return None
    buggy = src.replace(gap_start, gap_end, gap.replace(old, new, 1))
    start, end = src.span(node)
    return buggy, buggy[start:end + len(new) - len(old)]


# ─── Operators ────────────────────────────────────────────────────────────────

_COMPARE_FLIP = {
    ast.Lt: ("<", "<="), ast.LtE: ("<=", "<"),
    ast.Gt: (">", ">="), ast.GtE: (">=", ">"),
    ast.Eq: ("==", "!="), ast.NotEq: ("!=", "=="),
}


def _compare_flips(src, tree):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1
                and type(node.ops[0]) in _COMPARE_FLIP and _one_line(node)):
            continue
        old, new = _COMPARE_FLIP[type(node.ops[0])]
        swapped = _swap_in_gap(src, node, node.left, node.comparators[0], old, new)
        if swapped:
            buggy, buggy_expr = swapped
            yield Mutation("compare", buggy,
                           f"The condition `{buggy_expr}` uses the wrong comparison; "
                           f"it should be `{src.text(node)}`.")


_BINOP_SWAP = {
    ast.Add: ("+", "-"), ast.Sub: ("-", "+"),
    ast.FloorDiv: ("//", "/"), ast.Div: ("/", "//"),
}


def _binop_swaps(src, tree):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and type(node.op) in _BINOP_SWAP and _one_line(node)):
            continue
        old, new = _BINOP_SWAP[type(node.op)]
        swapped = _swap_in_gap(src, node, node.left, node.right, old, new)
        if swapped:
            buggy, buggy_expr = swapped
            yield Mutation("operator", buggy,
                           f"`{buggy_expr}` uses the wrong operator; it should be `{src.text(node)}`.")


def _range_off_by_one(src, tree):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "range" and 1 <= len(node.args) <= 3
                and not node.keywords and _one_line(node)):
            continue
        stop = node.args[0] if len(node.args) == 1 else node.args[1]
        stop_start, stop_end = src.span(stop)
        stop_text = src.code[stop_start:stop_end]
        call_start, call_end = src.span(node)
        for delta in (" + 1", " - 1"):
            buggy = src.replace(stop_start, stop_end, stop_text + delta)
            buggy_call = buggy[call_start:call_end + len(delta)]
            yield Mutation("off_by_one", buggy,
                           f"The loop range is off by one: `{buggy_call}` should be `{src.text(node)}`.")


def _name_typos(src, tree):
    defined = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    defined |= {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)}
    taken = defined | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | set(dir(builtins))
    seen = set()

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                and node.id in defined and len(node.id) >= 4 and node.id not in seen):
            continue
        seen.add(node.id)
        middle = len(node.id) // 2
        typo = node.id[:middle] + node.id[middle + 1:]
        if typo in taken:
            continue
        buggy = src.replace(*src.span(node), typo)
        yield Mutation("name_typo", buggy,
                       f"`{typo}` is a typo for `{node.id}`, so Python raises a NameError; use `{node.id}`.")


def _dropped_casts(src, tree):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("int", "str", "float") and len(node.args) == 1
                and not node.keywords and _one_line(node)):
            continue
        arg_text = src.text(node.args[0])
        buggy = src.replace(*src.span(node), arg_text)
        yield Mutation("missing_cast", buggy,
                       f"`{arg_text}` has to be converted first: use `{src.text(node)}` "
                       f"instead of `{arg_text}`.")


def _missing_returns(src, tree):
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or len(fn.body) < 2:
            continue
        last = fn.body[-1]
        if not (isinstance(last, ast.Return) and last.value is not None and _one_line(last)):
            continue
        line_start, line_end = src.starts[last.lineno - 1], src.starts[last.lineno]
        if src.code[line_start:line_end].strip() != src.text(last):
            continue
        buggy = src.code[:line_start] + src.code[line_end:]
        yield Mutation("missing_return", buggy,
                       f"The function never returns its result, so it returns `None`; "
                       f"add `{src.text(last)}` at the end.")


def _flipped_bool_returns(src, tree):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Return) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, bool)):
            continue
        old = str(node.value.value)
        new = "False" if node.value.value else "True"
        buggy = src.replace(*src.span(node.value), new)
        yield Mutation("wrong_bool", buggy,
                       f"This branch returns `{new}` but it should return `{old}`.")


def _boolop_swaps(src, tree):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BoolOp) and _one_line(node)):
            continue
        old, new = ("and", "or") if isinstance(node.op, ast.And) else ("or", "and")
        swapped = _swap_in_gap(src, node, node.values[0], node.values[1], old, new)
        if swapped:
            buggy, _ = swapped
            yield Mutation("and_or", buggy,
                           f"The checks should be combined with `{old}`, not `{new}`: "
                           f"`{src.text(node)}`.")


def _none_inits(src, tree):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and _one_line(node)):
            continue
        value = node.value
        empty_container = isinstance(value, (ast.List, ast.Dict, ast.Set)) and not getattr(
            value, "elts", getattr(value, "keys", None))
        zero_or_empty = (isinstance(value, ast.Constant) and not isinstance(value.value, bool)
                         and value.value in (0, "", 0.0))
        if not (empty_container or zero_or_empty):
            continue
        name, value_text = node.targets[0].id, src.text(value)
        buggy = src.replace(*src.span(value), "None")
        yield Mutation("none_init", buggy,
                       f"`{name}` starts as `None` instead of `{value_text}`; "
                       f"initialise it with `{name} = {value_text}`.")


_OPERATORS = [
    _compare_flips, _binop_swaps, _range_off_by_one, _name_typos, _dropped_casts,
    _missing_returns, _flipped_bool_returns, _boolop_swaps, _none_inits,
]


def all_mutations(code: str) -> list:
    """Every single-bug variant of `code` that still parses."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    src, out = SourceText(code), []
    for operator in _OPERATORS:
        for mutation in operator(src, tree):
            if mutation.buggy == code:
                continue
            try:
                ast.parse(mutation.buggy)
            except SyntaxError:
                continue
            out.append(mutation)
    return out
