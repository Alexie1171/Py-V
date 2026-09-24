"""
unrefactor.py — PY-V Data Pipeline v2 (data/scripts/sources/)
The reverse of refactoring: rewrites clean code into clumsy code that should
behave the same (comprehension → loop with append, enumerate → range(len()),
`return a == b` → if/else True/False, ...). The clean original is then the
"improved" answer. Equivalence is NOT assumed — improve_synthetic.py re-runs
the unit tests on the clumsy version and drops it if anything changed.
Pure text/AST work: nothing here runs code.
"""

import ast
from dataclasses import dataclass

from data.scripts.sources.mutations import SourceText


@dataclass
class Rewrite:
    kind: str
    code: str


def _whole_line(src, node):
    """(start, end, indent, newline) of the line holding a one-line statement, or None."""
    if node.lineno != node.end_lineno:
        return None
    start, end = src.starts[node.lineno - 1], src.starts[node.lineno]
    line = src.code[start:end]
    if line.strip() != src.text(node):
        return None   # something else shares the line (`;`, trailing comment)
    indent = line[:len(line) - len(line.lstrip())]
    return start, end, indent, "\n" if line.endswith("\n") else ""


def _replace_line(src, node, new_lines):
    spot = _whole_line(src, node)
    if spot is None:
        return None
    start, end, indent, newline = spot
    body = "\n".join(indent + line for line in new_lines)
    return src.code[:start] + body + newline + src.code[end:]


def _free_name(tree, wanted):
    taken = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    taken |= {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)}
    return None if wanted in taken else wanted


def _simple_comp(node):
    """A comprehension with one plain `for` and at most one `if`."""
    return (len(node.generators) == 1 and not node.generators[0].is_async
            and len(node.generators[0].ifs) <= 1)


def _loop_lines(src, comp, action):
    gen   = comp.generators[0]
    lines = [f"for {src.text(gen.target)} in {src.text(gen.iter)}:"]
    if gen.ifs:
        lines += [f"    if {src.text(gen.ifs[0])}:", f"        {action}"]
    else:
        lines += [f"    {action}"]
    return lines


# ─── Rewrites ─────────────────────────────────────────────────────────────────

def _comprehension_to_loop(src, tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, ast.ListComp) and _simple_comp(node.value):
            name, comp = node.targets[0].id, node.value
            new = [f"{name} = []"] + _loop_lines(src, comp, f"{name}.append({src.text(comp.elt)})")
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.ListComp) \
                and _simple_comp(node.value) and _free_name(tree, "result"):
            comp = node.value
            new = ["result = []"] + _loop_lines(src, comp, f"result.append({src.text(comp.elt)})") + ["return result"]
        else:
            continue
        code = _replace_line(src, node, new)
        if code:
            yield Rewrite("comprehension_to_loop", code)


def _sum_to_loop(src, tree):
    for node in ast.walk(tree):
        value = getattr(node, "value", None)
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "sum"
                and len(value.args) == 1 and not value.keywords
                and isinstance(value.args[0], (ast.GeneratorExp, ast.ListComp)) and _simple_comp(value.args[0])):
            continue
        comp = value.args[0]
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            new = [f"{name} = 0"] + _loop_lines(src, comp, f"{name} += {src.text(comp.elt)}")
        elif isinstance(node, ast.Return) and _free_name(tree, "total"):
            new = ["total = 0"] + _loop_lines(src, comp, f"total += {src.text(comp.elt)}") + ["return total"]
        else:
            continue
        code = _replace_line(src, node, new)
        if code:
            yield Rewrite("sum_to_loop", code)


def _bool_return_to_if(src, tree):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Return) and (
                isinstance(node.value, ast.Compare)
                or (isinstance(node.value, ast.UnaryOp) and isinstance(node.value.op, ast.Not)))):
            continue
        new = [f"if {src.text(node.value)}:", "    return True", "else:", "    return False"]
        code = _replace_line(src, node, new)
        if code:
            yield Rewrite("bool_return_to_if", code)


def _enumerate_to_range(src, tree):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "enumerate"
                and len(node.iter.args) == 1 and not node.iter.keywords
                and isinstance(node.iter.args[0], (ast.Name, ast.Attribute))
                and isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2
                and all(isinstance(e, ast.Name) for e in node.target.elts)
                and node.body[0].lineno > node.lineno):
            continue
        header_start, header_end = src.starts[node.lineno - 1], src.starts[node.lineno]
        header = src.code[header_start:header_end]
        if not header.strip().startswith("for ") or not header.rstrip().endswith(":"):
            continue
        index, value = (e.id for e in node.target.elts)
        seq = src.text(node.iter.args[0])
        indent = header[:len(header) - len(header.lstrip())]
        first = src.code[src.starts[node.body[0].lineno - 1]:src.starts[node.body[0].lineno]]
        body_indent = first[:len(first) - len(first.lstrip())]
        new_header = f"{indent}for {index} in range(len({seq})):\n{body_indent}{value} = {seq}[{index}]\n"
        yield Rewrite("enumerate_to_range", src.code[:header_start] + new_header + src.code[header_end:])


def _truthiness_to_len(src, tree):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.While)):
            continue
        test = node.test
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and isinstance(test.operand, ast.Name):
            new = f"len({test.operand.id}) == 0"
        elif isinstance(test, ast.Name):
            new = f"len({test.id}) > 0"
        else:
            continue
        yield Rewrite("truthiness_to_len", src.replace(*src.span(test), new))


def _max_min_to_if(src, tree):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id in ("max", "min") and len(node.value.args) == 2
                and not node.value.keywords
                and not any(isinstance(a, ast.Starred) for a in node.value.args)):
            continue
        name = node.targets[0].id
        a, b = (src.text(arg) for arg in node.value.args)
        op = ">=" if node.value.func.id == "max" else "<="
        new = [f"if {a} {op} {b}:", f"    {name} = {a}", "else:", f"    {name} = {b}"]
        code = _replace_line(src, node, new)
        if code:
            yield Rewrite("max_min_to_if", code)


def _ternary_to_if(src, tree):
    for node in ast.walk(tree):
        value = getattr(node, "value", None)
        if not isinstance(value, ast.IfExp):
            continue
        cond, yes, no = src.text(value.test), src.text(value.body), src.text(value.orelse)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            new = [f"if {cond}:", f"    {name} = {yes}", "else:", f"    {name} = {no}"]
        elif isinstance(node, ast.Return):
            new = [f"if {cond}:", f"    return {yes}", "else:", f"    return {no}"]
        else:
            continue
        code = _replace_line(src, node, new)
        if code:
            yield Rewrite("ternary_to_if", code)


def _augassign_to_assign(src, tree):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
                and isinstance(node.op, (ast.Add, ast.Sub))):
            continue
        op = "+" if isinstance(node.op, ast.Add) else "-"
        name = node.target.id
        code = _replace_line(src, node, [f"{name} = {name} {op} {src.text(node.value)}"])
        if code:
            yield Rewrite("augassign_to_assign", code)


_REWRITES = [
    _comprehension_to_loop, _sum_to_loop, _bool_return_to_if, _enumerate_to_range,
    _truthiness_to_len, _max_min_to_if, _ternary_to_if, _augassign_to_assign,
]


def rewrites(code: str) -> list:
    """Every single clumsy rewrite of `code` that still parses."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    src, out = SourceText(code), []
    for rewrite in _REWRITES:
        for candidate in rewrite(src, tree):
            try:
                ast.parse(candidate.code)
            except SyntaxError:
                continue
            if candidate.code != code:
                out.append(candidate)
    return out
