"""
long_file_fix.py — PY-V Data Pipeline v2 (data/scripts/sources/)
"Fix the bug in this file" (debug) records: a tested function hidden among
other working functions, with one bug planted inside it. The answer is ONLY
the corrected function + a one-line explanation — the habit V needs on long
input (long-question test 2026-09-26: Phi-2 copied the whole file and ran out
of room before reaching the bug).

Source: OpenCodeInstruct functions that pass their unit tests
(unit_tests.tested_functions; rows used by other sources are skipped). The
surrounding code comes from earlier functions in the same stream. The bug
(dataset-v2 mutations) must be inside the tested function and must make its
tests fail inside the file; the clean file must pass them.

Runs code from the internet — Colab only. License: CC-BY-4.0 — credit NVIDIA.
"""

import ast
import random
from collections import Counter, deque

from data.scripts.sources.common import make_record, pick
from data.scripts.sources.mutations import all_mutations
from data.scripts.sources.unit_tests import require_cloud, run_tests, tested_functions

SOURCE  = "long_file_fix"
LICENSE = "cc-by-4.0"

PROGRESS_EVERY = 100
POOL_SIZE      = 300     # surrounding-code functions kept to pick from

_ERROR_TEMPLATES = [
    "I get this error when I run my code:\n\n{error}\n\nOnly `{name}` is broken. Here is the whole file:\n\n```python\n{module}\n```",
    "This test fails:\n\n{test}\n\nIt raises {error}\n\nFind the bug in the file below and write only the corrected `{name}` function.\n\n```python\n{module}\n```",
    "Why does `{name}` raise {error}? Show me just the fixed function.\n\n```python\n{module}\n```",
]

_ASSERT_TEMPLATES = [
    "This check fails:\n\n{test}{got}\n\nFind the bug in the file below and write only the corrected `{name}` function.\n\n```python\n{module}\n```",
    "`{name}` gives the wrong result — `{test}` fails{got}. Here is the whole file; only fix that function:\n\n```python\n{module}\n```",
    "Something in this file is wrong: the test `{test}` fails{got}. Which function has the bug? Give me the corrected version of it.\n\n```python\n{module}\n```",
]


def _called_name(test: str):
    try:
        tree = ast.parse(test.strip())
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id
    return None


def _top_names(code: str) -> set:
    tree = ast.parse(code)
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def _function(code: str, name: str):
    """(source text, dumped tree) of top-level function `name`, or (None, None)."""
    for node in ast.parse(code).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(code, node), ast.dump(node)
    return None, None


def _surroundings(pool, taken: set, target_chars: int, rng) -> list:
    """Functions from the pool (no name clashes) until the file reaches target_chars."""
    chosen, names, size = [], set(taken), 0
    for code, code_names in rng.sample(list(pool), len(pool)):
        if names & code_names:
            continue
        chosen.append(code)
        names |= code_names
        size  += len(code)
        if size >= target_chars:
            break
    return chosen


def _instruction(result: dict, name: str, module: str, source_id: str) -> str:
    if result["kind"] == "error":
        return pick(_ERROR_TEMPLATES, source_id).format(
            error=result["error"], test=result["test"], name=name, module=module)
    got = f" (got {result['got']})" if result.get("got") else ""
    return pick(_ASSERT_TEMPLATES, source_id).format(test=result["test"], got=got, name=name, module=module)


def iter_records(cfg: dict, stats):
    require_cloud(SOURCE)
    pool       = deque(maxlen=POOL_SIZE)
    kind_count = Counter()
    min_chars, max_chars = cfg["module_chars"]
    kept       = 0

    for row_id, _, code, tests in tested_functions(cfg, stats):
        names = _top_names(code)
        name  = _called_name(tests[0])
        fixed, fixed_dump = _function(code, name) if name in names else (None, None)
        warm  = len(pool) >= cfg["warmup_functions"]
        pool.append((code, names))           # every clean function also becomes surrounding code
        if not warm:
            stats["used as surrounding code only (warm-up)"] += 1
            continue
        if fixed is None:
            stats["rejected: tested thing is not a top-level function"] += 1
            continue

        rng    = random.Random(row_id)
        around = _surroundings(pool, names, rng.randint(min_chars, max_chars) - len(code), rng)
        index  = rng.randint(0, len(around))
        clean  = "\n\n\n".join(around[:index] + [code] + around[index:])
        if len(clean) > max_chars * 1.3:
            stats["rejected: file too long"] += 1
            continue
        if (run_tests(clean, tests, cfg["timeout_seconds"]) or {}).get("kind") != "pass":
            stats["rejected: tests fail inside the file (name clash)"] += 1
            continue

        # the bug must be inside the tested function
        mutations = [m for m in all_mutations(code) if _function(m.buggy, name)[1] not in (None, fixed_dump)]
        rng.shuffle(mutations)
        mutations.sort(key=lambda m: kind_count[m.kind])   # least-used bug kind first
        chosen = None
        for mutation in mutations[:cfg["max_tries"]]:
            module = "\n\n\n".join(around[:index] + [mutation.buggy] + around[index:])
            result = run_tests(module, tests, cfg["timeout_seconds"])
            if result and result["kind"] in ("assert", "error"):
                chosen = (mutation, module, result)
                break
        if chosen is None:
            stats["rejected: no bug broke the tests"] += 1
            continue

        mutation, module, result = chosen
        kind_count[mutation.kind] += 1
        stats[f"bug kind: {mutation.kind}"] += 1
        stats[f"file size: {len(around) + 1} functions" if len(around) < 9 else "file size: 10+ functions"] += 1

        kept += 1
        if kept % PROGRESS_EVERY == 0:
            print(f"  ... {kept} long-file records ({stats['scanned']} rows scanned)", flush=True)

        yield make_record(
            _instruction(result, name, module, row_id),
            f"{fixed}\n\n{mutation.explanation}",
            SOURCE, "debug", LICENSE,
            source_id=row_id, bug_kind=mutation.kind, file_functions=len(around) + 1,
        )
