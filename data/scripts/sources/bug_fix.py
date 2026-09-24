"""
bug_fix.py — PY-V Data Pipeline v2 (data/scripts/sources/)
"Fix the error" (debug) records made by breaking working code on purpose.

Source: NVIDIA OpenCodeInstruct rows whose unit tests all pass (rows already
used for write-code records are skipped). For each function:
  1. run it against its unit tests — it must pass here too;
  2. apply one realistic bug (mutations.py) — the tests must now fail;
  3. record what the user would see: the error message or the failing check.
Instruction = what the user saw + the broken code.
Output      = the original code + a one-sentence explanation of the fix.
Bug kinds are balanced by always trying the least-used kind first.

Runs code from the internet — Colab only (refuses to run elsewhere unless
PYV_ALLOW_LOCAL_EXEC=1). License: CC-BY-4.0 — credit NVIDIA.
"""

import hashlib
import json
import os
import random
from collections import Counter

from datasets import load_dataset

from data.scripts.cleaner import is_parseable, is_mostly_ascii
from data.scripts.sources.common import make_record, first_python_block, trim_demo_code
from data.scripts.sources.mutations import all_mutations
from experiments.code_runner import run_python_capture

SOURCE  = "bug_fix"
LICENSE = "cc-by-4.0"

PROGRESS_EVERY = 100

_MARKER = "__PYV_RESULT__"

# Appended to the code: runs the tests one by one and prints the first failure
# as JSON — for a failing `assert f(x) == y` it also reports what f(x) returned.
_HARNESS = '''
import ast as __ast, json as __json
__tests = {tests!r}
for __t in __tests:
    try:
        exec(__t, globals())
    except AssertionError:
        __got = None
        try:
            __node = __ast.parse(__t.strip()).body[0]
            if isinstance(__node.test, __ast.Compare) and len(__node.test.ops) == 1:
                __left = compile(__ast.Expression(__node.test.left), "<test>", "eval")
                __got = repr(eval(__left, globals()))[:80]
        except Exception:
            pass
        print("{marker}" + __json.dumps({{"kind": "assert", "test": __t.strip(), "got": __got}}))
        raise SystemExit(1)
    except Exception as __e:
        __msg = type(__e).__name__ + ": " + str(__e)[:200]
        print("{marker}" + __json.dumps({{"kind": "error", "test": __t.strip(), "error": __msg}}))
        raise SystemExit(1)
print("{marker}" + __json.dumps({{"kind": "pass"}}))
'''

_ERROR_TEMPLATES = [
    "I get this error when I run my code:\n\n{error}\n\n{code}",
    "My code crashes with `{error}`. How do I fix it?\n\n{code}",
    "Why does this raise {error}?\n\n{code}",
]

_ASSERT_TEMPLATES = [
    "My function gives the wrong result. This check fails:\n\n{test}{got}\n\n{code}",
    "This test fails and I can't see why:\n\n{test}{got}\n\n{code}",
    "The output is wrong: `{test}` fails{got}.\n\n{code}",
]


def _run_tests(code: str, tests: list, timeout: int):
    """First test failure as a dict ({"kind": "pass"|"assert"|"error", ...}), or None if it crashed."""
    program = code + "\n\n" + _HARNESS.format(tests=tests, marker=_MARKER)
    _, stdout, _ = run_python_capture(program, timeout)
    for line in stdout.splitlines():
        if line.startswith(_MARKER):
            return json.loads(line[len(_MARKER):])
    return None   # syntax error at load, timeout, or blocked call


def _instruction(result: dict, buggy: str, source_id: str) -> str:
    pick = int(hashlib.md5(source_id.encode()).hexdigest(), 16)
    if result["kind"] == "error":
        template = _ERROR_TEMPLATES[pick % len(_ERROR_TEMPLATES)]
        return template.format(error=result["error"], code=buggy)
    got = f" (got {result['got']})" if result.get("got") else ""
    template = _ASSERT_TEMPLATES[pick % len(_ASSERT_TEMPLATES)]
    return template.format(test=result["test"], got=got, code=buggy)


def _used_ids(path) -> set:
    if not path or not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["metadata"].get("source_id") for line in f if line.strip()}


def iter_records(cfg: dict, stats):
    if not (os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("PYV_ALLOW_LOCAL_EXEC") == "1"):
        raise RuntimeError("bug_fix runs code from the internet - run it on Colab (runner notebook)")

    used       = _used_ids(cfg.get("exclude_ids_from"))
    kind_count = Counter()
    kept       = 0
    rows = load_dataset(cfg["hf_id"], cfg["config"], split=cfg["split"], streaming=True)

    for row in rows:
        stats["scanned"] += 1

        if row["id"] in used:
            stats["rejected: already used for write-code"] += 1
            continue

        try:
            score = float(row["average_test_score"])
            tests = [t.strip() for t in json.loads(row["unit_tests"]) if t.strip()]
        except (TypeError, ValueError):
            stats["rejected: unreadable tests"] += 1
            continue
        if score < cfg["min_test_score"] or not tests:
            stats["rejected: not all tests passed"] += 1
            continue
        tests = tests[:cfg["max_tests"]]

        code = first_python_block(row["output"])
        code = trim_demo_code(code) if code else None
        if not code or not is_parseable(code) or not is_mostly_ascii(row["input"] + code):
            stats["rejected: no clean code"] += 1
            continue
        if len([line for line in code.split("\n") if line.strip()]) > cfg["max_code_lines"]:
            stats["rejected: code too long"] += 1
            continue

        if (_run_tests(code, tests, cfg["timeout_seconds"]) or {}).get("kind") != "pass":
            stats["rejected: original fails its tests here"] += 1
            continue

        mutations = all_mutations(code)
        random.Random(row["id"]).shuffle(mutations)
        mutations.sort(key=lambda m: kind_count[m.kind])   # least-used kind first

        chosen = None
        for mutation in mutations[:cfg["max_tries"]]:
            result = _run_tests(mutation.buggy, tests, cfg["timeout_seconds"])
            if result and result["kind"] in ("assert", "error"):
                chosen = (mutation, result)
                break
        if chosen is None:
            stats["rejected: no bug broke the tests"] += 1
            continue

        mutation, result = chosen
        kind_count[mutation.kind] += 1
        stats[f"bug kind: {mutation.kind}"] += 1
        stats[f"user sees: {result['kind']}"] += 1

        kept += 1
        if kept % PROGRESS_EVERY == 0:
            print(f"  ... {kept} bug-fix records ({stats['scanned']} rows scanned)", flush=True)

        yield make_record(
            _instruction(result, mutation.buggy, row["id"]),
            f"{code}\n\n{mutation.explanation}",
            SOURCE, "debug", LICENSE,
            source_id=row["id"], bug_kind=mutation.kind,
        )
