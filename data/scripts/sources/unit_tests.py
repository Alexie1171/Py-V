"""
unit_tests.py — PY-V Data Pipeline v2 (data/scripts/sources/)
Shared by the sources that run code (bug_fix, improve_synthetic):
  - tested_functions(): streams OpenCodeInstruct rows whose unit tests all
    passed upstream and still pass here, skipping rows already used elsewhere
  - run_tests(): runs a function against its tests, reports the first failure
  - require_colab(): refuses to run internet code outside Colab
"""

import json
import os

from datasets import load_dataset

from data.scripts.cleaner import is_parseable, is_mostly_ascii
from data.scripts.sources.common import first_python_block, trim_demo_code
from experiments.code_runner import run_python_capture

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


def require_colab(source: str):
    if not (os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("PYV_ALLOW_LOCAL_EXEC") == "1"):
        raise RuntimeError(f"{source} runs code from the internet - run it on Colab (runner notebook)")


def run_tests(code: str, tests: list, timeout: int):
    """First test failure as a dict ({"kind": "pass"|"assert"|"error", ...}), or None if it crashed."""
    program = code + "\n\n" + _HARNESS.format(tests=tests, marker=_MARKER)
    _, stdout, _ = run_python_capture(program, timeout)
    for line in stdout.splitlines():
        if line.startswith(_MARKER):
            return json.loads(line[len(_MARKER):])
    return None   # syntax error at load, timeout, or blocked call


def used_ids(paths) -> set:
    """source_ids already taken by other record files (missing files are ignored)."""
    if isinstance(paths, str):
        paths = [paths]
    ids = set()
    for path in paths or []:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                ids |= {json.loads(line)["metadata"].get("source_id") for line in f if line.strip()}
    return ids


def tested_functions(cfg: dict, stats):
    """
    Yield (row_id, instruction, code, tests) for OpenCodeInstruct functions
    that pass all their unit tests (upstream and here), are short enough, and
    are not already used by the files in cfg["exclude_ids_from"].
    """
    exclude = used_ids(cfg.get("exclude_ids_from"))
    rows = load_dataset(cfg["hf_id"], cfg["config"], split=cfg["split"], streaming=True)

    for row in rows:
        stats["scanned"] += 1

        if row["id"] in exclude:
            stats["rejected: already used by another source"] += 1
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

        if (run_tests(code, tests, cfg["timeout_seconds"]) or {}).get("kind") != "pass":
            stats["rejected: original fails its tests here"] += 1
            continue

        yield row["id"], row["input"], code, tests
