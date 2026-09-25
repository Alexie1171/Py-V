"""
bug_fix.py — PY-V Data Pipeline v2 (data/scripts/sources/)
"Fix the error" (debug) records made by breaking working code on purpose.

Source: OpenCodeInstruct functions that pass their unit tests (unit_tests.py;
rows already used by other sources are skipped). For each function:
  1. apply one realistic bug (mutations.py) — the tests must now fail;
  2. record what the user would see: the error message or the failing check.
Instruction = what the user saw + the broken code.
Output      = the original code + a one-sentence explanation of the fix.
Bug kinds are balanced by always trying the least-used kind first.

Runs code from the internet — Colab only. License: CC-BY-4.0 — credit NVIDIA.
"""

import random
from collections import Counter

from data.scripts.sources.common import make_record, pick
from data.scripts.sources.mutations import all_mutations
from data.scripts.sources.unit_tests import require_cloud, run_tests, tested_functions

SOURCE  = "bug_fix"
LICENSE = "cc-by-4.0"

PROGRESS_EVERY = 100

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


def _instruction(result: dict, buggy: str, source_id: str) -> str:
    if result["kind"] == "error":
        return pick(_ERROR_TEMPLATES, source_id).format(error=result["error"], code=buggy)
    got = f" (got {result['got']})" if result.get("got") else ""
    return pick(_ASSERT_TEMPLATES, source_id).format(test=result["test"], got=got, code=buggy)


def iter_records(cfg: dict, stats):
    require_cloud(SOURCE)
    kind_count = Counter()
    kept       = 0

    for row_id, _, code, tests in tested_functions(cfg, stats):
        mutations = all_mutations(code)
        random.Random(row_id).shuffle(mutations)
        mutations.sort(key=lambda m: kind_count[m.kind])   # least-used kind first

        chosen = None
        for mutation in mutations[:cfg["max_tries"]]:
            result = run_tests(mutation.buggy, tests, cfg["timeout_seconds"])
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
            _instruction(result, mutation.buggy, row_id),
            f"{code}\n\n{mutation.explanation}",
            SOURCE, "debug", LICENSE,
            source_id=row_id, bug_kind=mutation.kind,
        )
