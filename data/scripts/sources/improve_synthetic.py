"""
improve_synthetic.py — PY-V Data Pipeline v2 (data/scripts/sources/)
"Improve this code" (refactor) records made by un-refactoring clean code.

Source: OpenCodeInstruct functions that pass their unit tests (unit_tests.py;
rows used by other sources are skipped). Clumsy rewrites (unrefactor.py) are
applied one at a time; after each one the tests are re-run, and a rewrite
that changes behaviour is dropped. Kept only if at least `min_rewrites`
survive. Instruction = request + clumsy code; output = the clean original
(refactor mode returns code only).

Runs code from the internet — Colab only. License: CC-BY-4.0 — credit NVIDIA.
"""

import random

from data.scripts.sources.common import IMPROVE_TEMPLATES, make_record, pick
from data.scripts.sources.unit_tests import require_colab, run_tests, tested_functions
from data.scripts.sources.unrefactor import rewrites

SOURCE  = "improve_synthetic"
LICENSE = "cc-by-4.0"

PROGRESS_EVERY = 100


def clumsify(code: str, tests: list, cfg: dict, seed: str) -> tuple:
    """Apply up to max_rewrites behaviour-preserving rewrites of different kinds.
    Also used by experiments/eval_fix.py to build its improve-code questions."""
    rng, kinds, tries = random.Random(seed), [], 0

    while len(kinds) < cfg["max_rewrites"] and tries < cfg["max_tries"]:
        options = [r for r in rewrites(code) if r.kind not in kinds]
        if not options:
            break
        candidate = rng.choice(options)
        tries += 1
        if (run_tests(candidate.code, tests, cfg["timeout_seconds"]) or {}).get("kind") == "pass":
            code = candidate.code
            kinds.append(candidate.kind)

    return code, kinds


def iter_records(cfg: dict, stats):
    require_colab(SOURCE)
    kept = 0

    for row_id, _, code, tests in tested_functions(cfg, stats):
        clumsy, kinds = clumsify(code, tests, cfg, row_id)
        if len(kinds) < cfg["min_rewrites"]:
            stats["rejected: too few safe rewrites"] += 1
            continue

        for kind in kinds:
            stats[f"rewrite: {kind}"] += 1

        kept += 1
        if kept % PROGRESS_EVERY == 0:
            print(f"  ... {kept} improve records ({stats['scanned']} rows scanned)", flush=True)

        yield make_record(
            pick(IMPROVE_TEMPLATES, row_id).format(code=clumsy),
            code,
            SOURCE, "refactor", LICENSE,
            source_id=row_id, rewrites=kinds,
        )
