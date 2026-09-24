"""
opencodeinstruct.py — PY-V Data Pipeline v2 (data/scripts/sources/)
NVIDIA OpenCodeInstruct → "write code" (generate) records.
Keeps only rows whose unit tests all passed, and only the code itself
(no ``` fences, no example-usage tail). License: CC-BY-4.0 — credit NVIDIA.
"""

from datasets import load_dataset

from data.scripts.cleaner import is_parseable, is_mostly_ascii
from data.scripts.sources.common import make_record, first_python_block, trim_demo_code

SOURCE  = "opencodeinstruct"
LICENSE = "cc-by-4.0"


def iter_records(cfg: dict, stats):
    rows = load_dataset(cfg["hf_id"], cfg["config"], split=cfg["split"], streaming=True)

    for row in rows:
        stats["scanned"] += 1

        try:
            score = float(row["average_test_score"])
        except (TypeError, ValueError):
            score = 0.0
        if score < cfg["min_test_score"]:
            stats["rejected: not all tests passed"] += 1
            continue

        code = first_python_block(row["output"])
        if code is None:
            stats["rejected: no python code block"] += 1
            continue

        code = trim_demo_code(code)
        if not is_parseable(code):
            stats["rejected: code does not parse"] += 1
            continue

        if not (is_mostly_ascii(row["input"]) and is_mostly_ascii(code)):
            stats["rejected: non-English text"] += 1
            continue

        yield make_record(row["input"], code, SOURCE, "generate", LICENSE,
                          source_id=row["id"], domain=row["domain"])
