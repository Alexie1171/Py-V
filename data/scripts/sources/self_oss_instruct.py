"""
self_oss_instruct.py — PY-V Data Pipeline v2 (data/scripts/sources/)
BigCode self-oss-instruct-sc2-exec-filter-50k → "write code" (generate)
records. Rows were already execution-checked by BigCode. Keeps only the
code from the response (drops the reasoning and explanation around it).
License: ODC-By.
"""

from datasets import load_dataset

from data.scripts.cleaner import is_parseable, is_mostly_ascii
from data.scripts.sources.common import make_record, first_python_block, trim_demo_code

SOURCE  = "self_oss_instruct"
LICENSE = "odc-by"


def iter_records(cfg: dict, stats):
    rows = load_dataset(cfg["hf_id"], cfg["config"], split=cfg["split"], streaming=True)

    for row in rows:
        stats["scanned"] += 1

        code = first_python_block(row["response"])
        if code is None:
            stats["rejected: no python code block"] += 1
            continue

        code = trim_demo_code(code)
        if not is_parseable(code):
            stats["rejected: code does not parse"] += 1
            continue

        if not (is_mostly_ascii(row["instruction"]) and is_mostly_ascii(code)):
            stats["rejected: non-English text"] += 1
            continue

        yield make_record(row["instruction"], code, SOURCE, "generate", LICENSE,
                          source_id=str(row["id"]))
