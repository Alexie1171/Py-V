"""
old_github.py — PY-V Data Pipeline v2 (data/scripts/sources/)
The v1 dataset (scraped GitHub functions) → best part only, as "write code"
(generate) records. Drops auto-generated instructions ("Write a Python
function named `x` that takes y"), low quality scores, long project-bound
functions and non-English rows. Highest code_score first.
License: mixed — original repositories.
"""

import json
import re

from data.scripts.cleaner import is_parseable, is_mostly_ascii
from data.scripts.sources.common import make_record

SOURCE  = "github_v1"
LICENSE = "per-repository"

_AUTO_INSTRUCTION = re.compile(r"^\s*write a python function named\b", re.I)


def iter_records(cfg: dict, stats):
    with open(cfg["path"], "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    rows.sort(key=lambda r: r.get("metadata", {}).get("code_score", 0), reverse=True)

    for row in rows:
        stats["scanned"] += 1
        instruction, code = row["instruction"], row["output"]
        score = row.get("metadata", {}).get("code_score", 0)

        if score < cfg["min_code_score"]:
            stats["rejected: code_score too low"] += 1
            continue

        n_lines = len([line for line in code.split("\n") if line.strip()])
        if n_lines > cfg["max_code_lines"]:
            stats["rejected: code too long"] += 1
            continue

        if _AUTO_INSTRUCTION.match(instruction):
            stats["rejected: auto-generated instruction"] += 1
            continue

        if not (is_mostly_ascii(instruction) and is_mostly_ascii(code)):
            stats["rejected: non-English text"] += 1
            continue

        if not is_parseable(code):
            stats["rejected: code does not parse"] += 1
            continue

        yield make_record(instruction, code, SOURCE, "generate", LICENSE,
                          code_score=score)
