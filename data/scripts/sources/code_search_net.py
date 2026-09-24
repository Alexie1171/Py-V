"""
code_search_net.py — PY-V Data Pipeline v2 (data/scripts/sources/)
CodeSearchNet (Python) → "explain" records.
Instruction = the function without its docstring; output = the first
paragraph of the docstring (parameter lists, examples etc. cut off).
The dataset is ordered by repository, so rows per repository are capped.
License: mixed — each row keeps its original repository's license.
"""

import re
from collections import Counter

from datasets import load_dataset

from data.scripts.cleaner import is_mostly_ascii
from data.scripts.sources.common import make_record, strip_docstring

SOURCE  = "code_search_net"
LICENSE = "per-repository"

INSTRUCTION = "Explain what this Python function does.\n\n{code}"

# Where the summary part of a docstring ends
_SECTION_START = re.compile(
    r"^\s*(:param|:type|:return|:raises|@param|@return|Args:|Arguments:|"
    r"Parameters|Returns|Raises|Yields|Example|Examples|Note:|>>>)",
    re.M,
)


def docstring_summary(doc: str) -> str:
    """First paragraph of a docstring, before any parameter/example section."""
    section = _SECTION_START.search(doc)
    if section:
        doc = doc[:section.start()]
    first_paragraph = re.split(r"\n\s*\n", doc.strip())[0]
    return " ".join(first_paragraph.split())


def iter_records(cfg: dict, stats):
    rows = load_dataset(cfg["hf_id"], cfg["config"], split=cfg["split"], streaming=True)

    min_chars, max_chars = cfg["summary_chars"]
    min_lines, max_lines = cfg["code_lines"]
    per_repo = Counter()

    for row in rows:
        stats["scanned"] += 1
        repo = row["repository_name"]

        if per_repo[repo] >= cfg["max_per_repo"]:
            stats["rejected: repository cap reached"] += 1
            continue

        summary = docstring_summary(row["func_documentation_string"] or "")
        if not min_chars <= len(summary) <= max_chars or len(summary.split()) < cfg["min_summary_words"]:
            stats["rejected: docstring summary too short/long"] += 1
            continue

        code = strip_docstring(row["func_code_string"] or "")
        if code is None:
            stats["rejected: code unusable"] += 1
            continue

        lines = [line.strip() for line in code.split("\n") if line.strip()]
        if not min_lines <= len(lines) <= max_lines:
            stats["rejected: function too short/long"] += 1
            continue

        comments = sum(1 for line in lines if line.startswith("#"))
        if comments > cfg["max_comment_share"] * len(lines):
            stats["rejected: too many comment lines"] += 1
            continue

        if not (is_mostly_ascii(summary) and is_mostly_ascii(code)):
            stats["rejected: non-English text"] += 1
            continue

        if not summary.endswith("."):
            summary += "."

        per_repo[repo] += 1
        yield make_record(INSTRUCTION.format(code=code), summary, SOURCE, "explain",
                          LICENSE, repository=repo)
