"""
glaive.py — PY-V Data Pipeline v2 (data/scripts/sources/)
Glaive code assistant → "explain" records (plain text answers).
Keeps Python-only "what / why / how does / explain / difference" questions
whose answers are mostly text; code blocks and their "Here's how:" lead-ins
are removed, because explain mode answers in words only. Answers that still
talk about the removed code ("In this code...") are dropped.
License: Apache-2.0.
"""

import re

from datasets import load_dataset

from data.scripts.cleaner import is_mostly_ascii
from data.scripts.sources.common import make_record, code_blocks, strip_code_blocks

SOURCE  = "glaive"
LICENSE = "apache-2.0"

_PYTHON = re.compile(r"\b(python|pandas|numpy|django|flask|pip|pytest|asyncio)\b", re.I)

_EXPLAIN = re.compile(
    r"\b(what (is|are|does|do)|why|how does|how do .{0,40} work|explain|"
    r"difference between|meaning of|purpose of)\b",
    re.I,
)

# Text that only makes sense next to the (removed) code
_CODE_REFERENCE = re.compile(
    r"\b((in|from|with|using) (this|the above|the following|that) (code|example|snippet|script)|"
    r"(this|the above|the following|above) (code|example|snippet)|"
    r"(revised|updated|modified|corrected|new) (version|code)|"
    r"as you can see|as shown (above|below)|in the code)\b",
    re.I,
)

_PYTHON_FENCES = {"python", "py", "python3", ""}


def iter_records(cfg: dict, stats):
    rows = load_dataset(cfg["hf_id"], cfg["config"], split=cfg["split"], streaming=True)

    for row in rows:
        stats["scanned"] += 1
        question, answer = row["question"], row["answer"]

        if not _PYTHON.search(question):
            stats["rejected: not a Python question"] += 1
            continue

        languages = {lang for lang, _ in code_blocks(question + "\n" + answer)}
        if languages - _PYTHON_FENCES:
            stats["rejected: other-language code"] += 1
            continue

        if not _EXPLAIN.search(question):
            stats["rejected: not an explain question"] += 1
            continue

        text = strip_code_blocks(answer)
        if len(text) < cfg["min_answer_chars"] or len(text) < cfg["min_text_share"] * len(answer):
            stats["rejected: answer mostly code"] += 1
            continue

        if _CODE_REFERENCE.search(text):
            stats["rejected: text refers to removed code"] += 1
            continue

        if not (is_mostly_ascii(question) and is_mostly_ascii(text)):
            stats["rejected: non-English text"] += 1
            continue

        yield make_record(question, text, SOURCE, "explain", LICENSE)
