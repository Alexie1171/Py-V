"""
decontaminate.py — PY-V Data Pipeline v2
Keeps the scoring test honest: finds training records that overlap the
benchmarks — MBPP (every config and split) and HumanEval — by
  - any 10-word run shared with a benchmark problem statement, or
  - identical code (whitespace-normalised) to a benchmark solution.
"""

import re

from datasets import load_dataset

from data.scripts.dedupe import hash_code

NGRAM = 10


def _words(text: str) -> list:
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def _ngrams(words: list) -> set:
    return {" ".join(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


class BenchmarkIndex:
    """Problem-statement n-grams and solution hashes of the scoring benchmarks."""

    def __init__(self):
        self.ngrams      = set()
        self.code_hashes = set()

        for config in ("full", "sanitized"):
            for split in load_dataset("google-research-datasets/mbpp", config).values():
                for row in split:
                    self._add(row.get("text") or row.get("prompt"), row["code"])

        for row in load_dataset("openai/openai_humaneval", split="test"):
            self._add(row["prompt"], row["prompt"] + row["canonical_solution"])

    def _add(self, text: str, code: str):
        self.ngrams |= _ngrams(_words(text))
        self.code_hashes.add(hash_code(code))

    def overlaps(self, record: dict) -> bool:
        if hash_code(record["output"]) in self.code_hashes:
            return True
        return not self.ngrams.isdisjoint(_ngrams(_words(record["instruction"])))
