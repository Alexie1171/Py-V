"""
formatter.py — PY-V Data Pipeline
Upgraded v3: Minimum quality score filter added on top of v2.
             Drops samples below MIN_CODE_SCORE after cleaning.
"""

import json
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_VAL_SPLIT  = 0.1
DEFAULT_SEED       = 42

# Minimum quality score to include a sample in the final dataset.
# Samples below this are too short, lack comments, or are stub-like.
# Raise this to get a smaller but higher-quality dataset.
MIN_CODE_SCORE     = 1.0

# ─── Formatting ───────────────────────────────────────────────────────────────

def format_sample(sample: dict) -> dict | None:
    """
    Normalize a raw sample into the standard training format.
    Handles both 'input/output' and 'instruction/output' schemas.
    Returns None if the sample can't be formatted.
    """
    output = sample.get("output", "").strip()
    if not output:
        return None

    instruction = (
        sample.get("instruction")
        or sample.get("input")
        or ""
    ).strip()

    if not instruction:
        return None

    formatted = {
        "instruction": instruction,
        "output":      output,
    }

    if "metadata" in sample:
        formatted["metadata"] = sample["metadata"]

    return formatted


def format_dataset(samples: list[dict]) -> list[dict]:
    """
    Format, validate, and apply minimum quality score filter.
    Drops samples with no metadata score or score below MIN_CODE_SCORE.
    """
    formatted  = []
    dropped    = 0
    low_quality = 0

    for sample in samples:
        result = format_sample(sample)
        if not result:
            dropped += 1
            continue

        # Apply quality score gate
        score = result.get("metadata", {}).get("code_score", 0.0)
        if score < MIN_CODE_SCORE:
            low_quality += 1
            continue

        formatted.append(result)

    logger.info(
        f"Formatted: {len(formatted)} valid, "
        f"{dropped} malformed dropped, "
        f"{low_quality} below quality threshold (score < {MIN_CODE_SCORE})"
    )
    return formatted

# ─── Splitting ────────────────────────────────────────────────────────────────

def shuffle_and_split(
    samples:   list[dict],
    val_split: float = DEFAULT_VAL_SPLIT,
    seed:      int   = DEFAULT_SEED,
) -> tuple[list[dict], list[dict]]:
    """Shuffle with fixed seed and split into train/val."""
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    split_idx  = int(len(shuffled) * (1 - val_split))
    train_data = shuffled[:split_idx]
    val_data   = shuffled[split_idx:]

    logger.info(f"Split: {len(train_data)} train / {len(val_data)} val")
    return train_data, val_data

# ─── Quality Sorting ─────────────────────────────────────────────────────────

def sort_by_quality(samples: list[dict]) -> list[dict]:
    """Sort samples by code_score descending. Falls back to code length."""
    def quality_key(s: dict) -> float:
        meta = s.get("metadata", {})
        if "code_score" in meta:
            return meta["code_score"]
        return len(s.get("output", ""))

    return sorted(samples, key=quality_key, reverse=True)

# ─── I/O ─────────────────────────────────────────────────────────────────────

def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(samples: list[dict], path: str) -> None:
    """Save samples as JSONL (one JSON object per line)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(samples)} samples → {path}")

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_formatter(
    input_path:   str   = "data/processed/deduped/deduped.json",
    train_path:   str   = "data/datasets/train.jsonl",
    val_path:     str   = "data/datasets/val.jsonl",
    val_split:    float = DEFAULT_VAL_SPLIT,
    seed:         int   = DEFAULT_SEED,
    sort_quality: bool  = True,
) -> None:
    raw     = load_json(input_path)
    samples = format_dataset(raw)

    if sort_quality:
        samples = sort_by_quality(samples)

    train, val = shuffle_and_split(samples, val_split=val_split, seed=seed)

    save_jsonl(train, train_path)
    save_jsonl(val,   val_path)

    logger.info(f"Done. Train: {len(train)}, Val: {len(val)}")

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/deduped/deduped.json"
    train_path = sys.argv[2] if len(sys.argv) > 2 else "data/datasets/train.jsonl"
    val_path   = sys.argv[3] if len(sys.argv) > 3 else "data/datasets/val.jsonl"

    run_formatter(
        input_path=input_path,
        train_path=train_path,
        val_path=val_path,
    )