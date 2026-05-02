"""
dedupe.py — PY-V Data Pipeline
Upgraded v2: Hash-based dedup on code content, near-duplicate detection
             via token fingerprinting, safe dict handling, proper structure.
"""

import re
import json
import hashlib
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

# Minimum Jaccard similarity to consider two samples near-duplicates
NEAR_DUPE_THRESHOLD = 0.85

# N-gram size for shingling (near-dupe detection)
SHINGLE_SIZE = 5

# ─── Exact Deduplication ──────────────────────────────────────────────────────

def hash_code(code: str) -> str:
    """
    Produce a stable hash of a code string.
    Normalizes whitespace before hashing so formatting differences
    don't create false uniques.
    """
    normalized = re.sub(r"\s+", " ", code.strip())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def exact_dedupe(samples: list[dict]) -> list[dict]:
    """
    Remove samples with identical code (output field).
    Preserves first occurrence, drops all subsequent duplicates.
    """
    seen    : set       = set()
    result  : list[dict] = []
    dropped : int        = 0

    for sample in samples:
        code = sample.get("output", "")
        h    = hash_code(code)

        if h in seen:
            dropped += 1
        else:
            seen.add(h)
            result.append(sample)

    logger.info(f"Exact dedup: {dropped} duplicates removed, {len(result)} remain")
    return result

# ─── Near-Duplicate Detection ─────────────────────────────────────────────────

def get_shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    """
    Build a set of k-word shingles from text.
    Used for Jaccard similarity comparison.
    """
    tokens  = re.findall(r"\w+", text.lower())
    return {" ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def near_dedupe(samples: list[dict], threshold: float = NEAR_DUPE_THRESHOLD) -> list[dict]:
    """
    Remove near-duplicate samples using Jaccard shingling.
    More expensive than exact dedup — run AFTER exact_dedupe.

    Warning: O(n²) — only run on datasets < ~5000 samples at a time,
    or increase shingle size / reduce threshold for speed.
    """
    shingles_list = [get_shingles(s.get("output", "")) for s in samples]
    keep          = [True] * len(samples)

    for i in range(len(samples)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(samples)):
            if not keep[j]:
                continue
            sim = jaccard(shingles_list[i], shingles_list[j])
            if sim >= threshold:
                keep[j] = False   # drop the later duplicate

    result  = [s for s, k in zip(samples, keep) if k]
    dropped = keep.count(False)
    logger.info(f"Near-dedup: {dropped} near-duplicates removed, {len(result)} remain")
    return result

# ─── Source Balancing ─────────────────────────────────────────────────────────

def log_source_distribution(samples: list[dict]) -> None:
    """Log how many samples came from each source."""
    counts: dict = defaultdict(int)
    for s in samples:
        source = s.get("metadata", {}).get("source", "unknown")
        counts[source] += 1
    for source, count in sorted(counts.items()):
        logger.info(f"  {source}: {count} samples")

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def dedupe(
    samples:          list[dict],
    run_near_dedupe:  bool  = True,
    near_threshold:   float = NEAR_DUPE_THRESHOLD,
) -> list[dict]:
    """
    Full deduplication pipeline:
    1. Exact hash-based dedup on normalized code
    2. Optional near-duplicate removal via Jaccard shingling
    """
    logger.info(f"Starting dedup with {len(samples)} samples")

    samples = exact_dedupe(samples)

    if run_near_dedupe:
        samples = near_dedupe(samples, threshold=near_threshold)

    log_source_distribution(samples)
    logger.info(f"Dedup complete. Final count: {len(samples)}")
    return samples

# ─── I/O Helpers ─────────────────────────────────────────────────────────────

def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(data)} samples to {path}")

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    input_path  = sys.argv[1] if len(sys.argv) > 1 else "data/processed/cleaned/cleaned.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/deduped/deduped.json"

    samples = load_json(input_path)
    deduped = dedupe(samples, run_near_dedupe=True)
    save_json(deduped, output_path)