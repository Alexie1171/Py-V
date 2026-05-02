"""
pipeline.py — PY-V Data Pipeline
Orchestrates all stages: scrape → clean → dedupe → format
Calls each module directly (no subprocess), passes data in memory,
handles errors per stage, saves checkpoints, logs stats throughout.
"""

import json
import logging
import sys
import time
from pathlib import Path

# ─── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/pipeline.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

PATHS = {
    "raw_github":        Path("data/raw/github/github_raw.json"),
    "raw_stackoverflow": Path("data/raw/stackoverflow/so_raw.json"),
    "raw_combined":      Path("data/raw/combined_raw.json"),
    "cleaned":           Path("data/processed/cleaned/cleaned.json"),
    "deduped":           Path("data/processed/deduped/deduped.json"),
    "train":             Path("data/datasets/train.jsonl"),
    "val":               Path("data/datasets/val.jsonl"),
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dirs():
    """Create all required directories if they don't exist."""
    for path in PATHS.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def save_json(data: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Checkpoint saved: {path} ({len(data)} samples)")


def load_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log_stage(name: str, before: int, after: int) -> None:
    dropped = before - after
    pct     = (dropped / before * 100) if before > 0 else 0
    logger.info(f"[{name}] {before} → {after} samples ({dropped} dropped, {pct:.1f}%)")

# ─── Stage Runners ────────────────────────────────────────────────────────────

def run_scraping(
    manual_repos:         list  = None,
    use_github_discovery: bool  = True,
    github_stars:         int   = 1000,
    github_pages:         int   = 2,
    so_tags:              list  = None,
    so_pages:             int   = 3,
    so_min_score:         int   = 10,
) -> list[dict]:
    """Run both scrapers, save raw checkpoints, return combined samples."""

    from data.scripts.github_scraper        import fetch_multiple_repos
    from data.scripts.stackoverflow_scraper import fetch_stackoverflow_samples

    all_samples = []

    # — GitHub —
    logger.info("=" * 50)
    logger.info("STAGE 1a: GitHub Scraper")
    logger.info("=" * 50)

    try:
        github_samples = fetch_multiple_repos(
            repos=manual_repos or [],
            use_discovery=use_github_discovery,
            discovery_stars=github_stars,
            discovery_pages=github_pages,
        )
        save_json(github_samples, PATHS["raw_github"])
        logger.info(f"GitHub: {len(github_samples)} samples collected")
        all_samples.extend(github_samples)
    except Exception as e:
        logger.error(f"GitHub scraper failed: {e}")
        logger.warning("Continuing without GitHub data...")

    # — StackOverflow —
    logger.info("=" * 50)
    logger.info("STAGE 1b: StackOverflow Scraper")
    logger.info("=" * 50)

    try:
        so_samples = fetch_stackoverflow_samples(
            tags=so_tags or ["python", "python-3.x"],
            pages=so_pages,
            min_q_score=so_min_score,
        )
        save_json(so_samples, PATHS["raw_stackoverflow"])
        logger.info(f"StackOverflow: {len(so_samples)} samples collected")
        all_samples.extend(so_samples)
    except Exception as e:
        logger.error(f"StackOverflow scraper failed: {e}")
        logger.warning("Continuing without StackOverflow data...")

    if not all_samples:
        raise RuntimeError("Both scrapers failed. No data to process. Aborting.")

    save_json(all_samples, PATHS["raw_combined"])
    logger.info(f"Combined raw total: {len(all_samples)} samples")
    return all_samples


def run_cleaning(samples: list[dict]) -> list[dict]:
    """Clean and validate all samples."""
    from data.scripts.cleaner import clean_dataset

    logger.info("=" * 50)
    logger.info("STAGE 2: Cleaning")
    logger.info("=" * 50)

    before  = len(samples)
    cleaned = clean_dataset(samples)
    log_stage("Cleaning", before, len(cleaned))

    save_json(cleaned, PATHS["cleaned"])
    return cleaned


def run_deduplication(samples: list[dict]) -> list[dict]:
    """Remove exact and near-duplicate samples."""
    from data.scripts.dedupe import dedupe

    logger.info("=" * 50)
    logger.info("STAGE 3: Deduplication")
    logger.info("=" * 50)

    before  = len(samples)
    deduped = dedupe(samples, run_near_dedupe=True)
    log_stage("Deduplication", before, len(deduped))

    save_json(deduped, PATHS["deduped"])
    return deduped


def run_formatting(samples: list[dict]) -> None:
    """Format, sort, split, and save final train/val datasets."""
    from data.scripts.formatter import format_dataset, sort_by_quality, shuffle_and_split, save_jsonl

    logger.info("=" * 50)
    logger.info("STAGE 4: Formatting")
    logger.info("=" * 50)

    before    = len(samples)
    formatted = format_dataset(samples)
    log_stage("Formatting", before, len(formatted))

    sorted_samples       = sort_by_quality(formatted)
    train_data, val_data = shuffle_and_split(sorted_samples)

    save_jsonl(train_data, str(PATHS["train"]))
    save_jsonl(val_data,   str(PATHS["val"]))

    logger.info(f"Final → train: {len(train_data)}, val: {len(val_data)}")

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    skip_scraping: bool = False,
    skip_cleaning: bool = False,
    skip_dedup:    bool = False,
    manual_repos:  list = None,
    github_stars:  int  = 1000,
    github_pages:  int  = 2,
    so_pages:      int  = 3,
    so_min_score:  int  = 10,
) -> None:
    """
    Run the full PY-V data pipeline.

    Skip flags let you resume from a checkpoint:
      skip_scraping=True  → load from data/raw/combined_raw.json
      skip_cleaning=True  → load from data/processed/cleaned/cleaned.json
      skip_dedup=True     → load from data/processed/deduped/deduped.json
    """
    start = time.time()
    logger.info("=" * 50)
    logger.info("PY-V DATA PIPELINE STARTING")
    logger.info("=" * 50)

    ensure_dirs()

    # Stage 1: Scraping
    if skip_scraping:
        logger.info("Skipping scraping — loading from checkpoint")
        samples = load_json(PATHS["raw_combined"])
        logger.info(f"Loaded {len(samples)} raw samples from checkpoint")
    else:
        samples = run_scraping(
            manual_repos=manual_repos,
            github_stars=github_stars,
            github_pages=github_pages,
            so_pages=so_pages,
            so_min_score=so_min_score,
        )

    # Stage 2: Cleaning
    if skip_cleaning:
        logger.info("Skipping cleaning — loading from checkpoint")
        samples = load_json(PATHS["cleaned"])
        logger.info(f"Loaded {len(samples)} cleaned samples from checkpoint")
    else:
        samples = run_cleaning(samples)

    # Stage 3: Deduplication
    if skip_dedup:
        logger.info("Skipping dedup — loading from checkpoint")
        samples = load_json(PATHS["deduped"])
        logger.info(f"Loaded {len(samples)} deduped samples from checkpoint")
    else:
        samples = run_deduplication(samples)

    # Stage 4: Formatting (always runs)
    run_formatting(samples)

    elapsed = time.time() - start
    logger.info("=" * 50)
    logger.info(f"PIPELINE COMPLETE ✔ ({elapsed:.1f}s)")
    logger.info(f"  train.jsonl → {PATHS['train']}")
    logger.info(f"  val.jsonl   → {PATHS['val']}")
    logger.info("=" * 50)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline(
        skip_scraping=False,
        skip_cleaning=False,
        skip_dedup=False,

        manual_repos=[
            ("psf",        "requests"),
            ("pallets",    "flask"),
            ("tiangolo",   "fastapi"),
            ("scrapy",     "scrapy"),
            ("sqlalchemy", "sqlalchemy"),
            ("encode",     "httpx"),
            ("pytest-dev", "pytest"),
            ("numpy",      "numpy"),
        ],

        github_stars=500,
        github_pages=5,

        so_pages=10,
        so_min_score=5,
    )