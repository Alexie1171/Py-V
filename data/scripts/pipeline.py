"""
pipeline.py — PY-V Data Pipeline
Orchestrates all stages: scrape → clean → dedupe → format
"""

import json
import logging
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/pipeline.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

PATHS = {
    "raw_github":        Path("data/raw/github/github_raw.json"),
    "raw_stackoverflow": Path("data/raw/stackoverflow/so_raw.json"),
    "raw_combined":      Path("data/raw/combined_raw.json"),
    "cleaned":           Path("data/processed/cleaned/cleaned.json"),
    "deduped":           Path("data/processed/deduped/deduped.json"),
    "train":             Path("data/datasets/train.jsonl"),
    "val":               Path("data/datasets/val.jsonl"),
}


def ensure_dirs():
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


def run_scraping(
    manual_repos:         list  = None,
    use_github_discovery: bool  = True,
    github_stars:         int   = 300,
    github_pages:         int   = 10,
    so_tags:              list  = None,
    so_pages:             int   = 20,
    so_min_score:         int   = 3,
) -> list[dict]:
    from data.scripts.github_scraper        import fetch_multiple_repos
    from data.scripts.stackoverflow_scraper import fetch_stackoverflow_samples

    all_samples = []

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


def run_pipeline(
    skip_scraping: bool = False,
    skip_cleaning: bool = False,
    skip_dedup:    bool = False,
    manual_repos:  list = None,
    github_stars:  int  = 300,
    github_pages:  int  = 10,
    so_pages:      int  = 20,
    so_min_score:  int  = 3,
) -> None:
    start = time.time()
    logger.info("=" * 50)
    logger.info("PY-V DATA PIPELINE STARTING")
    logger.info("=" * 50)

    ensure_dirs()

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

    if skip_cleaning:
        logger.info("Skipping cleaning — loading from checkpoint")
        samples = load_json(PATHS["cleaned"])
        logger.info(f"Loaded {len(samples)} cleaned samples from checkpoint")
    else:
        samples = run_cleaning(samples)

    if skip_dedup:
        logger.info("Skipping dedup — loading from checkpoint")
        samples = load_json(PATHS["deduped"])
        logger.info(f"Loaded {len(samples)} deduped samples from checkpoint")
    else:
        samples = run_deduplication(samples)

    run_formatting(samples)

    elapsed = time.time() - start
    logger.info("=" * 50)
    logger.info(f"PIPELINE COMPLETE ({elapsed:.1f}s)")
    logger.info(f"  train.jsonl → {PATHS['train']}")
    logger.info(f"  val.jsonl   → {PATHS['val']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_pipeline(
        skip_scraping=False,
        skip_cleaning=False,
        skip_dedup=False,

        manual_repos=[
            ("psf",          "requests"),
            ("pallets",      "flask"),
            ("tiangolo",     "fastapi"),
            ("scrapy",       "scrapy"),
            ("sqlalchemy",   "sqlalchemy"),
            ("encode",       "httpx"),
            ("pytest-dev",   "pytest"),
            ("numpy",        "numpy"),
            ("pandas-dev",   "pandas"),
            ("scikit-learn", "scikit-learn"),
            ("aio-libs",     "aiohttp"),
            ("django",       "django"),
            ("celery",       "celery"),
            ("pydantic",     "pydantic"),
            ("python-attrs", "attrs"),
            ("paramiko",     "paramiko"),
        ],

        github_stars=300,
        github_pages=10,

        so_pages=20,
        so_min_score=3,
    )