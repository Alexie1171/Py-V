"""
fetch_sources.py — PY-V Data Pipeline v2
Step 2 of the dataset plan: stream each source (Hugging Face or the old local
dataset), convert rows to PY-V records, write one JSONL per source to
CFG.dataset_v2.output_dir, and report what was kept and why rows were dropped.

Usage (from repo root):
    python -m data.scripts.fetch_sources --sample              # small batch per source, to check quality
    python -m data.scripts.fetch_sources                       # full fetch (counts from config.yaml)
    python -m data.scripts.fetch_sources --only glaive --sample
"""

import argparse
import json
from collections import Counter
from itertools import islice

from model.training.config_loader import CFG
from data.scripts.sources import SOURCES

PREVIEW_RECORDS = 2
PREVIEW_CHARS   = 300


def fetch_source(name: str, sample: bool) -> Counter:
    cfg   = CFG.dataset_v2.sources[name]
    limit = CFG.dataset_v2.sample_size if sample else cfg["fetch"]
    path  = CFG.dataset_v2.output_dir / f"{name}{'_sample' if sample else ''}.jsonl"
    stats = Counter()

    CFG.dataset_v2.output_dir.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in islice(SOURCES[name](cfg, stats), limit):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept"] += 1

            if sample and stats["kept"] <= PREVIEW_RECORDS:
                print(f"  --- example {stats['kept']} ---")
                print(f"  INSTRUCTION: {record['instruction'][:PREVIEW_CHARS]!r}")
                print(f"  OUTPUT:      {record['output'][:PREVIEW_CHARS]!r}")

    print(f"  kept {stats['kept']}/{limit} after scanning {stats['scanned']} rows -> {path}")
    for reason, count in sorted(stats.items()):
        if reason.startswith("rejected"):
            print(f"    {reason}: {count}")
    if stats["kept"] < limit:
        print(f"  WARNING: source ran out — only {stats['kept']} of {limit} records")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true",
                        help=f"fetch only {CFG.dataset_v2.sample_size} records per source")
    parser.add_argument("--only", choices=sorted(SOURCES), help="fetch a single source")
    args = parser.parse_args()

    names  = [args.only] if args.only else list(SOURCES)
    totals = {}

    for name in names:
        print(f"\n=== {name} ===")
        totals[name] = fetch_source(name, args.sample)["kept"]

    print("\n=== summary ===")
    for name, kept in totals.items():
        print(f"  {name:<18} {kept}")


if __name__ == "__main__":
    main()
