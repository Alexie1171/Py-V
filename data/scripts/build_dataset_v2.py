"""
build_dataset_v2.py — PY-V Data Pipeline v2
Step 5 of the dataset plan: mix the per-source record files into one
training set.
  1. load {CFG.dataset_v2.output_dir}/{source}.jsonl for every source in the plan
  2. drop exact duplicates, benchmark overlap (MBPP / HumanEval) and records
     too long to train on without cutting the answer
  3. take each source's share (seeded random sample)
  4. split train / val per task (seeded) and write JSONL + a report
Every drop and every shortfall is reported — no silent caps.

Usage (from repo root; on Colab, where the source files live):
    python -m data.scripts.build_dataset_v2
"""

import json
import logging
import random
from collections import Counter, defaultdict

from transformers import AutoTokenizer

from model.training.config_loader import CFG
from data.scripts.dedupe import hash_code
from data.scripts.decontaminate import BenchmarkIndex

# dedupe.py switches on INFO logging at import; keep the download libraries quiet
for _noisy in ("httpx", "httpcore", "urllib3", "fsspec", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def load_source(name: str) -> list:
    path = CFG.dataset_v2.output_dir / f"{name}.jsonl"
    if not path.exists():
        print(f"  WARNING: {path} missing — source {name} skipped")
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean(records: list, benchmarks, tokenizer, max_tokens: int, drops: Counter) -> list:
    """Drop duplicates (output or instruction), benchmark overlap and over-long records."""
    seen, kept = set(), []
    for record in records:
        keys = (hash_code(record["output"]), hash_code(record["instruction"]))
        if seen.intersection(keys):
            drops["duplicate"] += 1
            continue
        seen.update(keys)

        if benchmarks.overlaps(record):
            drops["overlaps MBPP/HumanEval"] += 1
            continue

        n_tokens = len(tokenizer(record["instruction"] + "\n\n" + record["output"])["input_ids"])
        if n_tokens > max_tokens:
            drops["too long"] += 1
            continue

        record["metadata"]["tokens"] = n_tokens
        kept.append(record)
    return kept


def main():
    build = CFG.dataset_v2.build
    rng   = random.Random(build["seed"])

    print("Loading benchmark index (MBPP + HumanEval) and tokenizer ...")
    benchmarks = BenchmarkIndex()
    tokenizer  = AutoTokenizer.from_pretrained(CFG.model.name)

    by_task, report = defaultdict(list), {}
    for name, take in build["take"].items():
        records = load_source(name)
        drops   = Counter()
        limit   = build.get("max_tokens_per_source", {}).get(name, build["max_tokens"])
        usable  = clean(records, benchmarks, tokenizer, limit, drops)
        rng.shuffle(usable)
        chosen  = usable[:take]

        for record in chosen:
            by_task[record["metadata"]["task"]].append(record)

        report[name] = {"loaded": len(records), "usable": len(usable), "taken": len(chosen),
                        "wanted": take, "dropped": dict(drops)}
        short = "  <- SHORT" if len(chosen) < take else ""
        print(f"  {name:<20} loaded {len(records):>5}  usable {len(usable):>5}  "
              f"taken {len(chosen):>5}/{take}{short}  dropped {dict(drops)}")

    out_dir = build["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val = [], []
    for task, records in sorted(by_task.items()):
        rng.shuffle(records)
        n_val = round(len(records) * build["val_share"])
        val   += records[:n_val]
        train += records[n_val:]
    rng.shuffle(train)
    rng.shuffle(val)

    for split, records in (("train", train), ("val", val)):
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total  = len(train) + len(val)
    tasks  = Counter(r["metadata"]["task"] for r in train + val)
    tokens = sorted(r["metadata"]["tokens"] for r in train + val)
    report["total"] = {"train": len(train), "val": len(val),
                       "tasks": dict(tasks), "tokens_median": tokens[len(tokens) // 2],
                       "tokens_p95": tokens[int(len(tokens) * 0.95)]}
    with open(out_dir / "build_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote {len(train)} train + {len(val)} val records -> {out_dir}")
    for task, count in sorted(tasks.items()):
        print(f"  {task:<9} {count:>5}  ({100 * count / total:.0f}%)")
    print(f"  tokens per record: median {report['total']['tokens_median']}, "
          f"95% under {report['total']['tokens_p95']}")


if __name__ == "__main__":
    main()
