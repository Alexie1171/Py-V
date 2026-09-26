"""
export.py — PY-V (learning/)
What V learned → training examples for the next retrain (Phase 13). Only what
the owner approved goes in; the retrain itself stays a Kaggle run the owner
starts — nothing here changes the brain.

  learned_answers.jsonl  answers marked "Good answer" in the panel, in the
                         write / fix / improve / explain modes (Python)
  learned_topics.jsonl   study notes of topics the owner approved
                         ("Use for training" in the memory view), as explain
                         examples: "<subtopic> (<topic>)" → the note in sentences

Left out for now (saved, not exported — PROJECT_STATUS.md section 5): chat
answers (the trainer uses the mode templates; V's chat is a persona
conversation) and answers in other languages (the adapter's templates are
Python). Records use the dataset format (instruction / output / metadata with
source + task); build_dataset_v2.py adds every learned_*.jsonl it finds in the
raw source folder or config learning.learned_dir, after the same cleaning.

Usage (from repo root):
    python -m learning.export              # → data/learned/ (config learning.learned_dir)
Then add the files to the Kaggle data upload next to the source files and
re-run the dataset build + training (scripts/pipeline_redo.json).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.training.config_loader import CFG
from learning.store import LearningStore

TRAINED_MODES = {"generate", "debug", "refactor", "explain"}


def answer_records(store: LearningStore) -> tuple:
    """(records, skipped count) from approved answers."""
    records, skipped = [], 0
    for a in store.approved_answers():
        if a["mode"] not in TRAINED_MODES or a["language"]:
            skipped += 1
            continue
        records.append({"instruction": a["question"].strip(), "output": a["answer"].strip(),
                        "metadata": {"source": "learned_chat", "task": a["mode"], "license": "owner",
                                     "approved": a["created"]}})
    return records, skipped


def topic_records(store: LearningStore) -> list:
    """Explain examples from the notes of approved topics."""
    records = []
    for t in store.topics():
        if not t["approved"]:
            continue
        for n in store.notes(t["id"]):
            sentences = [re.sub(r"^[-*•]\s*", "", line).strip() for line in n["text"].split("\n") if line.strip()]
            text = " ".join(s if s.endswith((".", "!", "?", ":")) else s + "." for s in sentences if s)
            if len(text) < 40:
                continue
            records.append({"instruction": f"{n['subtopic']} ({t['topic']})", "output": text,
                            "metadata": {"source": "learned_study", "task": "explain", "license": "notes from web pages",
                                         "topic": t["topic"], "url": n.get("url")}})
    return records


def export(store: LearningStore, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    answers, skipped = answer_records(store)
    topics = topic_records(store)
    for name, records in (("learned_answers.jsonl", answers), ("learned_topics.jsonl", topics)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"answers": len(answers), "answers_skipped": skipped, "topic_notes": len(topics), "folder": str(out_dir)}


def main():
    parser = argparse.ArgumentParser(description="V's approved answers + approved study topics → training records")
    parser.add_argument("--out", default=str(CFG.learning.learned_dir))
    parser.add_argument("--db", default=str(CFG.memory.db_path))
    args = parser.parse_args()
    result = export(LearningStore(args.db), Path(args.out))
    print(f"{result['answers']} approved answers ({result['answers_skipped']} chat / other-language ones left out) "
          f"and {result['topic_notes']} study notes from approved topics -> {result['folder']}")


if __name__ == "__main__":
    main()
