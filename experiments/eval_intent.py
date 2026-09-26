"""
eval_intent.py — PY-V (experiments/)
Mode detection test: does V pick the right mode (chat / explain / generate /
debug / refactor) for the messages in experiments/intent_cases.json?

  rules only (no model, instant):       word rules of inference/engine/controller.py
  --brain (loads the brain, one step each): rules, and the brain for the ones flagged
                                        "unclear" — what the app does (intent_classifier.py)

The cases were written together with the rules, so a perfect rules score only
shows the rules do what was meant — new real messages that go wrong belong in
the cases file.

Usage (from repo root):
    python -m experiments.eval_intent
    python -m experiments.eval_intent --brain              # the app's brain + adapter (model/lora)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.training.config_loader import CFG
from inference.engine.controller import Controller

CASES = Path(__file__).with_name("intent_cases.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain", action="store_true", help="let the brain decide the unclear ones (loads the model)")
    from experiments.eval_common import add_model_args
    add_model_args(parser)
    args = parser.parse_args()

    cases      = json.load(open(CASES, encoding="utf-8"))
    controller = Controller()
    model = tokenizer = None
    tag = "rules"
    if args.brain:
        from experiments.eval_common import load_for_eval
        from inference.engine.intent_classifier import classify_with_brain
        model, tokenizer, tag = load_for_eval(args)
        tag = f"rules+brain_{tag}"

    rows, brain_seconds = [], []
    for case in cases:
        intent = controller.detect_mode(case["message"])
        mode   = intent.mode
        picked = None
        if model is not None and "unclear" in intent.flags:
            start  = time.perf_counter()
            picked = classify_with_brain(model, tokenizer, case["message"])
            brain_seconds.append(time.perf_counter() - start)
            mode = picked or mode
        ok = mode == case["mode"]
        rows.append({"message": case["message"], "expected": case["mode"], "rules": intent.mode,
                     "flags": intent.flags, "brain": picked, "final": mode, "ok": ok})
        short = case["message"].replace("\n", " ")[:60]
        print(f"  {'PASS' if ok else 'FAIL'}  {case['mode']:<8} got {mode:<8} "
              f"{'(brain: ' + str(picked) + ') ' if 'unclear' in intent.flags and model is not None else ''}"
              f"{'[unclear] ' if 'unclear' in intent.flags else ''}{short!r}")

    passed  = sum(r["ok"] for r in rows)
    unclear = sum("unclear" in r["flags"] for r in rows)
    print(f"\nMODE score ({tag}): {passed}/{len(rows)} - {unclear} unclear"
          + (f", brain {sum(brain_seconds) / len(brain_seconds):.1f} s each" if brain_seconds else ""))

    out = CFG.evaluation.output_dir / f"intent_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"passed": passed, "cases": len(rows), "unclear": unclear,
               "brain_seconds_avg": round(sum(brain_seconds) / len(brain_seconds), 2) if brain_seconds else None,
               "rows": rows}, open(out, "w", encoding="utf-8"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
