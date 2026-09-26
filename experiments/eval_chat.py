"""
eval_chat.py — PY-V (experiments/)
Short chat test: can a brain do what V's everyday chatting needs?
8 questions, each checking one skill: explain a concept, answer from memory
notes, admit what it doesn't know, answer from search results, follow up on
an earlier turn, ask for a search when it lacks information, follow a format,
keep it simple. Chat mode, same generation path as the app, greedy.

The checks are simple keyword rules — a rough signal, not a grade. Every
answer is printed in full so a person can judge it too.

Usage (from repo root):
    python -m experiments.eval_chat --base --model ibm-granite/granite-4.2-3b --native-chat
    python -m experiments.eval_chat --adapter model/lora_v2
Results: {output_dir}/chat_{tag}.jsonl + chat_{tag}_summary.json
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.training.config_loader import CFG
from inference.engine.prompt_builder import build_prompt, build_chat_prompt, uses_chat_format
from inference.engine.generator import generate_from_prompt
from experiments.eval_common import add_model_args, load_for_eval

MAX_NEW = 256

NOTES = (
    "Notes from V's memory:\n"
    "- The user's name is Sam.\n"
    "- The user's laptop has a GTX 1650 graphics card with 4 GB of memory.\n"
    "- The user prefers answers in simple terms.\n"
    "- The user is building an assistant called V."
)

RESULTS = (
    "Search results:\n"
    "[1] Python 3.13 was released on October 7, 2024. It added an experimental free-threaded build.\n"
    "[2] The GIL (global interpreter lock) lets only one thread run Python bytecode at a time.\n"
    "[3] asyncio runs many tasks on a single thread using an event loop."
)

_UNKNOWN = ["don't know", "do not know", "not mentioned", "no information", "not sure", "haven't",
            "have not", "didn't", "did not", "not provided", "not in the", "can't tell", "cannot tell",
            "unknown", "doesn't say", "does not say", "not specified", "no record", "not available",
            "unable", "doesn't include", "does not include", "don't have", "do not have"]


def _has_any(text: str, words) -> bool:
    low = text.lower()
    return any(w in low for w in words)


def _numbered_three(text: str) -> bool:
    numbers = re.findall(r"^\s*([0-9]+)[.)]", text, flags=re.MULTILINE)
    return numbers == ["1", "2", "3"]


def _sentences(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+(?:\s|$)", text) if re.search(r"[A-Za-z]", s)])


QUESTIONS = [
    {"id": "explain", "skill": "explains a basic concept",
     "ask": "What is the difference between a list and a tuple in Python?",
     "check": lambda a: _has_any(a, ["immutable", "cannot be changed", "can't be changed",
                                     "can not be changed", "unchangeable", "cannot be modified",
                                     "can't be modified"])},
    {"id": "memory_recall", "skill": "answers from memory notes",
     "ask": f"{NOTES}\n\nWhich graphics card does my laptop have?",
     "check": lambda a: "1650" in a},
    {"id": "memory_unknown", "skill": "admits it doesn't know (no made-up facts)",
     "ask": f"{NOTES}\n\nWhen is my birthday?",
     "check": lambda a: _has_any(a, _UNKNOWN)},
    {"id": "search_results", "skill": "answers from search results",
     "ask": f"{RESULTS}\n\nAccording to these search results, when was Python 3.13 released?",
     "check": lambda a: "2024" in a and ("october" in a.lower() or "10-07" in a)},
    {"id": "follow_up", "skill": "follows up on an earlier turn",
     "ask": ("Earlier in this chat:\n"
             "Me: What does the len() function do?\n"
             "V: len() returns the number of items in an object, such as the characters in a string "
             "or the elements in a list.\n\n"
             "My next question: Does it work on dictionaries too?"),
     "check": lambda a: _has_any(a, ["yes", "it does", "works"]) and "key" in a.lower()},
    {"id": "tool_use", "skill": "asks for a search when it lacks information",
     "ask": ("You can search V's memory. If you need information you don't have, reply with exactly "
             "one line: SEARCH: <what to look for>. Otherwise answer normally.\n\n"
             "What did I decide last week about which brain to use on my laptop?"),
     "check": lambda a: re.search(r"SEARCH:\s*\S", a) is not None},
    {"id": "format", "skill": "follows a format (exactly 3 numbered tips)",
     "ask": "Give exactly three short tips for writing readable Python code, as a numbered list.",
     "check": _numbered_three},
    {"id": "simple", "skill": "short, simple explanation (1–3 sentences)",
     "ask": "Explain in two short sentences, in simple terms, what a Python virtual environment is.",
     "check": lambda a: 1 <= _sentences(a) <= 3 and _has_any(a, ["package", "librar", "separate",
                                                                   "isolat", "own"])},
]


def main():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    args = parser.parse_args()
    model, tokenizer, tag = load_for_eval(args)
    run(model, tokenizer, tag, args)


def run(model, tokenizer, tag: str, args):
    """Score an already-loaded model (also called by eval_all.py)."""
    out_dir = CFG.evaluation.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with open(out_dir / f"chat_{tag}.jsonl", "w", encoding="utf-8") as out:
        for q in QUESTIONS:
            t      = time.perf_counter()
            formatted = uses_chat_format(model, tokenizer)   # chat brain: persona + the message as written, like the app
            prompt = (build_chat_prompt(q["ask"], {}, None, tokenizer) if formatted
                      else build_prompt("chat", q["ask"], {}))
            answer = generate_from_prompt(model, tokenizer, prompt, mode="chat",
                                          max_tokens=MAX_NEW, temperature=0.0, formatted=formatted)
            row = {"id": q["id"], "skill": q["skill"], "passed": bool(q["check"](answer)),
                   "seconds": round(time.perf_counter() - t, 1), "answer": answer}
            results.append(row)
            out.write(json.dumps(row) + "\n")
            print(f"\n[{q['id']}] {'PASS' if row['passed'] else 'FAIL'} ({row['seconds']:.0f}s) "
                  f"- {q['skill']}\n{answer}", flush=True)

    summary = {"tag": tag, "base_model": CFG.model.name,
               "adapter": None if args.base else args.adapter,
               "prompt_format": model.v_prompt_format,
               "split_rules":   getattr(model, "v_split_rules", None),
               "passed": sum(r["passed"] for r in results), "questions": len(results),
               "by_skill": {r["id"]: r["passed"] for r in results}}
    with open(out_dir / f"chat_{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCHAT score ({tag}): {summary['passed']}/{summary['questions']}")


if __name__ == "__main__":
    main()
