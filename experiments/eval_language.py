"""
eval_language.py — PY-V (experiments/)
Language detection test: does V spot the programming language a message asks
about (inference/engine/language_detector.py)? Cases: experiments/language_cases.json
(many languages, plus traps like "in plain english", "to go", "read a json file"
that must stay Python). No model, instant.

Usage (from repo root):
    python -m experiments.eval_language
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.engine.language_detector import detect_language

CASES = Path(__file__).with_name("language_cases.json")


def main():
    cases  = json.load(open(CASES, encoding="utf-8"))
    passed = 0
    for case in cases:
        found = detect_language(case["message"])
        got   = found["name"] if found else None
        ok    = got == case["language"]
        passed += ok
        if not ok:
            print(f"  FAIL  want {case['language']!s:<12} got {got!s:<12} {case['message']!r}")
    print(f"\nLANGUAGE score: {passed}/{len(cases)}")


if __name__ == "__main__":
    main()
