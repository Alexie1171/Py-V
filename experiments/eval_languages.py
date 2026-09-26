"""
eval_languages.py — PY-V (experiments/)
Other-languages test (owner, 2026-09-26: "any language from assembly to the
latest"): small programs asked for the way a user would, answered through the
app's path — language_detector.detect_language → OTHER_LANGUAGE_TEMPLATES
(generate) → the brain with the adapter off (it was trained on Python only).

Graded by running the program when the language's tools are installed
(node, gcc, g++, bash, go, rustc, javac/java, ruby, php, lua, runghc, pwsh)
and checking what it prints; otherwise only that the answer is a code block
for that language ("format"). Both counts are reported separately — a missing
tool is never counted as a failure.

Usage (from repo root; loads the brain — Kaggle / Colab):
    python -m experiments.eval_languages --adapter model/lora --adapter-modes debug refactor
    python -m experiments.eval_languages --check-runners      # which tools this machine has (no brain)
Results: {output_dir}/languages_{tag}.jsonl + languages_{tag}_summary.json
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.training.config_loader import CFG

TIMEOUT = 30
MAX_NEW = 400

# (message as a user would type it, the language she should recognise, what the program must print — None = format only)
TASKS = [
    ("Write a complete JavaScript program that prints the sum of the numbers from 1 to 10.", "JavaScript", "55"),
    ("Write a complete C program that prints the factorial of 5.", "C", "120"),
    ("Write a complete C++ program that prints the largest of the numbers 4, 17 and 9.", "C++", "17"),
    ("Write a bash script that prints the numbers 1 to 5 on one line, separated by spaces.", "Bash", "1 2 3 4 5"),
    ("Write a complete Go program that prints hello world.", "Go", "hello world"),
    ("Write a complete Rust program that prints the 10th Fibonacci number, where fib(1) = 1 and fib(2) = 1.", "Rust", "55"),
    ('Write a complete Java program with a class named Main that prints the reverse of the string "abc".', "Java", "cba"),
    ('Write a Ruby script that prints the length of the string "hello".', "Ruby", "5"),
    ("Write a PHP script that prints the sum of 2 and 3.", "PHP", "5"),
    ("Write a Lua script that prints the square of 7.", "Lua", "49"),
    ("Write a Haskell program that prints the sum of the numbers from 1 to 10.", "Haskell", "55"),
    ("Write a PowerShell script that prints the numbers 1 to 3.", "PowerShell", "1"),
    ("Write a TypeScript function that adds two numbers and prints add(2, 3).", "TypeScript", "5"),
    ("give an example of a kotlin data class for a user with a name and an age", "Kotlin", None),
    ("write a C# method that checks whether a number is even", "C#", None),
    ("write an SQL query that selects all users older than 30 from a users table", "SQL", None),
    ("write hello world in x86 assembly (nasm)", "Assembly", None),
    ("write a swift function that returns the larger of two ints", "Swift", None),
    ("fibonacci in elixir", "Elixir", None),
    ("write a dockerfile for a small node app", "Dockerfile", None),
]

# language → (file name, [commands]; {f} = the file, {d} = its folder, {exe} = a built program)
RUNNERS = {
    "JavaScript": ("main.js", [["node", "{f}"]]),
    "C":          ("main.c", [["gcc", "{f}", "-o", "{exe}", "-lm"], ["{exe}"]]),
    "C++":        ("main.cpp", [["g++", "{f}", "-o", "{exe}"], ["{exe}"]]),
    "Bash":       ("main.sh", [["bash", "{f}"]]),
    "Go":         ("main.go", [["go", "run", "{f}"]]),
    "Rust":       ("main.rs", [["rustc", "{f}", "-o", "{exe}"], ["{exe}"]]),
    "Java":       ("Main.java", [["javac", "{f}"], ["java", "-cp", "{d}", "Main"]]),
    "Ruby":       ("main.rb", [["ruby", "{f}"]]),
    "PHP":        ("main.php", [["php", "{f}"]]),
    "Lua":        ("main.lua", [["lua", "{f}"]]),
    "Haskell":    ("Main.hs", [["runghc", "{f}"]]),
    "PowerShell": ("main.ps1", [["pwsh", "-NoProfile", "-File", "{f}"]]),
    "TypeScript": ("main.ts", [["node", "--experimental-strip-types", "--no-warnings", "{f}"]]),
}


def runner_available(language: str) -> bool:
    if language not in RUNNERS:
        return False
    tools = {cmd[0] for cmd in RUNNERS[language][1] if not cmd[0].startswith("{")}
    if not all(shutil.which(t) for t in tools):
        return False
    if language == "Bash" and os.name == "nt" and "system32" in shutil.which("bash").lower():
        return False               # Windows' bash.exe is the WSL launcher — no Linux installed behind it
    if language == "TypeScript":   # node 22.6+ runs .ts with its type-stripping flag
        try:
            out = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=10).stdout
            major, minor = (int(x) for x in out.strip().lstrip("v").split(".")[:2])
            return (major, minor) >= (22, 6)
        except Exception:
            return False
    return True


def code_block(answer: str):
    """(tag, code) of the first fenced block, or (None, None)."""
    match = re.search(r"```([\w+#.-]*)[^\n]*\n(.*?)```", answer, re.S)
    return (match.group(1).lower(), match.group(2)) if match else (None, None)


def run_program(language: str, code: str) -> tuple:
    """(ran ok, what it printed or the error)."""
    name, commands = RUNNERS[language]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / name
        if language == "PHP" and "<?php" not in code:
            code = "<?php\n" + code
        path.write_text(code, encoding="utf-8")
        exe = str(Path(d) / ("main.exe" if os.name == "nt" else "main"))
        out = ""
        for cmd in commands:
            cmd = [c.format(f=str(path), d=d, exe=exe) for c in cmd]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, cwd=d)
            except subprocess.TimeoutExpired:
                return False, "timeout"
            except OSError as e:
                return False, str(e)
            if res.returncode != 0:
                return False, (res.stderr or res.stdout)[-400:]
            out = res.stdout
        return True, out


def grade(language: str, expected, answer: str) -> dict:
    tag, code = code_block(answer)
    row = {"has_block": code is not None, "tag": tag}
    if code is None:
        return {**row, "passed": False, "graded": "run" if expected and runner_available(language) else "format",
                "status": "no code block"}
    if expected is None or not runner_available(language):
        return {**row, "passed": True, "graded": "format", "status": f"code block tagged {tag!r}"}
    ok, out = run_program(language, code)
    printed = re.sub(r"\s+", " ", out).strip().lower()
    passed  = ok and expected.lower() in printed
    return {**row, "passed": passed, "graded": "run", "status": printed[:200] if ok else f"error: {out[:200]}"}


def main():
    from experiments.eval_common import add_model_args, load_for_eval
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--check-runners", action="store_true", help="list the tools this machine has, no brain")
    args = parser.parse_args()
    if args.check_runners:
        for language in RUNNERS:
            print(f"  {language:<11} {'yes' if runner_available(language) else 'no'}")
        return
    model, tokenizer, tag = load_for_eval(args)
    run(model, tokenizer, tag, args)


def run(model, tokenizer, tag: str, args):
    from inference.engine.generator import generate_from_prompt
    from inference.engine.language_detector import detect_language
    from inference.engine.prompt_builder import build_prompt

    out_dir = CFG.evaluation.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out_dir / f"languages_{tag}.jsonl", "w", encoding="utf-8") as out:
        for message, language, expected in TASKS:
            t        = time.perf_counter()
            detected = detect_language(message)
            if detected is None or detected["name"] != language:
                row = {"message": message, "language": language, "detected": detected and detected["name"],
                       "passed": False, "graded": "detect", "status": "language not recognised", "response": ""}
            else:
                prompt   = build_prompt("generate", message, {}, language=detected)
                response = generate_from_prompt(model, tokenizer, prompt, mode="generate", max_tokens=MAX_NEW,
                                                temperature=0.0, adapter=False)
                row = {"message": message, "language": language, "detected": detected["name"],
                       **grade(language, expected, response), "response": response}
            row["seconds"] = round(time.perf_counter() - t, 1)
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[{language:>11}] {'PASS' if row['passed'] else 'FAIL'} ({row['graded']}: {row['status'][:60]}) "
                  f"{row['seconds']:.0f}s", flush=True)

    ran = [r for r in rows if r["graded"] == "run"]
    fmt = [r for r in rows if r["graded"] == "format"]
    summary = {"tag": tag, "base_model": CFG.model.name, "adapter": None if args.base else args.adapter,
               "ran":    {"passed": sum(r["passed"] for r in ran), "questions": len(ran),
                          "languages": sorted(r["language"] for r in ran)},
               "format": {"passed": sum(r["passed"] for r in fmt), "questions": len(fmt)},
               "not_recognised": [r["language"] for r in rows if r["graded"] == "detect"],
               "passed": sum(r["passed"] for r in rows), "questions": len(rows)}
    with open(out_dir / f"languages_{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nLANGUAGES ({tag}): programs that ran and printed the right thing {summary['ran']['passed']}/"
          f"{summary['ran']['questions']}, code blocks where no tool was installed {summary['format']['passed']}/"
          f"{summary['format']['questions']}, not recognised: {summary['not_recognised'] or 'none'}")


if __name__ == "__main__":
    main()
