"""
eval_fix.py — PY-V (experiments/)
Fix / improve test — the two skills MBPP (writing new code) does not measure.

  fix     — a bug is planted in a working function (dataset-v2 mutations);
            the model gets the failing test + the broken code (debug mode).
            Pass = the returned function passes all the problem's tests.
  improve — a clean function is made clumsy with behaviour-preserving
            rewrites (dataset-v2 unrefactor, tests still pass); the model is
            asked to improve it (refactor mode). Pass = tests still pass AND
            the returned function is simpler (fewer syntax-tree nodes) than
            the clumsy one.

Problems: MBPP sanitized test split from problem 101 on (eval_mbpp.py scores
the first 100) plus MBPP full train/validation/prompt — never in training
(build_dataset_v2 drops every MBPP overlap). MBPP solutions are short, so an
improve question needs only one clumsy rewrite. Questions are built once and
kept in git as experiments/fix_tasks.json, so every model and machine gets
the same ones.

Usage (from repo root):
    python -m experiments.eval_fix --base --model ibm-granite/granite-4.2-3b --native-chat
    python -m experiments.eval_fix --adapter model/lora
Results: {output_dir}/fix_{tag}.jsonl + fix_{tag}_summary.json
"""

import argparse
import ast
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset

from model.training.config_loader import CFG
from inference.engine.prompt_builder import build_prompt
from inference.engine.generator import generate_from_prompt
from experiments.code_runner import run_python
from experiments.eval_common import add_model_args, load_for_eval
from experiments.eval_long_context import answer_code, function_dump
from data.scripts.sources.improve_synthetic import clumsify
from data.scripts.sources.mutations import all_mutations
from data.scripts.sources.unit_tests import run_tests

TASKS      = Path(__file__).resolve().parent / "fix_tasks.json"
FIRST      = 100      # eval_mbpp.py scores problems [0, 100)
PER_KIND   = 40
MAX_NEW    = 400
TIMEOUT    = 10
SEED       = 11
CLUMSY_CFG = {"max_rewrites": 3, "max_tries": 6, "timeout_seconds": 5}
MIN_REWRITES = 1


# ─── Questions (built once, shared by every model) ───────────────────────────

def _called_name(test: str):
    for node in ast.walk(ast.parse(test.strip())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id
    return None


def _function_nodes(code: str, name: str):
    """Syntax-tree size of function `name` in code, or None if it is missing."""
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return sum(1 for _ in ast.walk(node))
    return None


def _fix_question(name: str, buggy: str, result: dict) -> str:
    if result["kind"] == "error":
        failure = f"It raises:\n\n{result['error']}"
    else:
        got = f" (it returns {result['got']})" if result.get("got") else ""
        failure = f"The result is wrong{got}."
    return (f"This test fails:\n\n{result['test']}\n\n{failure}\n\n"
            f"Fix the bug and write the corrected `{name}` function.\n\n```python\n{buggy}\n```")


def _improve_question(name: str, clumsy: str) -> str:
    return (f"Improve this code: make `{name}` simpler and more readable without changing "
            f"what it does.\n\n```python\n{clumsy}\n```")


def _mbpp_rows() -> list:
    """Held-out MBPP problems as {task_id, code, imports, tests}."""
    ev   = CFG.evaluation
    rows = [{"task_id": r["task_id"], "code": r["code"], "imports": "\n".join(r.get("test_imports") or []),
             "tests": r["test_list"]}
            for r in list(load_dataset(ev.dataset, ev.config, split=ev.split))[FIRST:]]
    for split in ("train", "validation", "prompt"):
        rows += [{"task_id": r["task_id"], "code": r["code"], "imports": r["test_setup_code"] or "",
                  "tests": r["test_list"]}
                 for r in load_dataset(ev.dataset, "full", split=split)]
    return rows


def build_tasks(path: Path) -> list:
    print("Building fix / improve questions (runs each MBPP solution's tests) ...", flush=True)
    rng  = random.Random(SEED)
    rows = _mbpp_rows()
    rng.shuffle(rows)

    tasks = {"fix": [], "improve": []}
    for row in rows:
        imports, tests = row["imports"], row["tests"]
        code  = row["code"].replace("\r\n", "\n").strip()
        name  = _called_name(tests[0])
        full  = f"{imports}\n{code}" if imports else code
        if not name or _function_nodes(code, name) is None:
            continue
        if (run_tests(full, tests, TIMEOUT) or {}).get("kind") != "pass":
            continue
        base = {"task_id": row["task_id"], "name": name, "imports": imports, "tests": tests}

        # Improve first — only some solutions allow a clumsy rewrite; the rest become fix questions
        if len(tasks["improve"]) < PER_KIND:
            clumsy, kinds = clumsify(full, tests, CLUMSY_CFG, str(row["task_id"]))
            clumsy = clumsy[len(imports) + 1:] if imports and clumsy.startswith(imports) else clumsy
            clumsy_nodes, clean_nodes = _function_nodes(clumsy, name), _function_nodes(code, name)
            # the rewrite must have made the tested function itself clumsier (not a helper)
            if len(kinds) >= MIN_REWRITES and clumsy_nodes > clean_nodes:
                tasks["improve"].append({**base, "kind": "improve", "mode": "refactor", "rewrites": kinds,
                                         "shown": clumsy, "clumsy_nodes": clumsy_nodes,
                                         "clean_nodes": clean_nodes,
                                         "question": _improve_question(name, clumsy)})
                continue
        if len(tasks["fix"]) < PER_KIND:
            # the bug must be inside the tested function — the question asks to fix that one
            mutations = [m for m in all_mutations(code)
                         if function_dump(m.buggy, name) != function_dump(code, name)]
            rng.shuffle(mutations)
            for mutation in mutations:
                result = run_tests(f"{imports}\n{mutation.buggy}" if imports else mutation.buggy, tests, TIMEOUT)
                if result and result["kind"] in ("assert", "error"):
                    tasks["fix"].append({**base, "kind": "fix", "mode": "debug", "bug": mutation.kind,
                                         "shown": mutation.buggy,
                                         "question": _fix_question(name, mutation.buggy, result)})
                    break
        if len(tasks["fix"]) >= PER_KIND and len(tasks["improve"]) >= PER_KIND:
            break

    all_tasks = [{"id": f"{t['kind']}-{t['task_id']}", **t} for t in tasks["fix"] + tasks["improve"]]
    print(f"  {len(tasks['fix'])} fix + {len(tasks['improve'])} improve questions", flush=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_tasks, f, indent=1)
    return all_tasks


# ─── Grading ─────────────────────────────────────────────────────────────────

def grade(task: dict, response: str) -> dict:
    code = answer_code(response, task["name"])
    if code is None:
        return {"passed": False, "tests_pass": False, "status": f"no `{task['name']}` function in the answer"}
    # The answer is loaded on top of the code it was shown, so helpers defined
    # next to the tested function (other functions, classes, constants) still exist
    program = "\n\n".join(p for p in (task["imports"], task["shown"], code, "\n".join(task["tests"])) if p)
    ok, error = run_python(program, TIMEOUT)
    if not ok and error.startswith("timeout"):
        ok, error = run_python(program, TIMEOUT)
    row = {"passed": ok, "tests_pass": ok, "status": "pass" if ok else error}

    if task["kind"] == "improve":
        nodes = _function_nodes(code, task["name"])
        row["answer_nodes"] = nodes
        simpler = nodes is not None and nodes < task["clumsy_nodes"]
        row["passed"] = ok and simpler
        if ok and not simpler:
            row["status"] = f"tests pass but not simpler ({nodes} vs {task['clumsy_nodes']} nodes)"
    return row


# ─── Run ─────────────────────────────────────────────────────────────────────

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
    tasks   = json.load(open(TASKS, encoding="utf-8")) if TASKS.exists() else build_tasks(TASKS)

    results = []
    with open(out_dir / f"fix_{tag}.jsonl", "w", encoding="utf-8") as out:
        for task in tasks:
            t        = time.perf_counter()
            prompt   = build_prompt(task["mode"], task["question"], {})
            response = generate_from_prompt(model, tokenizer, prompt, mode=task["mode"],
                                            max_tokens=MAX_NEW, temperature=0.0)
            row = {"id": task["id"], "kind": task["kind"], **grade(task, response),
                   "seconds": round(time.perf_counter() - t, 1), "response": response}
            results.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[{task['id']:>12}] {'PASS' if row['passed'] else 'FAIL'} ({row['status'][:70]}) "
                  f"{row['seconds']:.0f}s", flush=True)

    summary = {"tag": tag, "base_model": CFG.model.name, "adapter": None if args.base else args.adapter,
               "prompt_format": model.v_prompt_format,
               "split_rules": getattr(model, "v_split_rules", None)}
    for kind in ("fix", "improve"):
        rows = [r for r in results if r["kind"] == kind]
        summary[kind] = {"passed": sum(r["passed"] for r in rows), "questions": len(rows),
                         "tests_pass": sum(r["tests_pass"] for r in rows)}
    with open(out_dir / f"fix_{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFIX score ({tag}): fix {summary['fix']['passed']}/{summary['fix']['questions']}, "
          f"improve {summary['improve']['passed']}/{summary['improve']['questions']} "
          f"(tests still pass in {summary['improve']['tests_pass']})")


if __name__ == "__main__":
    main()
