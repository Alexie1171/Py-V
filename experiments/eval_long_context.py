"""
eval_long_context.py — PY-V (experiments/)
Long-question test: "find the bug in this long file".

Builds Python modules of growing size from MBPP train/validation/prompt
solutions (never the test split that eval_mbpp.py scores on), plants one bug
in one function with the dataset-v2 mutation operators, and asks the model
(debug mode, same generation path as the app) for the corrected function.
Graded by running that function's own MBPP tests with the model's answer
loaded on top of the buggy module.

Also records prompt length and peak GPU memory per question, so it shows how
long a question each model handles on this laptop before it runs out of GPU
memory or of context window. Every model gets the same questions: built once
and kept in git as experiments/longctx_tasks.json, so the laptop and Colab
score the same set.

Usage (from repo root):
    python -m experiments.eval_long_context --adapter model/lora_v2
    python -m experiments.eval_long_context --base [--model NAME] [--native-chat]
Results: {output_dir}/longctx_{tag}.jsonl + longctx_{tag}_summary.json
"""

import argparse
import ast
import json
import os
import random
import re
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from model.training.config_loader import CFG
from inference.engine.prompt_builder import build_prompt
from inference.engine.generator import generate_from_prompt
from experiments.code_runner import run_python
from experiments.eval_common import add_model_args, load_for_eval
from data.scripts.sources.mutations import all_mutations
from data.scripts.sources.unit_tests import run_tests

SIZES    = [500, 1000, 1500, 3000, 6000]   # module size in tokens (the kept set was built with Phi-2's tokenizer)
PER_SIZE = 2                               # questions per size (bug early / late in the file)
MAX_NEW  = 320                             # answer budget: one fixed function + a sentence
SEED     = 7
TIMEOUT  = 10
TASKS    = Path(__file__).resolve().parent / "longctx_tasks.json"


# ─── Questions (built once, shared by every model) ───────────────────────────

def _called_name(test: str):
    for node in ast.walk(ast.parse(test.strip())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id
    return None


def _snippet_pool(tokenizer) -> list:
    """MBPP solutions (non-test splits) that parse and pass their own tests here."""
    pool = []
    for split in ("train", "validation", "prompt"):
        for row in load_dataset(CFG.evaluation.dataset, "full", split=split):
            code  = row["code"].replace("\r\n", "\n").strip()
            tests = [row["test_setup_code"]] * bool(row["test_setup_code"]) + row["test_list"]
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            fname = _called_name(row["test_list"][0])
            if fname not in names:
                continue
            result = run_tests(code, tests, TIMEOUT)
            if not result or result["kind"] != "pass":
                continue
            pool.append({"task_id": row["task_id"], "code": code, "tests": tests, "name": fname,
                         "names": sorted(names), "tokens": len(tokenizer(code)["input_ids"])})
    return pool


def _plant_bug(snippet: dict, module_parts: list, index: int, rng):
    """First mutation of the target snippet that makes its tests fail inside the module."""
    mutations = all_mutations(snippet["code"])
    rng.shuffle(mutations)
    for mutation in mutations:
        parts  = module_parts[:index] + [mutation.buggy] + module_parts[index + 1:]
        module = "\n\n\n".join(parts)
        result = run_tests(module, snippet["tests"], TIMEOUT)
        if result and result["kind"] in ("assert", "error"):
            return module, mutation, result
    return None


def _question(snippet: dict, result: dict, module: str) -> str:
    if result["kind"] == "error":
        failure = f"It raises:\n\n{result['error']}"
    else:
        got = f" (it returns {result['got']})" if result.get("got") else ""
        failure = f"The result is wrong{got}."
    return (f"This test fails:\n\n{result['test']}\n\n{failure}\n\n"
            f"Find the bug in the module below and write the corrected `{snippet['name']}` function.\n\n"
            f"```python\n{module}\n```")


def build_tasks(path: Path) -> list:
    tokenizer = AutoTokenizer.from_pretrained(CFG.model.name)   # sizes counted in the config brain's tokens
    print("Building questions (runs each MBPP solution's tests once) ...", flush=True)
    pool = _snippet_pool(tokenizer)
    rng  = random.Random(SEED)
    print(f"  {len(pool)} usable functions", flush=True)

    tasks = []
    for size in SIZES:
        for position in (0.25, 0.75)[:PER_SIZE]:
            for _attempt in range(20):
                order = pool[:]
                rng.shuffle(order)
                chosen, names, tokens = [], set(), 0
                for snippet in order:
                    if names & set(snippet["names"]):
                        continue
                    chosen.append(snippet)
                    names |= set(snippet["names"])
                    tokens += snippet["tokens"]
                    if tokens >= size:
                        break
                index   = min(int(len(chosen) * position), len(chosen) - 1)
                planted = _plant_bug(chosen[index], [s["code"] for s in chosen], index, rng)
                if planted:
                    break
            if not planted:
                print(f"  size {size}: no bug the tests catch after 20 tries - question skipped")
                continue
            module, mutation, result = planted
            target = chosen[index]
            tasks.append({
                "id":        f"{size}-{int(position * 100)}",
                "size":      size,
                "position":  position,
                "functions": len(chosen),
                "name":      target["name"],
                "task_id":   target["task_id"],
                "bug":       mutation.kind,
                "module":    module,
                "tests":     target["tests"],
                "question":  _question(target, result, module),
            })
            print(f"  question {tasks[-1]['id']}: {len(chosen)} functions, "
                  f"bug '{mutation.kind}' in {target['name']}", flush=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=1)
    return tasks


# ─── Grading ─────────────────────────────────────────────────────────────────

def answer_code(response: str, name: str):
    """The corrected function from the answer (fenced or not), with any imports above it."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", response, flags=re.DOTALL)
    for text in blocks + [response]:
        lines = text.split("\n")
        start = next((i for i, line in enumerate(lines)
                      if re.match(rf"\s*def {re.escape(name)}\s*\(", line)), None)
        if start is None:
            continue
        imports = [line.strip() for line in lines[:start] if re.match(r"\s*(import|from)\s", line)]
        for end in range(len(lines), start, -1):
            chunk = textwrap.dedent("\n".join(lines[start:end]))
            try:
                ast.parse(chunk)
            except SyntaxError:
                continue
            return "\n".join(imports + [chunk])
    return None


def grade(task: dict, response: str) -> tuple:
    code = answer_code(response, task["name"])
    if code is None:
        return False, f"no `{task['name']}` function in the answer"
    program = "\n\n\n".join([task["module"], code, "\n".join(task["tests"])])
    ok, error = run_python(program, TIMEOUT)
    if not ok and error.startswith("timeout"):
        ok, error = run_python(program, TIMEOUT)
    return ok, error


# ─── Run ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    args = parser.parse_args()

    out_dir = CFG.evaluation.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks   = json.load(open(TASKS, encoding="utf-8")) if TASKS.exists() else build_tasks(TASKS)

    model, tokenizer, tag, wrap = load_for_eval(args)
    context   = getattr(model.config, "max_position_embeddings", None)
    weights_mb = torch.cuda.memory_allocated() / 2**20
    print(f"{tag}: context window {context} tokens, weights on GPU {weights_mb:.0f} MB", flush=True)

    results, first_oom = [], None
    with open(out_dir / f"longctx_{tag}.jsonl", "w", encoding="utf-8") as out:
        for task in tasks:
            prompt   = wrap(build_prompt("debug", task["question"], {}))
            n_prompt = len(tokenizer(prompt)["input_ids"])
            row      = {"id": task["id"], "size": task["size"], "prompt_tokens": n_prompt,
                        "passed": False, "status": "", "seconds": 0.0, "peak_mb": None, "response": ""}

            if context and n_prompt + MAX_NEW > context:
                row["status"] = "longer than context window"
            elif first_oom is not None and task["size"] > first_oom:
                row["status"] = "skipped - bigger than first out-of-memory"
            else:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t = time.perf_counter()
                try:
                    response = generate_from_prompt(model, tokenizer, prompt, mode="debug",
                                                    max_tokens=MAX_NEW, temperature=0.0)
                    row["passed"], row["status"] = grade(task, response)
                    row["status"]   = row["status"] or "pass"
                    row["response"] = response
                except torch.cuda.OutOfMemoryError:
                    first_oom     = task["size"] if first_oom is None else first_oom
                    row["status"] = "out of GPU memory"
                    torch.cuda.empty_cache()
                row["seconds"] = round(time.perf_counter() - t, 1)
                row["peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)

            results.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[{task['id']:>7}] {n_prompt:>5} prompt tokens: "
                  f"{'PASS' if row['passed'] else 'FAIL'} ({row['status'][:70]}) "
                  f"{row['seconds']:.0f}s, peak {row['peak_mb']} MB", flush=True)

    by_size = defaultdict(list)
    for row in results:
        by_size[row["size"]].append(row)
    summary = {
        "tag": tag, "base_model": CFG.model.name, "adapter": None if args.base else args.adapter,
        "prompt_format": "native chat" if args.native_chat else "template",
        "context_window": context, "weights_mb": round(weights_mb), "max_new_tokens": MAX_NEW,
        "passed": sum(r["passed"] for r in results), "questions": len(results),
        "by_size": {size: {"passed": sum(r["passed"] for r in rows), "questions": len(rows),
                           "prompt_tokens": max(r["prompt_tokens"] for r in rows),
                           "peak_mb": max((r["peak_mb"] or 0) for r in rows),
                           "status": sorted({r["status"] if not r["passed"] else "pass" for r in rows})}
                    for size, rows in sorted(by_size.items())},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with open(out_dir / f"longctx_{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nLONG-CONTEXT score ({tag}): {summary['passed']}/{summary['questions']}")
    for size, s in summary["by_size"].items():
        print(f"  ~{size:>5} tokens: {s['passed']}/{s['questions']}  peak {s['peak_mb']} MB  {s['status']}")


if __name__ == "__main__":
    main()
