"""
eval_mbpp.py — PY-V (experiments/)
Scoring test: the model writes a function for each MBPP problem, then the
problem's asserts are run against it. Score = problems passed.

Uses the same prompt and generation path as the chat engine (generate mode),
with greedy decoding so a score can be repeated. Settings come from the
`evaluation` section of configs/config.yaml. Never train on this dataset.

Usage (from repo root):
    python -m experiments.eval_mbpp          # the app's brain + its LoRA adapter
    python -m experiments.eval_mbpp --base   # base model only (no adapter), for comparison
    python -m experiments.eval_mbpp --adapter model/lora_v2   # another adapter
    python -m experiments.eval_mbpp --base --model Qwen/Qwen3-4B-Base   # another base model
    python -m experiments.eval_mbpp --base --model ibm-granite/granite-4.2-3b --native-chat
Results go to {output_dir}/mbpp_{tag}.jsonl (one line per problem) and
mbpp_{tag}_summary.json (score + every setting that can change it), where tag
is "base_<brain>" or "<brain>_<adapter folder>" (+ "_native" with
--native-chat) — see eval_common.py.
"""

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import re
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


def load_problems() -> list:
    ev   = CFG.evaluation
    data = load_dataset(ev.dataset, ev.config, split=ev.split)
    return list(data.select(range(min(ev.num_problems, len(data)))))


def build_task(problem: dict) -> str:
    """Problem text plus the first assert, so the model knows the function name."""
    return (
        f"{problem['prompt']}\n"
        f"Your code should pass this test:\n{problem['test_list'][0]}"
    )


def extract_code(response: str) -> str:
    """Take the first fenced block if the model used one, else the whole answer."""
    match = re.search(r"```(?:python)?\n(.*?)```", response, flags=re.DOTALL)
    return match.group(1) if match else response


def run_tests(problem: dict, code: str, timeout: int) -> tuple:
    """Run the problem's asserts. A timeout is retried once: a busy laptop can
    stall process start-up (seen when a download finished mid-run) — a real
    endless loop still fails the second time."""
    program   = build_program(problem, code)
    ok, error = run_python(program, timeout)
    if not ok and error.startswith("timeout"):
        ok, error = run_python(program, timeout)
    return ok, error


def build_program(problem: dict, code: str) -> str:
    return "\n".join([
        *problem.get("test_imports", []),
        code,
        "",
        *problem["test_list"],
    ])


def main():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    args = parser.parse_args()
    model, tokenizer, tag = load_for_eval(args)
    run(model, tokenizer, tag, args)


def run(model, tokenizer, tag: str, args):
    """Score an already-loaded model (also called by eval_all.py)."""
    ev       = CFG.evaluation
    problems = load_problems()

    ev.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = ev.output_dir / f"mbpp_{tag}.jsonl"

    passed = 0
    start  = time.perf_counter()

    with open(out_path, "w", encoding="utf-8") as out:
        for i, problem in enumerate(problems, 1):
            t = time.perf_counter()

            prompt    = build_prompt("generate", build_task(problem), {}, [])
            response  = generate_from_prompt(model, tokenizer, prompt,
                                             mode="generate", temperature=0.0)
            code      = extract_code(response)
            ok, error = run_tests(problem, code, ev.timeout_seconds)
            seconds   = time.perf_counter() - t

            passed += ok
            out.write(json.dumps({
                "task_id":  problem["task_id"],
                "passed":   ok,
                "error":    error,
                "seconds":  round(seconds, 1),
                "response": response,
            }) + "\n")
            out.flush()

            print(f"[{i}/{len(problems)}] task {problem['task_id']}: "
                  f"{'PASS' if ok else 'FAIL'} ({seconds:.0f}s) "
                  f"- score so far {passed}/{i}", flush=True)

    minutes = (time.perf_counter() - start) / 60
    print(f"\nMBPP score ({tag}): {passed}/{len(problems)} "
          f"= {100 * passed / len(problems):.1f}% in {minutes:.0f} min -> {out_path}")

    write_summary(ev.output_dir / f"mbpp_{tag}_summary.json", args, model, passed, len(problems), minutes)


def write_summary(path: Path, args, model, passed: int, total: int, minutes: float):
    """Score plus every setting that can change it, so runs stay comparable."""
    import peft, torch, transformers

    adapter = None
    if not args.base:
        weights = Path(args.adapter) / "adapter_model.safetensors"
        adapter = {"path": args.adapter,
                   "md5":  hashlib.md5(weights.read_bytes()).hexdigest()}

    summary = {
        "score":        passed,
        "problems":     total,
        "minutes":      round(minutes, 1),
        "date":         datetime.datetime.now().isoformat(timespec="seconds"),
        "base_model":   CFG.model.name,
        "adapter":      adapter,
        "benchmark":    dataclasses.asdict(CFG.evaluation) | {"output_dir": str(CFG.evaluation.output_dir)},
        "prompt_mode":  "generate",
        "prompt_format": model.v_prompt_format,
        "decoding":     {"temperature": 0.0, "max_new_tokens": CFG.model.max_tokens,
                         **dataclasses.asdict(CFG.generation.for_mode("generate"))},
        "device":       torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "versions":     {"torch": torch.__version__, "transformers": transformers.__version__,
                         "peft": peft.__version__},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Settings saved -> {path}")


if __name__ == "__main__":
    main()
