"""
eval_all.py — PY-V (experiments/)
All four tests with ONE model load: MBPP (write code), long-file bug fixes,
chat, fix/improve. Loading a 4-bit brain takes about a minute, so on Colab
this saves several minutes per brain. A test that crashes does not stop the
others; the exit code is non-zero if any failed.

Usage (from repo root):
    python -m experiments.eval_all --base --model ibm-granite/granite-4.2-3b --native-chat --skip-done
    python -m experiments.eval_all --adapter model/lora
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.training.config_loader import CFG
from experiments import eval_chat, eval_fix, eval_long_context, eval_mbpp
from experiments.eval_common import add_model_args, load_for_eval, result_tag

TESTS = {"mbpp": eval_mbpp, "longctx": eval_long_context, "chat": eval_chat, "fix": eval_fix}


def main():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--skip-done", action="store_true",
                        help="skip tests whose summary file already exists (continue after a stop)")
    parser.add_argument("--only", nargs="+", choices=list(TESTS), default=None,
                        help="run only these tests (the RAG stage: mbpp fix)")
    args = parser.parse_args()

    out_dir = CFG.evaluation.output_dir
    tag     = result_tag(args)
    todo    = {p: m for p, m in TESTS.items()
               if (not args.only or p in args.only)
               and not (args.skip_done and (out_dir / f"{p}_{tag}_summary.json").exists())}
    if not todo:
        print(f"{tag}: all tests already done")
        return

    model, tokenizer, tag = load_for_eval(args)
    failed = []
    for prefix, module in todo.items():
        print(f"\n########## {tag}: {prefix} ##########", flush=True)
        try:
            module.run(model, tokenizer, tag, args)
        except Exception:
            traceback.print_exc()
            failed.append(prefix)
        torch.cuda.empty_cache()

    if failed:
        print(f"\n{tag}: FAILED tests: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
