"""
eval_common.py — PY-V (experiments/)
Model choice shared by the scoring scripts (eval_mbpp, eval_long_context,
eval_chat): which base model, with or without a LoRA adapter, which prompt
format — and the tag that names their result files.
"""

from pathlib import Path

from model.training.config_loader import CFG
from inference.engine.model_loader import load_model, load_lora_model
from inference.engine.prompt_builder import to_native_chat

DEFAULT_MODEL = CFG.model.name   # captured before --model overrides it


def add_model_args(parser):
    parser.add_argument("--base", action="store_true",
                        help="score a base model without a LoRA adapter")
    parser.add_argument("--adapter", default=str(CFG.paths.model_output),
                        help="LoRA adapter folder to score (default: the app's adapter)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="base model to load (default: config model.name)")
    parser.add_argument("--native-chat", action="store_true",
                        help="wrap prompts in the model's own chat format (chat-tuned brains), thinking off")


def result_tag(args) -> str:
    """"base_<brain>" or "<brain>_<adapter folder>", + "_native" for the chat format.
    The brain is always in the name, so results of different brains never mix
    (files from before 2026-09-26 are named "base", "lora", "lora_v2", "base_<model>" — all Phi-2 unless named)."""
    brain = args.model.split("/")[-1]
    tag   = f"base_{brain}" if args.base else f"{brain}_{Path(args.adapter).name}"
    return tag + ("_native" if args.native_chat else "")


def load_for_eval(args):
    """Load the chosen model. Returns (model, tokenizer, tag, wrap), where
    wrap(prompt) applies the chosen prompt format to a template prompt."""
    tag            = result_tag(args)
    CFG.model.name = args.model
    model, tokenizer = load_model() if args.base else load_lora_model(args.adapter, require_adapter=True)
    model.eval()
    if args.native_chat:
        return model, tokenizer, tag, lambda prompt: to_native_chat(prompt, tokenizer)
    return model, tokenizer, tag, lambda prompt: prompt
