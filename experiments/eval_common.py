"""
eval_common.py — PY-V (experiments/)
Model choice shared by the scoring scripts (eval_mbpp, eval_long_context,
eval_chat): which base model, with or without a LoRA adapter, which prompt
format — and the tag that names their result files.
"""

import json
from pathlib import Path

from model.training.config_loader import CFG
from inference.engine.model_loader import ADAPTER_META, load_model, load_lora_model

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
    parser.add_argument("--split-rules-from", default=None,
                        help="base model: take the text-splitting rules from this brain's tokenizer.json "
                             "(default: config model.split_rules_from, else the brain's own)")
    parser.add_argument("--adapter-all-modes", action="store_true",
                        help="use the adapter in every mode, ignoring its v_adapter.json \"use_in_modes\" "
                             "(the pipeline measures a new adapter this way)")


def adapter_modes(args):
    """Modes the adapter is limited to (its v_adapter.json "use_in_modes"), None = every mode."""
    meta = Path(args.adapter) / ADAPTER_META
    if args.base or getattr(args, "adapter_all_modes", False) or not meta.exists():
        return None
    return json.loads(meta.read_text(encoding="utf-8")).get("use_in_modes")


def result_tag(args) -> str:
    """"base_<brain>" or "<brain>_<adapter folder>", + "_native" for the chat format,
    + "_split-<brain>" when a base brain gets another brain's splitting rules,
    + "_modes-<modes>" when the adapter is only used in some modes (the app's setup).
    The brain is always in the name, so results of different brains never mix
    (files from before 2026-09-26 are named "base", "lora", "lora_v2", "base_<model>" — all Phi-2 unless named)."""
    brain = args.model.split("/")[-1]
    tag   = f"base_{brain}" if args.base else f"{brain}_{Path(args.adapter).name}"
    split = getattr(args, "split_rules_from", None)
    modes = adapter_modes(args)
    return (tag + ("_native" if args.native_chat else "")
            + (f"_split-{split.split('/')[-1]}" if args.base and split and split != args.model else "")
            + (f"_modes-{'-'.join(modes)}" if modes else ""))


def load_for_eval(args):
    """Load the chosen model. Returns (model, tokenizer, tag).
    The prompt format travels with the model (model.v_prompt_format: config, or
    the adapter's v_adapter.json; --native-chat forces the brain's own chat
    format) and the generator applies it — the scripts pass template prompts.
    Splitting rules: --split-rules-from for a base brain; an adapter uses its own.
    Adapter modes: as its v_adapter.json says (the app's setup), or every mode
    with --adapter-all-modes."""
    tag            = result_tag(args)
    CFG.model.name = args.model
    if args.base and getattr(args, "split_rules_from", None):
        CFG.model.split_rules_from = args.split_rules_from
    model, tokenizer = load_model() if args.base else load_lora_model(args.adapter, require_adapter=True)
    model.eval()
    if args.native_chat:
        model.v_prompt_format = "native_chat"
    if getattr(args, "adapter_all_modes", False):
        model.v_adapter_modes = None
    return model, tokenizer, tag
