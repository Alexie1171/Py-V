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
    parser.add_argument("--adapter-modes", nargs="+", default=None,
                        help="use the adapter only in these modes (the app's setup, e.g. debug refactor) - "
                             "overrides v_adapter.json (a Kaggle copy of the adapter lacks the hand-added setting)")
    parser.add_argument("--rag", action="store_true",
                        help="add RAG examples from the training set to write / fix / improve prompts "
                             "(index: config rag.index_path; strong matches only, config rag.min_score)")
    parser.add_argument("--rag-min-score", type=float, default=None,
                        help="override config rag.min_score for this run")


def adapter_modes(args):
    """Modes the adapter is limited to (--adapter-modes, else its v_adapter.json "use_in_modes"), None = every mode."""
    if args.base or getattr(args, "adapter_all_modes", False):
        return None
    if getattr(args, "adapter_modes", None):
        return list(args.adapter_modes)
    meta = Path(args.adapter) / ADAPTER_META
    if not meta.exists():
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
            + (f"_modes-{'-'.join(modes)}" if modes else "")
            + ("_rag" if getattr(args, "rag", False) else ""))


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
    elif getattr(args, "adapter_modes", None) and not args.base:
        model.v_adapter_modes = list(args.adapter_modes)
    return model, tokenizer, tag


_retriever = {}


def retrieve(args, query: str, mode: str) -> list:
    """RAG examples for a test question (--rag): the app's retriever and cut-off, only in config
    rag.active_modes; [] without --rag. The index must exist (python -m retrieval.indexer)."""
    if not getattr(args, "rag", False) or mode not in CFG.rag.active_modes:
        return []
    if "r" not in _retriever:
        from retrieval.retriever import Retriever
        min_score = args.rag_min_score if args.rag_min_score is not None else CFG.rag.min_score
        _retriever["r"] = Retriever(index_path=str(CFG.rag.index_path), device=CFG.rag.device, min_score=min_score)
    return _retriever["r"].search(query, k=CFG.rag.top_k)
