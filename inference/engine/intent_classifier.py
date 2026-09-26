"""
intent_classifier.py — PY-V (inference/engine/)
The brain picks the mode for a message the word rules could not place clearly
(controller.py flags it "unclear"; ChatEngine asks when config
intent.brain_for_unclear is on). One question in the brain's own prompt format,
adapter off (the plain brain follows instructions best) — and no writing: ONE
forward pass, then the five mode words are compared by how likely the brain is
to start its answer with each (all spellings: "debug", "Debug", " debug").
The first pieces of the five words differ, so the first token decides.

2026-09-26 laptop checks (GTX 1650, Granite chat): letting it write the word
and scoring the words gave the same picks — 28/41 test messages alone (it says
"chat" / "explain" for many code requests), 39/41 with the word rules = no gain
over the rules alone — at 3.4 s per message (reading the question is the slow
part). So config intent.brain_for_unclear is off; kept for a better question
(examples in it) or a faster brain.
"""

from contextlib import nullcontext

import torch

from inference.engine.prompt_builder import build_intent_prompt, format_for_model

MODES = ("chat", "explain", "generate", "debug", "refactor")


def _mode_tokens(tokenizer) -> dict:
    """mode → the first-token ids of its spellings (kept on the tokenizer after the first call)."""
    cached = getattr(tokenizer, "v_mode_tokens", None)
    if cached is None:
        cached = {mode: sorted({tokenizer(spelling, add_special_tokens=False)["input_ids"][0]
                                for spelling in (mode, mode.capitalize(), " " + mode)})
                  for mode in MODES}
        firsts = [i for ids in cached.values() for i in ids]
        if len(firsts) != len(set(firsts)):
            raise ValueError("Two mode words start with the same token - the brain can't tell them apart")
        tokenizer.v_mode_tokens = cached
    return cached


def mode_scores(model, tokenizer, message: str) -> dict:
    """{mode: log-probability that the brain's answer starts with that mode word}."""
    prompt = format_for_model(build_intent_prompt(message), model, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    off    = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()

    with torch.inference_mode(), off:
        logits = model(**inputs).logits[0, -1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    return {mode: torch.logsumexp(log_probs[ids], dim=0).item() for mode, ids in _mode_tokens(tokenizer).items()}


def classify_with_brain(model, tokenizer, message: str) -> str:
    """The mode the brain finds most likely for this message."""
    scores = mode_scores(model, tokenizer, message)
    return max(scores, key=scores.get)
