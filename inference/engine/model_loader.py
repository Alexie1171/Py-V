"""
model_loader.py — PY-V (inference/engine/)
Thin inference-layer wrapper around the shared model loader.
Adds load_lora_model() which applies the fine-tuned LoRA adapter
on top of the base quantized model.
"""

import json
from pathlib import Path

from peft import PeftModel
from model.utils.model_loader import load_model
from model.training.config_loader import CFG

__all__ = ["load_model", "load_lora_model"]


def load_lora_model(lora_path: str = None, require_adapter: bool = False):
    """
    Load the brain (CFG.model.name, 4-bit quantized) and apply the fine-tuned
    LoRA adapter saved at lora_path (default: CFG.paths.model_output).

    An adapter only fits the brain it was trained on: one made for another
    base model is refused. When no adapter exists yet (e.g. right after a
    brain upgrade) the plain brain is returned — unless require_adapter is set,
    so scoring a specific adapter never silently scores the plain brain.

    Returns:
        model:     PeftModel with LoRA adapter applied (or the plain brain), in eval mode
        tokenizer: matching AutoTokenizer
    """
    lora_path   = Path(lora_path or CFG.paths.model_output)
    config_file = lora_path / "adapter_config.json"

    if config_file.exists():
        trained_on = json.loads(config_file.read_text(encoding="utf-8")).get("base_model_name_or_path")
        if trained_on != CFG.model.name:
            raise ValueError(f"The adapter at {lora_path} was trained on {trained_on}, but the brain "
                             f"is {CFG.model.name} (config model.name) — retrain it for this brain")
    elif require_adapter:
        raise FileNotFoundError(f"No LoRA adapter at {lora_path}")

    print(f"Loading base model: {CFG.model.name} ...")
    model, tokenizer = load_model()

    if not config_file.exists():
        print(f"No LoRA adapter at {lora_path} yet - running the plain {CFG.model.name}.")
        model.eval()
        return model, tokenizer

    print(f"Applying LoRA adapter from: {lora_path} ...")
    model = PeftModel.from_pretrained(model, str(lora_path))
    model.eval()

    print("LoRA model ready.")
    return model, tokenizer
