"""
model_loader.py — PY-V (inference/engine/)
Thin inference-layer wrapper around the shared model loader.
Adds load_lora_model() which applies the fine-tuned LoRA adapter
on top of the base quantized model.
"""

from peft import PeftModel
from model.utils.model_loader import load_model
from model.training.config_loader import CFG

__all__ = ["load_model", "load_lora_model"]


def load_lora_model():
    """
    Load the base Phi-2 model (4-bit quantized) and apply the
    fine-tuned LoRA adapter saved at CFG.paths.model_output.

    Returns:
        model:     PeftModel with LoRA adapter applied, set to eval mode
        tokenizer: matching AutoTokenizer
    """
    lora_path = str(CFG.paths.model_output)

    print(f"Loading base model: {CFG.model.name} ...")
    model, tokenizer = load_model()

    print(f"Applying LoRA adapter from: {lora_path} ...")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    print("LoRA model ready.")
    return model, tokenizer