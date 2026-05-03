import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from peft import PeftModel
from inference.engine.model_loader import load_model
from inference.engine.generator import generate_code

model, tokenizer = load_model()
model = PeftModel.from_pretrained(model, "model/lora")
model.eval()

prompts = [
    "Write a Python function to check if a number is prime.",
    "Write a Python function to reverse a linked list.",
    "Write a Python function to flatten a nested list.",
]

for prompt in prompts:
    print(f"\nINSTRUCTION: {prompt}")
    print("OUTPUT:")
    print(generate_code(model, tokenizer, prompt))
    print("-" * 60)