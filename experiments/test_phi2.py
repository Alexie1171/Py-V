from inference.engine.model_loader import load_model
from inference.engine.generator import generate_code

# Load model once
model, tokenizer = load_model()

prompt = "Write a Python function to check if a number is prime."

output = generate_code(model, tokenizer, prompt)

print(output)