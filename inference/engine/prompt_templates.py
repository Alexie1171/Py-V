# inference/engine/prompt_templates.py

TEMPLATES = {

"generate": """### Instruction:
You are a Python expert. Write a complete, working Python function for the following task. Output code only, no explanations.

{retrieved_context}{context}
Task: {user_input}

### Answer:
""",

"debug": """### Instruction:
A Python program has the following error. Identify the bug and write the corrected code. Then explain the fix in one sentence.

Error reported: {user_input}

{retrieved_context}{context}

### Answer:
""",

"explain": """### Instruction:
Explain the following Python concept in plain English. Write only sentences and paragraphs. Do not write any code, functions, or code comments. If you want to give an example, describe it in words only — do not show code syntax.

{context}
Concept: {user_input}

### Answer:
""",

"refactor": """### Instruction:
You are a Python expert. Refactor and improve the following code. Return only the improved code.

{retrieved_context}{context}
Code: {user_input}

### Answer:
""",

"chat": """### Instruction:
You are a helpful Python assistant. Answer the following question using only plain English sentences and paragraphs. Do not write code, functions, or code comments. Describe any examples in words only.

{context}
Question: {user_input}

### Answer:
""",
}