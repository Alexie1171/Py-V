# inference/engine/prompt_templates.py

TEMPLATES = {

"generate": """### Instruction:
You are a Python expert. Write a complete, working Python function for the following task. Output code only, no explanations.

{context}
Task: {user_input}

### Answer:
""",

"debug": """### Instruction:
A Python program has the following error. Identify the bug and write the corrected code. Then explain the fix in one sentence.

Error reported: {user_input}

{context}

### Answer:
""",

"explain": """### Instruction:
Explain the following Python concept in plain English. Do not write any code. Use simple sentences only.

{context}
Concept: {user_input}

### Answer:
""",

"refactor": """### Instruction:
You are a Python expert. Refactor and improve the following code. Return only the improved code.

{context}
Code: {user_input}

### Answer:
""",

"chat": """### Instruction:
You are a helpful Python assistant. Answer the following question in plain English only. Do not write code.

{context}
Question: {user_input}

### Answer:
""",
}