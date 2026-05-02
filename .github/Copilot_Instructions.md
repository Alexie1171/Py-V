# Copilot Instructions for PY-V

## 🧠 Purpose

This document defines how AI assistants (e.g., Copilot-style tools) should behave when contributing to the PY-V codebase.

It ensures:

* Consistency
* Clean architecture
* No deviation from project goals

---

## 📌 Core Principles

### 1. Modularity First

* Each file must have a **single responsibility**
* No monolithic scripts
* Reusable components only

---

### 2. Strict Folder Discipline

AI must NEVER:

* Place files in incorrect directories
* Mix data, model, and inference logic

Correct placement is mandatory.

---

### 3. No Hardcoding

* All paths must come from:

  * `config.yaml`
* No absolute paths
* No inline constants for configs

---

### 4. GPU Awareness

All generated code must:

* Be optimized for **low VRAM (4GB)**
* Use:

  * 4-bit quantization
  * small batch sizes
  * gradient accumulation

---

## 📁 File-Specific Rules

### 🔹 data/scripts/

* Only scraping and preprocessing logic
* No model-related code
* Must output structured JSON/JSONL

---

### 🔹 model/training/

* Training logic only
* Must support:

  * LoRA
  * checkpoint saving
* No API or UI logic

---

### 🔹 inference/engine/

* Model loading and generation
* Must be optimized for fast inference
* Should handle:

  * tokenization
  * prompt formatting

---

### 🔹 inference/api/

* FastAPI routes only
* No heavy logic inside routes
* Call engine functions

---

### 🔹 extension/

* JavaScript/TypeScript only
* No Python code here
* Must communicate via HTTP API

---

## 🧾 Coding Standards

### Python

* Use `snake_case`
* Type hints required
* Modular functions

### JavaScript (Extension)

* Use async/await
* Clean API calls
* Minimal logic

---

## 🔧 Model Handling Rules

* Always load models using:

  * quantization (4-bit or 8-bit)
* Never attempt full fine-tuning
* Use LoRA for all training

---

## 📊 Dataset Rules

* Input must be structured:

```json
{
  "input": "...",
  "output": "..."
}
```

* No raw text training
* No mixed formats

---

## 🚫 Forbidden Actions

AI must NOT:

* Train models from scratch
* Use large models (>7B)
* Ignore config system
* Merge unrelated logic into one file
* Generate unoptimized GPU code

---

## ✅ Expected Behaviors

AI SHOULD:

* Suggest modular improvements
* Optimize memory usage
* Keep functions small and reusable
* Follow project architecture strictly

---

## 🧪 Testing Expectations

Generated code must:

* Be runnable
* Be minimal
* Avoid unnecessary dependencies

---

## 🔄 Workflow Awareness

AI must understand pipeline:

```
Data → Processing → Training → Inference → Extension
```

No step should overlap improperly.

---

## 📌 Prompting Guidelines (for AI usage)

When using Copilot or similar tools:

### Good Prompt:

> "Write a LoRA training script using HuggingFace for Phi-2 with 4-bit quantization and config.yaml support"

### Bad Prompt:

> "Train a big AI model"

---

## 🧭 Long-Term Alignment

All generated code must align with:

* Local-first AI
* Python specialization
* Low-resource efficiency

---

## ⚠️ Final Rule

If uncertain:

> Default to simplicity, modularity, and low resource usage.

---

**This document is authoritative. All AI-generated code must comply.**
