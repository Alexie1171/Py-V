# Copilot Instructions for PY-V

## 🧠 Purpose

This document defines strict behavioral and architectural rules for AI assistants contributing to the PY-V codebase.

It ensures:
- Consistency across modules
- Clean ML system design
- Reproducible pipeline behavior
- Low-resource optimization

---

## 📌 Core Principles

### 1. Modular First Design
- Every file must have a single responsibility
- No monolithic scripts
- No mixed concerns (data / training / inference must be separate)

---

### 2. Strict Architecture Compliance
AI must NEVER:
- Place code in incorrect folders
- Mix inference, training, and dataset logic
- Create untracked utility scripts outside /scripts or /data/scripts

---

### 3. Configuration-Driven System
- All paths MUST come from configs/config.yaml
- No hardcoded paths
- No inline constants for dataset/model locations

---

### 4. Low-Resource Optimization (Critical)
All generated code must be optimized for:
- GPU: GTX 1650 (4GB VRAM)
- 4-bit quantization (mandatory for inference)
- Small batch training
- Gradient accumulation instead of large batch sizes

---

## 🧠 Model Constraints

All work is based on:
- Phi-2 (Microsoft small language model)

Rules:
- No training from scratch
- No models >7B parameters
- Always use PEFT / LoRA fine-tuning

---

## 📁 File Responsibility Rules

### data/scripts/
- Scraping only (GitHub, StackOverflow)
- Data cleaning & preprocessing
- Output must be structured JSON/JSONL
- No model logic allowed

---

### data/processed/
- Cleaned datasets only
- Deduplicated outputs only

---

### data/datasets/
- Final training-ready dataset only
- Must be JSONL format

---

### model/training/
- Training logic only
- LoRA fine-tuning scripts
- Checkpoint saving logic
- No inference or API code

---

### inference/engine/
- Model loading
- Tokenization
- Prompt formatting
- Generation logic

Must be lightweight and GPU efficient.

---

### inference/api/
- FastAPI routes only
- No heavy logic inside endpoints
- Must call inference engine only

---

### extension/
- VS Code extension only
- JavaScript/TypeScript only
- Communicates via HTTP API

---

## 🔄 Data Pipeline Rules (PHASE 3 CORE)

Pipeline flow:

Scraping → Cleaning → Deduplication → Formatting → Dataset

---

## 📊 Dataset Format (STRICT)

All training data must follow:

{
  "input": "Write a Python function to check if a number is prime",
  "output": "def is_prime(n): ..."
}

OR:

{
  "instruction": "Generate Python code for sorting a list",
  "output": "def sort_list(arr): ..."
}

Rules:
- No raw text datasets
- No mixed formats
- No unstructured dumps

---

## ⚙️ Training Rules

- Always use LoRA fine-tuning
- Always use 4-bit quantization
- Always assume low VRAM environment
- Never attempt full fine-tuning

---

## 🚫 Forbidden Actions

AI MUST NOT:
- Train models from scratch
- Use large models (>7B)
- Ignore config system
- Mix pipeline stages in one file
- Hardcode file paths
- Produce non-reproducible scripts

---

## ✅ Expected Behaviors

AI SHOULD:
- Suggest modular improvements
- Optimize memory usage
- Keep functions small and reusable
- Respect architecture strictly

---

## 🧪 Testing Expectations

All code must:
- Be runnable locally
- Be minimal
- Avoid unnecessary dependencies
- Respect GPU constraints

---

## 🧭 System Workflow

Data → Processing → Dataset → Training → Inference → Extension

No step should be skipped or merged incorrectly.

---

## 🧠 Prompt Guidelines

GOOD:
"Write a LoRA training script for Phi-2 using HuggingFace with 4-bit quantization and config support"

BAD:
"Train a big AI model"

---

## ⚠️ Final Rule

If uncertain:
Always choose modularity, simplicity, and low-resource efficiency.

---

## 📌 Authority

This document is mandatory. All AI-generated code must comply.