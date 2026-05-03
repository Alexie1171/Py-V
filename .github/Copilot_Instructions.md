# Copilot Instructions for PY-V

## 🧠 Purpose

This document defines strict behavioral and architectural rules for AI assistants contributing to the PY-V codebase.

It ensures:
- Consistency across modules
- Clean ML system design
- Reproducible pipeline behavior
- Low-resource optimization

---

## 📌 Current Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Project structure & architecture | ✅ Complete |
| Phase 2 | Phi-2 model setup & 4-bit inference | ✅ Complete |
| Phase 3 | Full data pipeline (scrape → clean → dedupe → format) | ✅ Complete |
| Phase 4 | LoRA fine-tuning on Python dataset | ✅ Complete |
| Phase 5 | FastAPI inference server | ✅ Complete |
| Phase 6 | VS Code extension | 🔄 In Progress |

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
- Create untracked utility scripts outside `/scripts` or `/data/scripts`

---

### 3. Configuration-Driven System
- All paths and model settings MUST come from `configs/config.yaml`
- No hardcoded paths anywhere in the codebase
- No inline constants for dataset/model locations
- Import config via: `from model.training.config_loader import CFG`

---

### 4. Low-Resource Optimization (Critical)
All generated code must be optimized for:
- GPU: GTX 1650 (4GB VRAM)
- 4-bit quantization (mandatory for inference and training)
- Small batch training (batch_size=1)
- Gradient accumulation instead of large batch sizes (accumulation=16)

---

## 🧠 Model Constraints

All work is based on:
- **Phi-2** (Microsoft, ~2.7B parameters)
- Fine-tuned LoRA adapter saved at `model/lora/`

Rules:
- No training from scratch
- No models >7B parameters
- Always use PEFT / LoRA fine-tuning
- Always load base model with 4-bit BitsAndBytes quantization
- Always resume from checkpoint when one exists

---

## 📁 File Responsibility Rules

### `configs/config.yaml`
- Single source of truth for ALL configuration
- Model name, paths, training hyperparameters all live here
- Never duplicate values from here into code

---

### `model/training/config_loader.py`
- Parses `config.yaml` into typed dataclasses (`ModelConfig`, `TrainingConfig`, `PathsConfig`)
- Exports a module-level `CFG` singleton
- All other modules import `CFG` from here — never re-parse yaml elsewhere

---

### `model/utils/model_loader.py`
- Single shared model loader for the entire project
- Reads model name from `CFG.model.name`
- Both `inference/engine/` and `model/training/` use this — no duplication

---

### `inference/engine/model_loader.py`
- Thin re-export wrapper: `from model.utils.model_loader import load_model`
- Keeps inference layer decoupled without duplicating logic

---

### `inference/engine/prompt_builder.py`
- Central prompt formatting for Phi-2
- Used by BOTH training (`dataset_loader.py`) and inference (`generator.py`)
- Phi-2 format: `Instruct: {instruction}\nOutput:\n{code}`
- Never define prompt format in any other file

---

### `data/scripts/`
- Scraping only (GitHub, StackOverflow)
- Data cleaning & preprocessing
- Output must be structured JSON/JSONL
- No model logic allowed

---

### `data/processed/`
- Cleaned datasets only
- Deduplicated outputs only

---

### `data/datasets/`
- Final training-ready dataset only
- Must be JSONL format
- Keys: `instruction`, `output`, optional `metadata`

---

### `model/training/`
- Training logic only
- LoRA fine-tuning scripts
- Checkpoint saving and resumption logic
- No inference or API code

---

### `inference/engine/`
- Model loading (via wrapper)
- Prompt formatting
- Generation logic
- Must be lightweight and GPU efficient

---

### `inference/api/`
- FastAPI routes only (`main.py`, `routes.py`, `schemas.py`)
- No heavy logic inside endpoints
- Must call inference engine only
- Model loaded once at startup via lifespan, stored in app state

---

### `extension/`
- VS Code extension only
- JavaScript/TypeScript only
- Communicates with backend via HTTP API (`/api/v1/generate`)
- No Python logic

---

## 🔄 Data Pipeline Rules

Pipeline flow:

```
Scraping → Cleaning → Deduplication → Formatting → Dataset
```

- `github_scraper.py` → AST-based function extraction, quality scoring
- `stackoverflow_scraper.py` → accepted answer extraction, Python filtering
- `cleaner.py` → AST validation, length bounds, noise removal
- `dedupe.py` → exact hash dedup + Jaccard near-dedup
- `formatter.py` → instruction/output format, train/val split
- `pipeline.py` → orchestrates all stages with checkpoint support

---

## 📊 Dataset Format (STRICT)

All training data must follow:

```json
{
  "instruction": "Write a Python function to check if a number is prime",
  "output": "def is_prime(n): ...",
  "metadata": {
    "source": "github",
    "code_score": 2.5
  }
}
```

Rules:
- No raw text datasets
- No mixed formats
- No unstructured dumps
- Metadata is optional but recommended

---

## ⚙️ Training Rules

- Always use LoRA fine-tuning (`peft.LoraConfig`)
- Always use 4-bit quantization (`BitsAndBytesConfig`)
- Always assume low VRAM environment
- Never attempt full fine-tuning
- Always check for existing checkpoints before starting (`resolve_checkpoint()`)
- Training hyperparameters come from `CFG.training.*`

---

## 🌐 API Rules

- FastAPI app entry point: `inference/api/main.py`
- Run with: `uvicorn inference.api.main:app --host 0.0.0.0 --port 8000`
- Routes: `GET /api/v1/health`, `POST /api/v1/generate`
- Model loads once at startup, never per-request
- The LoRA adapter must be applied on top of the base model before serving

---

## 🔌 VS Code Extension Rules (Phase 6)

- Lives entirely in `extension/`
- Written in TypeScript
- Calls `POST http://localhost:8000/api/v1/generate`
- Request body: `{ "instruction": "...", "max_tokens": 512, "temperature": 0.2 }`
- No bundled Python or ML logic
- Must handle server-not-running gracefully

---

## 🚫 Forbidden Actions

AI MUST NOT:
- Train models from scratch
- Use large models (>7B)
- Ignore the config system
- Mix pipeline stages in one file
- Hardcode file paths
- Duplicate model loading logic
- Define prompt format outside `prompt_builder.py`
- Add heavy logic inside FastAPI route handlers
- Write extension logic in Python

---

## ✅ Expected Behaviors

AI SHOULD:
- Always import `CFG` for any path or config value
- Use `prompt_builder` for any prompt construction
- Keep functions small and single-purpose
- Optimize for memory usage on GTX 1650
- Respect architecture boundaries strictly
- Resume training from checkpoint when available

---

## 🧪 Smoke Tests

```bash
# Config wiring
python -c "from model.training.config_loader import CFG; print(CFG.model.name, CFG.paths.dataset)"

# Prompt builder
python -c "from inference.engine.prompt_builder import build_inference_prompt; print(build_inference_prompt('test'))"

# API boot
uvicorn inference.api.main:app --host 0.0.0.0 --port 8000

# Fine-tuned model output
python -m experiments.test_phi2
```

---

## 🧭 System Workflow

```
Data → Processing → Dataset → Training → Inference API → VS Code Extension
```

No step should be skipped or merged incorrectly.

---

## ⚠️ Final Rule

If uncertain:
Always choose modularity, simplicity, and low-resource efficiency.

---

## 📌 Authority

This document is mandatory. All AI-generated code must comply.