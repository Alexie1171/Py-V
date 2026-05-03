# Copilot Instructions for PY-V

## Purpose

This document defines strict behavioral and architectural rules for AI assistants contributing to the PY-V codebase.

It ensures:
- Consistency across modules
- Clean ML system design
- Reproducible pipeline behavior
- Low-resource optimization

---

## Current Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Project structure & architecture | Complete |
| Phase 2 | Phi-2 model setup & 4-bit inference | Complete |
| Phase 3 | Full data pipeline (scrape → clean → dedupe → format) | Complete |
| Phase 4 | LoRA fine-tuning on Python dataset | Complete |
| Phase 5 | FastAPI inference server | Complete |
| Phase 6 | VS Code extension | Complete |
| Phase 7 | Chat system (context-aware assistant + controller) | Complete |
| Phase 8 | RAG (Retrieval Augmented Generation) | Planned |
| Phase 9 | Multi-LoRA adapters (multi-language support) | Planned |

---

## Core Principles

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

## Model Constraints

All work is based on:
- Phi-2 (Microsoft, ~2.7B parameters)
- Fine-tuned LoRA adapter saved at `model/lora/`
- Training result: loss 1.087 → 0.872 over 115 steps (1 epoch, GTX 1650)

Rules:
- No training from scratch
- No models >7B parameters
- Always use PEFT / LoRA fine-tuning
- Always load base model with 4-bit BitsAndBytes quantization
- Always resume from checkpoint when one exists (`resolve_checkpoint()`)

---

## File Responsibility Rules

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
- Wraps `model/utils/model_loader.py`
- Also exposes `load_lora_model()` — loads base model then applies LoRA adapter
- This is what the API server calls at startup

---

### `inference/engine/prompt_builder.py`
- Central prompt formatting for Phi-2
- Used by BOTH training (`dataset_loader.py`) and inference (`generator.py`)
- Prompt format: `### Instruction:\n{instruction}\n\n### Answer:\n{output}`
- Never define prompt format in any other file
- Exposes: `build_prompt()`, `build_training_prompt()`, `build_inference_prompt()`, `format_context()`
- For `explain` and `chat` modes, context history is NOT injected to prevent code pattern bias

---

### `inference/engine/prompt_templates.py`
- Stores all mode-specific prompt templates
- Templates use `### Instruction:` / `### Answer:` format throughout
- Modes: `generate`, `debug`, `explain`, `refactor`, `chat`
- Never define templates outside this file

---

### `inference/engine/generator.py`
- Handles raw model generation via `_run_generation()`
- `generate_from_prompt()` — main entry point used by `chat.py`
- `remove_code_if_not_allowed()` — strips code from explain/chat outputs
- Stop-word cleanup uses earliest-match strategy via `apply_stop_words()`
- Retry logic uses temperature 0.5 on second attempt for chat/explain modes
- Generation settings: `repetition_penalty=1.1`, `no_repeat_ngram_size=4`

---

### `inference/engine/controller.py`
- Detects user intent and routes to correct mode
- Modes: `generate`, `debug`, `explain`, `refactor`, `chat`
- Keyword-weighted scoring with fallback to `chat` mode

---

### `inference/engine/context_manager.py`
- Loads and saves session context to `sessions/{session_id}.json`
- Appends conversation history per turn
- Updates session state (mode, entities, etc.)

---

### `inference/engine/context_schema.py`
- Defines `SessionContext` and `ChatTurn` dataclasses
- No logic — types only

---

### `inference/engine/chat.py`
- Top-level chat orchestrator
- Wires together: controller → prompt builder → generator → context manager
- Single public method: `chat(session_id, user_input)`

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

### `inference/api/`
- FastAPI routes only (`main.py`, `routes.py`, `schemas.py`)
- No heavy logic inside endpoints
- Must call inference engine only
- Model loaded once at startup via lifespan, stored in app state

---

### `extension/`
- VS Code extension only
- TypeScript only — no Python logic
- `src/extension.ts` — command registration, status bar
- `src/api.ts` — HTTP client for the FastAPI server
- `src/provider.ts` — editor insertion and instruction extraction
- Communicates with backend via `POST /api/v1/generate`
- Handles server-not-running gracefully with clear error messages

---

## Data Pipeline Rules

Pipeline flow:

```
Scraping → Cleaning → Deduplication → Formatting → Dataset
```

- `github_scraper.py` — AST-based function extraction, quality scoring
- `stackoverflow_scraper.py` — accepted answer extraction, Python filtering
- `cleaner.py` — AST validation, length bounds, noise removal
- `dedupe.py` — exact hash dedup + Jaccard near-dedup (threshold 0.85)
- `formatter.py` — instruction/output format, 90/10 train/val split
- `pipeline.py` — orchestrates all stages with checkpoint support

---

## Dataset Format (STRICT)

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

## Training Rules

- Always use LoRA fine-tuning (`peft.LoraConfig`)
- Always use 4-bit quantization (`BitsAndBytesConfig`)
- Always assume low VRAM environment
- Never attempt full fine-tuning
- Always check for existing checkpoints before starting (`resolve_checkpoint()`)
- Training hyperparameters come from `CFG.training.*`

---

## API Rules

- FastAPI app entry point: `inference/api/main.py`
- Run with: `uvicorn inference.api.main:app --host 0.0.0.0 --port 8000`
- Routes: `GET /api/v1/health`, `POST /api/v1/generate`, `POST /api/v1/chat`
- Model loads once at startup via lifespan — never per request
- LoRA adapter applied on top of base model via `load_lora_model()` before serving

---

## VS Code Extension Rules

- Lives entirely in `extension/`
- Written in TypeScript
- Three commands: `pyv.generate`, `pyv.generateFromInput`, `pyv.checkServer`
- Keybindings: `Ctrl+Shift+G` (generate), `Ctrl+Shift+P` (prompt input)
- Status bar item shows live server state: `PY-V`, `PY-V OK`, `PY-V ERR`
- Instruction extraction priority: selected text → `#` comment on current line
- Calls `POST http://localhost:8000/api/v1/generate`
- Must handle ECONNREFUSED and timeout errors gracefully

---

## Chat System Rules (Phase 7)

- Entry point: `inference/engine/chat.py` — `ChatEngine.chat()`
- Controller detects mode from user input keywords
- Context history is stored per session in `sessions/`
- History is injected into `generate`, `debug`, `refactor` prompts only
- History is NOT injected into `explain` or `chat` prompts to prevent code bias
- Retry on empty output uses temperature 0.5 (not 0.2) for chat/explain modes
- Code removal filter applies only to `chat` and `explain` mode outputs

---

## Forbidden Actions

AI MUST NOT:
- Train models from scratch
- Use large models (>7B)
- Ignore the config system
- Mix pipeline stages in one file
- Hardcode file paths
- Duplicate model loading logic
- Define prompt format outside `prompt_builder.py`
- Define prompt templates outside `prompt_templates.py`
- Add heavy logic inside FastAPI route handlers
- Write extension logic in Python
- Inject context history into explain or chat mode prompts

---

## Expected Behaviors

AI SHOULD:
- Always import `CFG` for any path or config value
- Use `prompt_builder` for any prompt construction
- Keep functions small and single-purpose
- Optimize for memory usage on GTX 1650
- Respect architecture boundaries strictly
- Resume training from checkpoint when available

---

## Smoke Tests

```bash
# Config wiring
python -c "from model.training.config_loader import CFG; print(CFG.model.name, CFG.paths.dataset)"

# Prompt builder
python -c "from inference.engine.prompt_builder import build_inference_prompt; print(build_inference_prompt('test'))"

# Fine-tuned model output
python -m experiments.test_phi2

# Chat system
python test_chat.py

# API boot (serves LoRA model)
uvicorn inference.api.main:app --host 0.0.0.0 --port 8000

# VS Code extension
cd extension && npm install && npm run compile
# Then press F5 in VS Code to launch dev instance
```

---

## System Workflow

```
Data → Processing → Dataset → Training → Inference API → VS Code Extension
                                                      → Chat System
```

No step should be skipped or merged incorrectly.

---

## Final Rule

If uncertain: always choose modularity, simplicity, and low-resource efficiency.

---

## Authority

This document is mandatory. All AI-generated code must comply.