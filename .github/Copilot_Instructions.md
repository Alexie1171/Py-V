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
| Phase 8 | RAG (Retrieval Augmented Generation) | Complete |
| Phase 9 | Multi-LoRA adapters (multi-language support) | Planned |
| Phase 10 | VS Code chat panel (full UI, no terminal) | Planned |

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
- Model name, paths, training hyperparameters, RAG settings all live here
- Never duplicate values from here into code

---

### `model/training/config_loader.py`
- Parses `config.yaml` into typed dataclasses (`ModelConfig`, `TrainingConfig`, `PathsConfig`, `RAGConfig`)
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
- Exposes: `build_prompt()`, `build_training_prompt()`, `build_inference_prompt()`, `format_context()`, `format_retrieved_context()`
- For `explain` and `chat` modes, context history is NOT injected to prevent code pattern bias
- RAG context is only injected for `generate`, `debug`, `refactor` modes

---

### `inference/engine/prompt_templates.py`
- Stores all mode-specific prompt templates
- Templates use `### Instruction:` / `### Answer:` format throughout
- Modes: `generate`, `debug`, `explain`, `refactor`, `chat`
- `generate`, `debug`, `refactor` templates have `{retrieved_context}` slot
- `explain` and `chat` templates do NOT have `{retrieved_context}` slot
- Never define templates outside this file

---

### `inference/engine/generator.py`
- Handles raw model generation via `_run_generation()`
- `generate_from_prompt()` — main entry point used by `chat.py`
- `remove_code_if_not_allowed()` — strips code from explain/chat outputs
- `_strip_artifacts()` — removes training data artifacts (Exercise:, Task:, [...])
- Stop-word cleanup uses earliest-match strategy via `_apply_stop_words()`
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
- Wires together: controller → retriever → prompt builder → generator → context manager
- Single public method: `chat(session_id, user_input)`
- Instantiates `Retriever` at startup if RAG is enabled in config
- Gracefully disables RAG if index is missing

---

### `retrieval/indexer.py`
- Builds the FAISS index from dataset JSONL + codebase Python files
- Uses `chunker.py` for AST function/class-level splitting of codebase files
- Scans: `inference/`, `model/`, `data/scripts/`, `retrieval/`
- Skips: `__pycache__`, `sessions`, `experiments`, `extension`, `.git`, `venv`
- Stores `content` in both top-level and `metadata["content"]` for retriever compatibility
- Must be re-run after any codebase changes or dataset updates

---

### `retrieval/chunker.py`
- AST-based function and class extraction from Python source files
- Returns list of dicts with `content` and `metadata` (file, name, type, lines)
- Used exclusively by `indexer.py`

---

### `retrieval/embedder.py`
- Sentence embedding wrapper around `BAAI/bge-small-en-v1.5`
- Returns normalized numpy arrays for cosine similarity with FAISS IndexFlatIP

---

### `retrieval/vector_store.py`
- FAISS IndexFlatIP wrapper
- Handles add, search, save, load
- Always normalizes embeddings before add for correct cosine similarity

---

### `retrieval/retriever.py`
- Full retrieval pipeline: query expansion → embedding → FAISS search → intent filter → rerank
- `_get_content()` — unified content extraction from both top-level and metadata fields
- `detect_intent()` — routes query to sorting / searching / ml / web / general
- `expand_query()` — adds algorithm-specific terms to improve recall
- `_rerank()` — combines FAISS score + keyword overlap + intent boosts/penalties
- Never import model or training code

---

### `data/scripts/`
- Scraping only (GitHub, StackOverflow)
- Data cleaning & preprocessing
- Output must be structured JSON/JSONL
- No model logic allowed

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
- Endpoints: `GET /health`, `POST /generate`, `POST /chat`

---

### `extension/`
- VS Code extension only
- TypeScript only — no Python logic
- `src/extension.ts` — command registration, status bar
- `src/api.ts` — HTTP client for the FastAPI server
- `src/provider.ts` — editor insertion and instruction extraction
- `src/panel.ts` — (Phase 10) WebviewPanel lifecycle and VS Code ↔ webview bridge
- `src/chat_view.ts` — (Phase 10) chat UI logic inside the webview
- `media/chat.css` — (Phase 10) panel styling
- `media/chat.js` — (Phase 10) webview-side event handlers and VS Code API bridge
- Communicates with backend via `POST /api/v1/generate` and `POST /api/v1/chat`
- Handles ECONNREFUSED and timeout errors gracefully

---

## RAG Rules (Phase 8)

- RAG only fires for `generate`, `debug`, `refactor` modes
- RAG is never injected for `explain` or `chat` modes
- `top_k` and `index_path` come from `CFG.rag.*` — never hardcoded
- If the index does not exist, the chat engine starts without RAG — no crash
- Index must be rebuilt after dataset updates: `python -m retrieval.indexer`
- Content field must exist at both `chunk["content"]` and `chunk["metadata"]["content"]`
- Reranker uses intent detection — sorting queries penalise ML/dataset chunks heavily

---

## Phase 9 Rules — Multi-LoRA Adapters (Planned)

Phase 9 adds per-language LoRA adapters. All rules below apply when implementing Phase 9.

- One LoRA adapter per language, saved at `model/lora/{language}/`
- Base model (Phi-2, 4-bit) is shared — only adapter weights change between languages
- `model/adapters/adapter_registry.py` maps language identifiers to adapter paths
- `model/adapters/adapter_router.py` selects and loads the correct adapter at runtime
- `inference/engine/language_detector.py` detects language from file extension or VS Code `languageId`
- `configs/adapters.yaml` holds per-language adapter config — never hardcode adapter paths
- Adapter switching must not reload the base model — only swap the PEFT adapter
- Training data for non-Python languages lives in `data/datasets/{language}/`
- JavaScript/TypeScript scraper lives at `data/scripts/js_scraper.py`
- Python adapter remains the primary adapter — all existing behavior unchanged

---

## Phase 10 Rules — VS Code Chat Panel (Planned)

Phase 10 replaces terminal interaction with a Copilot-style chat panel inside VS Code. All rules below apply when implementing Phase 10.

- The chat panel is a VS Code `WebviewPanel` registered as a sidebar view
- `extension/src/panel.ts` owns the WebviewPanel lifecycle — creation, disposal, message passing
- `extension/src/chat_view.ts` owns the chat UI logic — rendering messages, handling input, scrolling
- `extension/media/chat.css` owns all panel styling — no inline styles in TypeScript or HTML
- `extension/media/chat.js` owns webview-side event handling and the VS Code API bridge
- The panel communicates with the FastAPI server via `POST /api/v1/chat` — same endpoint as terminal chat
- Session ID is generated once per panel instance and reused for the conversation lifetime
- Mode badge and `rag_chunks` count must be displayed on each response
- Active file context (language, file name, selected text) must be automatically injected into generate/debug prompts
- A streaming endpoint `POST /api/v1/chat/stream` (SSE) is required for progressive token display
- The streaming endpoint lives in `inference/api/routes.py` — no new files for routes
- Copy-to-editor button must be present on all code responses
- Clear session button must reset both the panel UI and the server-side session file
- The existing `pyv.generate` and `pyv.generateFromInput` commands remain unchanged
- The panel is activated by a new command: `pyv.openChat`
- No Python logic in any extension file — all backend calls go through `api.ts`

---

## Data Pipeline Rules

Pipeline flow:

```
Scraping → Cleaning → Deduplication → Formatting → Dataset → RAG Index
```

- `github_scraper.py` — AST-based function extraction, quality scoring
- `stackoverflow_scraper.py` — accepted answer extraction, Python filtering
- `cleaner.py` — AST validation, length bounds, noise removal
- `dedupe.py` — exact hash dedup + Jaccard near-dedup (threshold 0.85)
- `formatter.py` — instruction/output format, 90/10 train/val split
- `pipeline.py` — orchestrates all stages with checkpoint support
- After pipeline runs, always rebuild RAG index: `python -m retrieval.indexer`

---

## Dataset Format (STRICT)

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

---

## Training Rules

- Always use LoRA fine-tuning (`peft.LoraConfig`)
- Always use 4-bit quantization (`BitsAndBytesConfig`)
- Always assume low VRAM environment
- Never attempt full fine-tuning
- Always check for existing checkpoints before starting (`resolve_checkpoint()`)
- Training hyperparameters come from `CFG.training.*`
- Second epoch learning rate must be reduced (5e-5 or lower) — never reuse first epoch LR
- Back up `model/lora/` before any training run

---

## API Rules

- FastAPI app entry point: `inference/api/main.py`
- Run with: `uvicorn inference.api.main:app --host 0.0.0.0 --port 8000`
- Routes: `GET /api/v1/health`, `POST /api/v1/generate`, `POST /api/v1/chat`
- Phase 10 adds: `POST /api/v1/chat/stream` (SSE streaming)
- Model loads once at startup via lifespan — never per request
- LoRA adapter applied on top of base model via `load_lora_model()` before serving

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
- Inject RAG context into explain or chat mode prompts
- Reload the base model when switching LoRA adapters (Phase 9)
- Add Python logic to any extension TypeScript file (Phase 10)

---

## Expected Behaviors

AI SHOULD:
- Always import `CFG` for any path or config value
- Use `prompt_builder` for any prompt construction
- Keep functions small and single-purpose
- Optimize for memory usage on GTX 1650
- Respect architecture boundaries strictly
- Resume training from checkpoint when available
- Rebuild RAG index after any dataset or codebase change
- Store `content` at both top-level and `metadata["content"]` in all index chunks

---

## Smoke Tests

```bash
# Config wiring
python -c "from model.training.config_loader import CFG; print(CFG.model.name, CFG.paths.dataset)"

# Prompt builder
python -c "from inference.engine.prompt_builder import build_inference_prompt; print(build_inference_prompt('test'))"

# RAG index build
python -m retrieval.indexer

# RAG retrieval test
python -m retrieval.test_rag

# Fine-tuned model output
python -m experiments.test_phi2

# Chat system (terminal — Phase 7/8)
python test_chat.py

# API boot
uvicorn inference.api.main:app --host 0.0.0.0 --port 8000

# VS Code extension
cd extension && npm install && npm run compile
# Press F5 in VS Code to launch dev instance
```

---

## System Workflow

```
Data → Processing → Dataset → RAG Index
                            ↓
                        Training → LoRA Adapter
                                        ↓
                                Inference API
                                /           \
                    VS Code Extension     Chat Panel (Phase 10)
                    (generate commands)   (full chat UI)
```

---

## Final Rule

If uncertain: always choose modularity, simplicity, and low-resource efficiency.

---

## Authority

This document is mandatory. All AI-generated code must comply.