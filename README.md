# PY-V (Python Virtual Assistant)

## Overview

PY-V is a lightweight, locally running AI code assistant designed for Python development and expanding into multi-language support. It replicates core features of tools like GitHub Copilot — code completion, function generation, debugging, and chat-based assistance — while being fully optimized for low-resource environments (GTX 1650, 4GB VRAM).

The system is built around a complete local ML pipeline: a custom dataset scraped from GitHub and StackOverflow, LoRA fine-tuning on Phi-2, a FastAPI inference server, a VS Code extension, a context-aware chat system, and a RAG pipeline backed by a FAISS vector store.

---

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project structure & architecture | Complete |
| 2 | Phi-2 model setup, 4-bit quantization, modular engine | Complete |
| 3 | Full data pipeline (scrape → clean → dedupe → format) | Complete |
| 4 | LoRA fine-tuning on Python dataset | Complete |
| 5 | FastAPI inference server | Complete |
| 6 | VS Code extension | Complete |
| 7 | Chat system (context-aware assistant + controller) | Complete |
| 8 | RAG (Retrieval Augmented Generation) | Complete |
| 9 | Multi-LoRA adapters (multi-language support) | Planned |
| 10 | VS Code chat panel (full UI, no terminal) | Planned |

---

## System Evolution Goal

PY-V is evolving from:

```
Code generator → Code assistant → Context-aware coding system → Chat-based coding agent → Full IDE integration
```

---

## Project Architecture

```
PY-V/
|
+-- configs/
|   +-- config.yaml
|
+-- data/
|   +-- raw/
|   |   +-- github/
|   |   +-- stackoverflow/
|   +-- processed/
|   |   +-- cleaned/
|   |   +-- deduped/
|   +-- datasets/
|   |   +-- train.jsonl
|   |   +-- val.jsonl
|   +-- scripts/
|
+-- model/
|   +-- base/
|   +-- lora/
|   +-- training/
|   +-- utils/
|
+-- inference/
|   +-- engine/
|   |   +-- model_loader.py
|   |   +-- prompt_builder.py
|   |   +-- prompt_templates.py
|   |   +-- generator.py
|   |   +-- controller.py
|   |   +-- context_manager.py
|   |   +-- context_schema.py
|   |   +-- chat.py
|   +-- api/
|       +-- main.py
|       +-- routes.py
|       +-- schemas.py
|
+-- retrieval/
|   +-- indexer.py
|   +-- retriever.py
|   +-- chunker.py
|   +-- embedder.py
|   +-- vector_store.py
|
+-- extension/
|   +-- src/
|   |   +-- extension.ts
|   |   +-- api.ts
|   |   +-- provider.ts
|   |   +-- panel.ts          (Phase 10)
|   |   +-- chat_view.ts      (Phase 10)
|   +-- media/
|   |   +-- chat.css          (Phase 10)
|   |   +-- chat.js           (Phase 10)
|   +-- package.json
|
+-- experiments/
+-- sessions/
+-- README.md
```

---

## Data Pipeline

```bash
python -m data.scripts.pipeline
```

Stages:
- GitHub scraping (AST-based function extraction)
- StackOverflow scraping (accepted answers only)
- Cleaning (AST validation, length bounds, noise removal)
- Deduplication (exact hash + Jaccard similarity at threshold 0.85)
- Formatting (JSONL output, 90/10 train/val split)

Output:
- `data/datasets/train.jsonl`
- `data/datasets/val.jsonl`

---

## Configuration

All configuration is centralized in `configs/config.yaml`. No paths or hyperparameters are hardcoded anywhere in the codebase. All modules import config via `from model.training.config_loader import CFG`.

---

## Model

- Base model: Microsoft Phi-2 (~2.7B parameters)
- Quantization: 4-bit NF4 (BitsAndBytes)
- Fine-tuning: LoRA (r=8, alpha=32)
- Training result: loss 1.087 → 0.872 over 115 steps (1 epoch, GTX 1650)
- Optimized for GTX 1650 (4GB VRAM)

Prompt format:

```
### Instruction:
{instruction}

### Answer:
{output}
```

---

## Inference API

```bash
uvicorn inference.api.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/generate` | Stateless code generation |
| POST | `/api/v1/chat` | Context-aware chat with RAG |

---

## RAG System (Phase 8)

PY-V uses a hybrid retrieval system backed by FAISS to inject relevant context into generate, debug, and refactor prompts.

### Components

| File | Role |
|------|------|
| `retrieval/indexer.py` | Builds the FAISS index from dataset + codebase |
| `retrieval/chunker.py` | AST-based function/class-level splitting |
| `retrieval/embedder.py` | Sentence embedding via `BAAI/bge-small-en-v1.5` |
| `retrieval/vector_store.py` | FAISS index wrapper with save/load |
| `retrieval/retriever.py` | Query expansion, intent detection, reranking |

### Build the index

```bash
python -m retrieval.indexer
```

### Index contents

- 1838 dataset chunks (instruction/output pairs)
- 146 codebase chunks (AST function/class level from inference/, model/, data/scripts/, retrieval/)
- Total: 1984 chunks

### RAG behavior

- Fires only for `generate`, `debug`, `refactor` modes
- Disabled for `explain` and `chat` to prevent code pattern bias
- Top-k chunks injected into prompt (k=3, configurable in `config.yaml`)
- Gracefully degrades if index is missing — chat engine starts without RAG

---

## Chat System (Phase 7)

PY-V includes a context-aware chat system that routes user input to the correct mode automatically.

Supported modes:
- `generate` — writes Python code from a description
- `debug` — identifies bugs and provides a corrected version
- `explain` — explains a concept in plain English, no code
- `refactor` — improves and cleans up existing code
- `chat` — general conversational assistant

Key behaviors:
- Session context is stored per session in `sessions/`
- Context history is injected for generate/debug/refactor modes
- Context history is NOT injected for explain/chat modes to prevent code pattern bias
- Code output is filtered out of explain and chat responses
- Retry logic uses higher temperature on second attempt if output is empty

```bash
python test_chat.py
```

---

## VS Code Extension (Phase 6)

The extension communicates with the local inference server to provide code generation inside VS Code.

### Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| PY-V: Generate Code from Selection | Ctrl+Shift+G | Generate from selected text or comment |
| PY-V: Generate Code from Prompt | Ctrl+Shift+P | Generate from typed instruction |
| PY-V: Check Server Status | Status bar click | Ping the inference server |

### Setup

```bash
uvicorn inference.api.main:app --host 0.0.0.0 --port 8000

cd extension
npm install
npm run compile
# Press F5 in VS Code to launch dev instance
```

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `pyv.serverUrl` | `http://localhost:8000` | Inference server URL |
| `pyv.maxTokens` | `256` | Max tokens to generate |
| `pyv.temperature` | `0.2` | Sampling temperature |

---

## Phase 9 — Multi-LoRA Adapters (Planned)

Phase 9 adds multi-language support by training separate LoRA adapters for additional languages and routing to the correct adapter at inference time based on the active file's language.

### Architecture

```
User query
    ↓
Language detector (file extension / VS Code languageId)
    ↓
Adapter router
    ↓
Load correct LoRA adapter (python / javascript / typescript / ...)
    ↓
Generate with language-specific adapter
```

### Planned components

| File | Role |
|------|------|
| `model/adapters/adapter_registry.py` | Maps language → LoRA adapter path |
| `model/adapters/adapter_router.py` | Selects and loads the correct adapter at runtime |
| `inference/engine/language_detector.py` | Detects language from context or file metadata |
| `data/scripts/js_scraper.py` | JavaScript/TypeScript dataset scraper |
| `configs/adapters.yaml` | Per-language adapter configuration |

### Training plan

- One LoRA adapter per language, each fine-tuned on a language-specific dataset
- Base model (Phi-2, 4-bit) shared across all adapters — only the adapter weights swap
- Adapter switching happens at inference time with no model reload
- Python adapter continues to be the primary, highest-quality adapter

---

## Phase 10 — VS Code Chat Panel (Planned)

Phase 10 replaces terminal-based interaction with a full Copilot-style chat panel inside VS Code. The user types in a sidebar panel, sees streamed responses, and can interact with the full chat system without leaving the editor.

### Architecture

```
VS Code Sidebar Panel (WebviewPanel)
    ↓
chat_view.ts — renders message history, handles input
    ↓
panel.ts — manages WebviewPanel lifecycle, message passing
    ↓
api.ts — POST /api/v1/chat (existing endpoint, unchanged)
    ↓
FastAPI → ChatEngine → RAG → Phi-2 → response
    ↓
Streamed back to panel via VS Code message passing
```

### Planned components

| File | Role |
|------|------|
| `extension/src/panel.ts` | WebviewPanel creation, lifecycle, VS Code ↔ webview bridge |
| `extension/src/chat_view.ts` | Chat UI logic — message rendering, input handling, scroll |
| `extension/media/chat.css` | Panel styling — message bubbles, mode badges, input bar |
| `extension/media/chat.js` | Webview-side JS — VS Code API bridge, event handlers |

### Planned features

- Persistent chat history visible in the panel across turns
- Mode badge on each response (generate / debug / explain / refactor / chat)
- RAG chunk count displayed per response
- Active file context automatically injected into generate/debug prompts
- Streaming display as tokens arrive (requires streaming API endpoint)
- Clear session button
- Copy-to-editor button on code responses

### New API endpoint needed

```
POST /api/v1/chat/stream
```

Streams tokens via Server-Sent Events (SSE) so the panel can display responses progressively rather than waiting for the full generation to complete.

---

## Context System

Each session maintains state in `sessions/{session_id}.json`:

```json
{
  "session_id": "abc123",
  "language": "python",
  "mode": "debug",
  "current_task": "...",
  "history": [],
  "entities": [],
  "errors_seen": [],
  "functions_touched": [],
  "last_summary": null
}
```

---

## Hardware Constraints

- GPU: GTX 1650 (4GB VRAM)
- Batch size: 1
- Gradient accumulation: 16
- 4-bit quantization required at all times
- Designed for low-resource inference

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

# Chat system (terminal)
python test_chat.py

# API server
uvicorn inference.api.main:app --host 0.0.0.0 --port 8000

# VS Code extension
cd extension && npm install && npm run compile
# Press F5 in VS Code to launch dev instance
```

---

## Author

Alexie1171 — PY-V Local AI Coding Assistant Project