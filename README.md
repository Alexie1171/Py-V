# PY-V (Python Virtual Assistant)

## Overview

PY-V is a lightweight, locally running AI code assistant designed for Python development and expanding into multi-language support. It replicates core features of tools like GitHub Copilot — code completion, function generation, debugging, and chat-based assistance — while being fully optimized for low-resource environments (GTX 1650, 4GB VRAM).

The system is built around a complete local ML pipeline: a custom dataset scraped from GitHub and StackOverflow, LoRA fine-tuning on Phi-2, a FastAPI inference server, a VS Code extension, and a context-aware chat system.

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
| 8 | RAG (Retrieval Augmented Generation) | Planned |
| 9 | Multi-LoRA adapters (multi-language support) | Planned |

---

## System Evolution Goal

PY-V is evolving from:

Code generator → Code assistant → Context-aware coding system → Chat-based coding agent

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
+-- retrieval/                  (Phase 8)
|   +-- indexer.py
|   +-- retriever.py
|
+-- extension/
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
| POST | `/api/v1/generate` | Code generation |
| POST | `/api/v1/chat` | Context-aware chat (Phase 7) |

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

Test the chat system:

```bash
python test_chat.py
```

---

## VS Code Extension

The extension communicates with the local inference server to provide code generation inside VS Code.

### Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| PY-V: Generate Code from Selection | Ctrl+Shift+G | Generate from selected text or comment |
| PY-V: Generate Code from Prompt | Ctrl+Shift+P | Generate from typed instruction |
| PY-V: Check Server Status | Status bar click | Ping the inference server |

### Setup

```bash
# Start the inference server first
uvicorn inference.api.main:app --host 0.0.0.0 --port 8000

# Build the extension
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

# Fine-tuned model output
python -m experiments.test_phi2

# Chat system
python test_chat.py

# API server
uvicorn inference.api.main:app --host 0.0.0.0 --port 8000
```

---

## Future Improvements

- RAG-based codebase understanding (Phase 8)
- Multi-language support via LoRA adapters (Phase 9)
- AST-aware context tracking
- Smarter controller routing
- Persistent chat memory optimization

---

## Author

Alexie1171 — PY-V Local AI Coding Assistant Project