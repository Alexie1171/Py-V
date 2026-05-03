# PY-V (Python Virtual Assistant)

## 🧠 Overview

**PY-V** is a lightweight, locally running AI code assistant designed for Python development and expanding into multi-language support. It replicates core features of tools like GitHub Copilot — code completion, function generation, debugging, and chat-based assistance — while being fully optimized for low-resource environments (GTX 1650, 4GB VRAM).

The system is built around a complete local ML pipeline: a custom dataset scraped from GitHub and StackOverflow, LoRA fine-tuning on Phi-2, a FastAPI inference server, and a VS Code extension.

---

## 🚀 Development Phases

| Phase | Description | Status |
|------|-------------|--------|
| 1 | Project structure & architecture | ✅ Complete |
| 2 | Phi-2 model setup, 4-bit quantization, modular engine | ✅ Complete |
| 3 | Full data pipeline (scrape → clean → dedupe → format) | ✅ Complete |
| 4 | LoRA fine-tuning on Python dataset | ✅ Complete |
| 5 | FastAPI inference server | ✅ Complete |
| 6 | VS Code extension | ✅ Complete |
| 7 | Chat system (context-aware assistant + controller) | 🔜 In Progress |
| 8 | RAG (Retrieval Augmented Generation) | 🔜 Planned |
| 9 | Multi-LoRA adapters (multi-language support) | 🔜 Planned |

---

## 🧭 System Evolution Goal

PY-V is evolving from:

> Code generator → Code assistant → Context-aware coding system → Chat-based coding agent

---

## 🏗️ Project Architecture

```
PY-V/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   ├── github/
│   │   └── stackoverflow/
│   ├── processed/
│   │   ├── cleaned/
│   │   └── deduped/
│   ├── datasets/
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   └── scripts/
│
├── model/
│   ├── base/
│   ├── lora/
│   ├── training/
│   └── utils/
│
├── inference/
│   ├── engine/
│   │   ├── model_loader.py
│   │   ├── prompt_builder.py
│   │   ├── generator.py
│   │   ├── controller.py             # NEW (Phase 7)
│   │   ├── context_manager.py        # NEW (Phase 7)
│   │   └── chat.py                   # NEW (Phase 7)
│   └── api/
│       ├── main.py
│       ├── routes.py
│       └── schemas.py
│
├── retrieval/                        # Phase 8 (RAG)
│   ├── indexer.py
│   └── retriever.py
│
├── extension/
│
├── experiments/
└── README.md
```

---

## 🔄 Data Pipeline

```bash
python -m data.scripts.pipeline
```

Stages:

- GitHub scraping (AST-based extraction)
- StackOverflow scraping (accepted answers)
- Cleaning (AST validation, noise removal)
- Deduplication (hash + Jaccard similarity)
- Formatting (JSONL dataset creation)

Output:

- `train.jsonl`
- `val.jsonl`

---

## ⚙️ Configuration

All configuration is centralized in `configs/config.yaml`.

---

## 🧠 Model

- **Base model**: Microsoft Phi-2 (~2.7B parameters)
- **Quantization**: 4-bit NF4 (BitsAndBytes)
- **Fine-tuning**: LoRA (r=8, alpha=32)
- **Training optimized for GTX 1650 (4GB VRAM)**

Prompt format:

```
Instruct: ...
Output:
...
```

---

## 🌐 Inference API

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

## 💬 Chat System (Phase 7)

PY-V is transitioning into a chat-based coding assistant.

Features:

- Session-based context memory (context.json)
- Mode detection (debug / explain / generate / refactor)
- Language detection (Python initially)
- Editor context injection
- Controller-based routing
- Lightweight conversation history

---

## 🔌 VS Code Extension

Supports real-time interaction with the model.

### Commands:

- Generate code from comments
- Prompt-based generation
- Chat interface (Phase 7)



## 🧠 Context System

Each session maintains:

```json
{
  "session_id": "abc123",
  "language": "python",
  "mode": "debug",
  "current_task": "...",
  "last_summary": "...",
  "recent_actions": []
}
```

## 🔍 Future Improvements

- RAG-based codebase understanding
- Multi-language support via LoRA adapters
- AST-aware context tracking
- Smarter controller routing
- Persistent chat memory optimization

---

## ⚠️ Hardware Constraints

- GPU: GTX 1650 (4GB VRAM)
- Batch size: 1
- Gradient accumulation: 16
- 4-bit quantization required
- Designed for low-resource inference

---

## 👨‍💻 Author

**Alexie1171**

PY-V — Local AI Coding Assistant Project