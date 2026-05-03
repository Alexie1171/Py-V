# PY-V (Python Virtual Assistant)

## 🧠 Overview

**PY-V** is a lightweight, locally running AI code assistant designed specifically for Python development. It replicates core features of tools like GitHub Copilot — code completion, function generation, and debugging assistance — while being fully optimized for low-resource environments (GTX 1650, 4GB VRAM).

The project is built around a complete local ML pipeline: a custom Python dataset scraped from GitHub and StackOverflow, LoRA fine-tuning on Phi-2, a FastAPI inference server, and a VS Code extension for real-time suggestions.

---

## 🚀 Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project structure & architecture | ✅ Complete |
| 2 | Phi-2 model setup, 4-bit quantization, modular engine | ✅ Complete |
| 3 | Full data pipeline (scrape → clean → dedupe → format) | ✅ Complete |
| 4 | LoRA fine-tuning on Python dataset | ✅ Complete |
| 5 | FastAPI inference server | ✅ Complete |
| 6 | VS Code extension | 🔄 In Progress |

---

## 🏗️ Project Architecture

```
PY-V/
│
├── configs/
│   └── config.yaml                  # Single source of truth for all config
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
│       ├── github_scraper.py
│       ├── stackoverflow_scraper.py
│       ├── cleaner.py
│       ├── dedupe.py
│       ├── formatter.py
│       └── pipeline.py
│
├── model/
│   ├── base/                        # Downloaded Phi-2 weights (gitignored)
│   ├── lora/                        # Fine-tuned LoRA adapter (gitignored)
│   ├── training/
│   │   ├── config_loader.py         # Parses config.yaml → typed CFG singleton
│   │   ├── dataset_loader.py        # Loads JSONL, applies prompt format
│   │   └── train_lora.py            # LoRA fine-tuning script
│   └── utils/
│       └── model_loader.py          # Shared base model loader (4-bit quant)
│
├── inference/
│   ├── engine/
│   │   ├── model_loader.py          # Thin wrapper → model/utils/model_loader
│   │   ├── prompt_builder.py        # Phi-2 prompt format (shared by train+infer)
│   │   └── generator.py             # Generation logic
│   └── api/
│       ├── main.py                  # FastAPI app, lifespan model loading
│       ├── routes.py                # /health, /generate endpoints
│       └── schemas.py               # Pydantic request/response types
│
├── extension/                       # VS Code extension (TypeScript)
│   ├── src/
│   ├── package.json
│   └── README.md
│
├── experiments/
│   └── test_phi2.py                 # Fine-tuned model output testing
│
├── requirements.txt
└── README.md
```

---

## 🔄 Data Pipeline

The full pipeline runs via a single command:

```bash
python -m data.scripts.pipeline
```

Stages:

1. **GitHub Scraping** — AST-based function extraction from high-star Python repos, quality scoring per function
2. **StackOverflow Scraping** — accepted answer extraction, Python code filtering, multi-block support
3. **Cleaning** — AST validation, length bounds, encoding fixes, noise pattern removal
4. **Deduplication** — exact hash dedup + Jaccard shingling near-dedup (threshold: 0.85)
5. **Formatting** — instruction/output JSONL, quality sort, 90/10 train/val split

Output: `data/datasets/train.jsonl` + `data/datasets/val.jsonl`

---

## ⚙️ Configuration

All settings live in `configs/config.yaml`:

```yaml
model:
  name: "microsoft/phi-2"
  max_tokens: 512

training:
  batch_size: 1
  gradient_accumulation: 16
  epochs: 3
  learning_rate: 0.0002
  lora_r: 8
  lora_alpha: 32
  lora_dropout: 0.05
  max_seq_length: 384

paths:
  dataset: "./data/datasets/train.jsonl"
  val_dataset: "./data/datasets/val.jsonl"
  model_output: "./model/lora"
```

Import anywhere with:
```python
from model.training.config_loader import CFG
print(CFG.model.name)      # microsoft/phi-2
print(CFG.paths.dataset)   # ./data/datasets/train.jsonl
```

---

## 🧠 Model

- **Base model**: [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) (~2.7B parameters)
- **Quantization**: 4-bit NF4 via BitsAndBytes (fits in 4GB VRAM)
- **Fine-tuning**: LoRA (r=8, alpha=32) via PEFT
- **Training result**: Loss 1.087 → 0.872 over 115 steps (~6.7 hours on GTX 1650)
- **Prompt format**:
  ```
  Instruct: Write a Python function to check if a number is prime.
  Output:
  def is_prime(n):
      ...
  ```

---

## 🌐 Inference API

Start the server:

```bash
uvicorn inference.api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/v1/health` | Liveness check |
| POST | `/api/v1/generate` | Generate Python code |

Example request:

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Write a function to check if a string is a palindrome", "max_tokens": 256, "temperature": 0.2}'
```

---

## 🔌 VS Code Extension (Phase 6)

The extension lives in `extension/` and is written in TypeScript. It communicates with the local inference API to provide real-time code suggestions inside VS Code.

To develop:

```bash
cd extension
npm install
npm run compile
```

---

## 💻 Installation

```bash
pip install -r requirements.txt
```

Set your API keys in `.env`:

```env
GITHUB_TOKEN=your_token_here
HF_HOME=C:\Users\YourName\.cache\huggingface
```

---

## 🧪 Smoke Tests

```bash
# Verify config loads
python -c "from model.training.config_loader import CFG; print(CFG.model.name)"

# Verify prompt builder
python -c "from inference.engine.prompt_builder import build_inference_prompt; print(build_inference_prompt('test'))"

# Test fine-tuned model
python -m experiments.test_phi2

# Boot the API
uvicorn inference.api.main:app --host 0.0.0.0 --port 8000
```

---

## ⚠️ Hardware Constraints

- GPU: GTX 1650 (4GB VRAM)
- 4-bit quantization required for both training and inference
- batch_size=1 with gradient_accumulation=16
- Expect ~190s/step during training

---

## 🧭 Future Improvements

- Additional training epochs for lower loss
- Larger dataset (more repos, more SO tags)
- RAG (Retrieval Augmented Generation) for codebase-aware suggestions
- AST-aware context window
- Multi-language support

---

## 👨‍💻 Author

**Alexie1171**
Project: PY-V
Purpose: Experimental Local AI Coding Assistant