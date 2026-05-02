# PY-V (Python Virtual Assistant)

## 🧠 Overview

**PY-V** is a lightweight, locally running AI code assistant designed specifically for Python development. It aims to replicate core features of tools like Copilot—such as code completion, snippet generation, and debugging assistance—while being optimized for low-resource environments (e.g., GTX 1650 4GB GPU).

This project is built around a fully local machine learning pipeline using a small language model (Phi-2) and parameter-efficient fine-tuning (PEFT).

---

## 🎯 Objectives

- Build a Python-focused AI assistant
- Run inference locally with minimal hardware
- Implement LoRA-based fine-tuning
- Create a VS Code extension for real-time suggestions
- Maintain clean, scalable ML architecture
- Build full dataset pipeline from real-world sources

---

## 🏗️ Project Architecture

PY-V/
│
├── data/
│ ├── raw/
│ │ ├── github/
│ │ └── stackoverflow/
│ │
│ ├── processed/
│ │ ├── cleaned/
│ │ └── deduped/
│ │
│ ├── datasets/
│ │ ├── train.jsonl
│ │ └── val.jsonl
│ │
│ └── scripts/
│ ├── github_scraper.py
│ ├── stackoverflow_scraper.py
│ ├── cleaner.py
│ ├── dedupe.py
│ ├── formatter.py
│ └── pipeline.py
│
├── model/
│ ├── base/
│ ├── lora/
│ ├── configs/
│ ├── training/
│ │ ├── config_loader.py
│ │ ├── dataset_loader.py
│ │ └── train_lora.py
│ │
│ └── utils/
│ └── model_loader.py
│
├── inference/
│ ├── api/
│ │ ├── main.py
│ │ ├── routes.py
│ │ └── schemas.py
│ │
│ ├── engine/
│ │ ├── model_loader.py
│ │ ├── generator.py
│ │ └── prompt_builder.py
│ │
│ └── utils/
│
├── extension/
│ ├── src/
│ ├── package.json
│ └── README.md
│
├── experiments/
│ ├── logs/
│ ├── outputs/
│ └── notebooks/
│
├── configs/
│ └── config.yaml
│
├── scripts/
├── requirements.txt
├── README.md
└── Copilot_Instructions.md

---

## 🔄 Full Data Pipeline (PHASE 3 CORE)

1. Data Collection
   - GitHub repository scraping
   - StackOverflow Q/A extraction
   - Stored in data/raw/

2. Data Processing
   - Cleaning invalid/noisy code
   - Normalizing formatting
   - Output → data/processed/cleaned/

3. Deduplication
   - Remove duplicate samples
   - Output → data/processed/deduped/

4. Dataset Formatting
   - Convert to instruction format JSONL
   - Output → data/datasets/train.jsonl

5. Pipeline Automation
   - Single command execution via pipeline.py

---

## ⚙️ Configuration

configs/config.yaml

Example:

model:
  name: "phi-2"
  max_tokens: 512

training:
  batch_size: 1
  gradient_accumulation: 16
  epochs: 3

paths:
  dataset: "./data/datasets/train.jsonl"
  model_output: "./model/lora"

---

## 💻 Requirements

pip install -r requirements.txt

---

## 🚀 Development Phases

Phase 1: Structure ✔
- Project architecture
- Config system

Phase 2: Model Setup ✔
- Phi-2 inference working
- 4-bit quantization
- Modular engine

Phase 3: Data Pipeline 🔄 (CURRENT)
- GitHub scraping
- StackOverflow scraping
- Cleaning + deduplication
- JSONL dataset generation

Phase 4: Fine-Tuning
- LoRA training
- PEFT optimization
- Python specialization

Phase 5: Backend
- FastAPI inference server
- Model serving layer

Phase 6: VS Code Extension
- Real-time code suggestions
- Copilot-like experience

---

## ⚠️ Constraints

- GPU: GTX 1650 (4GB VRAM)
- Requires 4-bit quantization
- Small batch training only
- Efficiency over scale

---

## 🧪 Expected Capabilities

- Python code completion
- Function generation
- Debugging suggestions
- Offline AI assistant

---

## 🚧 Limitations

- No deep multi-file reasoning
- Limited context window
- Dependent on dataset quality

---

## 🧭 Future Improvements

- Retrieval Augmented Generation (RAG)
- AST-aware training
- Reinforcement learning
- Multi-language support

---

## 📌 Notes

- Dataset quality > model size
- Keep training and inference separated
- Avoid hardcoded paths
- Modular design is mandatory
- Pipeline is the core intelligence layer

---

## 👨‍💻 Author

Alexie1171  
Project: PY-V  
Purpose: Experimental Local AI Coding System