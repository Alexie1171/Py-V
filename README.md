# PY-V (Python Virtual Assistant)

## 🧠 Overview

**PY-V** is a lightweight, locally running AI code assistant designed specifically for Python development. It aims to replicate core features of tools like Copilot—such as code completion, snippet generation, and debugging assistance—while being optimized for low-resource environments (e.g., GTX 1650 4GB GPU).

This project focuses on:

* Efficient fine-tuning of small language models
* Domain-specific intelligence (Python-only)
* Full local control (no dependency on external APIs)
* Modular, production-grade architecture

---

## 🎯 Objectives

* Build a Python-focused AI assistant
* Run inference locally with minimal hardware
* Implement LoRA-based fine-tuning
* Create a VS Code extension for real-time suggestions
* Maintain clean, scalable architecture

---

## 🏗️ Project Architecture

```
PY-V/
│
├── data/
│   ├── raw/                # Unprocessed scraped data
│   ├── processed/          # Cleaned data
│   ├── datasets/           # Final training-ready JSONL
│   └── scripts/            # Scraping & preprocessing scripts
│
├── model/
│   ├── base/               # Base model (Phi-2)
│   ├── lora/               # Fine-tuned adapters
│   ├── configs/            # Training configs
│   ├── training/           # Training scripts
│   └── utils/              # Helper functions
│
├── inference/
│   ├── api/                # FastAPI backend
│   ├── engine/             # Model loading & generation
│   └── utils/              # Prompt formatting, tokenization
│
├── extension/              # VS Code extension
│   ├── src/
│   ├── package.json
│   └── README.md
│
├── experiments/
│   ├── logs/
│   ├── outputs/
│   └── notebooks/
│
├── configs/
│   └── config.yaml
│
├── scripts/                # Utility scripts
├── requirements.txt
├── README.md
└── Copilot_Instructions.md
```

---

## 🔄 Workflow Pipeline

1. **Data Collection**

   * Scrape GitHub, StackOverflow
   * Store raw data in `data/raw/`

2. **Data Processing**

   * Clean & normalize → `data/processed/`
   * Format into JSONL → `data/datasets/`

3. **Model Training**

   * Load base model (Phi-2)
   * Apply LoRA fine-tuning
   * Save adapters → `model/lora/`

4. **Inference Engine**

   * Load model + LoRA
   * Serve via FastAPI

5. **VS Code Extension**

   * Send code context → API
   * Display suggestions inline

---

## ⚙️ Configuration

All paths and parameters are managed in:

```
configs/config.yaml
```

Example:

```yaml
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
```

---

## 💻 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Development Phases

### Phase 1: Structure ✅

* Project layout
* Config system

### Phase 2: Model Setup

* Download base model (Phi-2)
* Test inference

### Phase 3: Data Pipeline

* Scraping scripts
* Cleaning & formatting

### Phase 4: Fine-Tuning

* LoRA training
* Evaluation

### Phase 5: Backend

* FastAPI server
* Inference pipeline

### Phase 6: VS Code Extension

* Editor integration
* Real-time suggestions

---

## ⚠️ Constraints

* GPU: GTX 1650 (4GB VRAM)
* Requires quantization (4-bit)
* Small batch sizes only
* Focus on efficiency over scale

---

## 🧪 Expected Capabilities

* Python code completion
* Function generation
* Basic debugging suggestions

Limitations:

* No deep multi-file reasoning
* Limited context window

---

## 🧭 Future Improvements

* RAG (Retrieval Augmented Generation)
* AST-aware training
* Reinforcement tuning
* Multi-language support

---

## 📌 Notes

* Clean data is more important than large data
* Modular design is strictly enforced
* Avoid hardcoding paths
* Keep training and inference separate

---

## 👨‍💻 Author
Alexie1171
Project: PY-V
Purpose: Experimental AI System
