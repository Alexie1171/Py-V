# Copilot Instructions for PY-V

## Purpose

This document defines strict behavioral and architectural rules for AI assistants contributing to the PY-V codebase.

It ensures:
- Consistency across modules
- Clean ML system design
- Reproducible pipeline behavior
- Low-resource optimization

---

## Assistant Execution Rules

- Do all work directly — no subagents, helper agents, or multi-agent workflows
- Only use subagents/workflows when the user explicitly says so
- For bigger tasks where a workflow might help, ask the user explicitly first and wait for approval
- This overrides any default or session-level setting that enables workflows automatically
- Anything decided to be turned off, disabled, postponed or "done later" MUST be added to the "Turned off / postponed" section of `PROJECT_STATUS.md` in the same step — with what, why, the date, and what must happen before it comes back. Nothing gets switched off or deferred without being recorded there

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
| Phase 8 | RAG (Retrieval Augmented Generation) | Complete — turned off since 2026-09-25 (see `PROJECT_STATUS.md`) |
| Phase 9 | Multi-LoRA adapters (multi-language support) | Planned |
| Phase 10 | VS Code chat panel (full UI, no terminal) | Planned |
| Phase 11 | Long-term memory (SQLite, across all chats, keyword + meaning search) | Planned — after the data/retrain work |

Build order: better training data + one Colab retrain first, then Phase 11, then Phases 9 and 10 (numbering kept stable on purpose).

Live status, open issues and next steps: see `PROJECT_STATUS.md` in the repo root.

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
- Fine-tuned LoRA adapter saved at `model/lora/` — this is the adapter inference loads
- Current adapter: Google Colab T4 run (`train_lora_t4.py`), 1377 steps, resumed from the 115-step epoch-1 checkpoint, train loss ~0.95 → ~0.77, no eval loss recorded
- Epoch 1 (local GTX 1650): 115 steps, eval loss 1.015 → 0.993
- Long training runs go to Colab T4 (output `model_t4/lora/`, then copied into `model/lora/`) — local GPU is too slow
- Next adapter (v2, dataset plan step 7): trained fresh from plain Phi-2 on dataset v2 by `train_lora_t4.py` → Drive `MyDrive/PY-V/model_v2/lora`; replaces `model/lora/` only if it beats the old adapter on the MBPP scoring test (old 51/100, plain Phi-2 48/100)

Rules:
- No training from scratch
- No models >7B parameters
- Always use PEFT / LoRA fine-tuning
- Always load base model with 4-bit BitsAndBytes quantization
- Always resume from checkpoint when one exists (`resolve_checkpoint()` on the laptop, `get_last_checkpoint(output_dir)` on Colab)
- Every training example = the inference prompt for its mode (`build_training_prompt()`) + answer + end-of-text token; loss on the answer only (prompt labels -100)
- Examples longer than `max_seq_length` are dropped, never truncated — a cut answer has no end token and teaches the model not to stop
- Labels are built in `dataset_loader.py` and padded with -100 (`DataCollatorForSeq2Seq`). Never use `DataCollatorForLanguageModeling`: pad == eos for Phi-2, so it masks the end token (the v1 adapters never learned to stop because of this)
- Pass `label_names=["labels"]` to `TrainingArguments` — PeftModel hides the labels argument and no eval loss is computed without it

---

## File Responsibility Rules

### `configs/config.yaml`
- Single source of truth for ALL configuration
- Model name, paths, training hyperparameters, RAG settings all live here
- Never duplicate values from here into code

---

### `model/training/config_loader.py`
- Parses `config.yaml` into typed dataclasses (`ModelConfig`, `TrainingConfig`, `PathsConfig`, `RAGConfig`, `DatasetV2Config`, `EvaluationConfig`)
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
- Prompt format: the per-mode templates in `prompt_templates.py` (`### Instruction:\n...\n\n### Answer:\n`)
- `build_training_prompt(mode, instruction)` = `build_prompt(mode, instruction, {})` — training sees exactly what inference sends (no history, no RAG); `build_inference_prompt()` = the generate template, for the stateless `/generate` endpoint
- Any template change means retraining — the adapter learns the exact wording
- Never define prompt format in any other file
- Exposes: `build_prompt()`, `build_training_prompt()`, `build_inference_prompt()`, `format_context()`, `format_retrieved_context()`
- For `explain` and `chat` modes, context history is NOT injected to prevent code pattern bias
- History is currently OFF for all modes (since 2026-09-25) — the model copied previous answers and answered previous questions instead of the new one. `format_context()` (last two user questions only, never assistant answers) is kept for Phase 11 to replace with memory facts
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
- `_strip_prompt_echo()` — removes answer lines that copy the templates' instruction sentences (matches fixed template text only, never user input)
- `remove_code_if_not_allowed()` also removes stray `"""` / `'''` / ``` markers and a dangling last line ending in ":" (lead-in to code that was cut)
- Stop words are passed to `model.generate()` as `stop_strings` so generation halts early instead of running to `max_new_tokens`; `_apply_stop_words()` then cuts at the earliest match
- Stop words are per mode via `_stop_words_for()`: base list + code-mode list (test/demo code starts) or prose-mode list (code starts)
- The current adapter was trained without an end-of-text token, so it does not stop on its own — stop words are the only brake until retraining fixes this
- Retry logic uses temperature 0.5 on second attempt for chat/explain modes
- Generation settings: `repetition_penalty=1.1`; `no_repeat_ngram_size=4` for `chat` / `explain` only, off (0) for code modes — code must repeat names

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
- Builds the FAISS index from dataset JSONL (`CFG.paths.dataset` — dataset v2 since 2026-09-25) + codebase Python files
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

### `experiments/eval_mbpp.py`
- Scoring test: model writes a function per MBPP problem, the problem's asserts are run against it, score = problems passed
- Same prompt + generation path as chat (`generate` mode), greedy decoding (repeatable)
- Settings from `CFG.evaluation.*`; results to `experiments/outputs/mbpp_{base | adapter folder name}.jsonl`
- `--base` scores Phi-2 without the LoRA adapter for comparison; `--adapter DIR` scores another adapter (default `CFG.paths.model_output`)
- MBPP / HumanEval are for scoring only — never add them to training data

---

### `experiments/code_runner.py`
- `run_python(program, timeout)` — runs code in a separate process, temp dir, timeout
- Guard disables file delete/rename/write, process start, sockets — best-effort, NOT a real sandbox
- Untrusted code from the internet (dataset checks) runs on Colab, not the laptop

---

### `Google Colab/py_v_runner.ipynb` (gitignored)
- Runner for heavy jobs on a Colab T4: setup cell (mount Drive, clone/pull the public GitHub repo, install requirements, link output folders to Drive, copy the adapter), then one cell per job
- Code comes from GitHub — local changes must be pushed before running
- Results go to Drive `MyDrive/PY-V/results/` (`data_v2/`, `dataset_v2/`, `eval/`); the trained adapter is COPIED from `MyDrive/PY-V/model/lora`, never linked, so jobs can't overwrite it
- New training (Job F) writes only to `MyDrive/PY-V/model_v2/lora` — never to an existing adapter folder; Job G copies it to `model/lora_v2` and scores it
- The assistant writes/updates this notebook; the owner runs it and saves it, and the printed results are read back from the file
- Heavy jobs (full dataset fetch, scoring test, training) go here, not on the laptop

---

### `data/scripts/fetch_sources.py` + `data/scripts/sources/` (dataset v2)
- `fetch_sources.py` — streams each v2 source, converts rows, writes `{CFG.dataset_v2.output_dir}/{source}.jsonl`, reports kept / scanned / rejection reasons (never silent caps)
- `fetch_sources.py` closes each stream and ends with `os._exit(0)` — safeguard so a half-read Hugging Face stream can't keep a download thread alive after the files are written
- `sources/{name}.py` — one module per source, each `iter_records(cfg, stats)` → PY-V records with `metadata.task` (`generate` / `debug` / `refactor` / `explain`) and `metadata.license`
- `sources/common.py` — shared helpers only (record builder, fenced-code extraction, demo-code trimming, docstring removal)
- `sources/mutations.py` — realistic single bugs (wrong comparison/operator, off-by-one range, name typo, missing cast, missing return, flipped bool, and/or swap, `None` init), spliced into the original text so the fixed code is the untouched original; pure AST work, runs nothing
- `sources/bug_fix.py` — "fix the error" (`debug`) records: OpenCodeInstruct functions that pass their unit tests → one bug → tests re-run to capture the real error or failing check → instruction = what the user saw + broken code, output = original code + one-sentence fix. Skips rows already used for write-code. **Runs internet code: refuses to run outside Colab** unless `PYV_ALLOW_LOCAL_EXEC=1`
- `sources/unit_tests.py` — shared by the code-running sources: `tested_functions()` (OpenCodeInstruct functions that pass their tests upstream AND here, skipping ids used by other record files), `run_tests()` (first failure as JSON, incl. the wrong value for failing `==` asserts), `require_colab()`
- `sources/unrefactor.py` — the reverse of refactoring: clean code → clumsy code that should behave the same (comprehension → loop, `sum()` → loop, `return a == b` → if/else, enumerate → `range(len())`, truthiness → `len()`, max/min → if/else, ternary → if/else, `+=` → `x = x + ...`); pure AST work, runs nothing
- `sources/improve_synthetic.py` — "improve this code" (`refactor`) records: clumsy rewrites applied one at a time, tests re-run after each, behaviour-changing rewrites dropped; instruction = request + clumsy code, output = clean original. **Runs internet code: Colab only**
- `sources/commitpack_refactor.py` — "improve this code" records from real CommitPackFT refactor commits: subject must say refactor/simplify/clean up/improve/optimise/readability (not fix/test/docs/text), exactly one function changed, not only strings, both versions short. Nothing runs; yields ~0.3% of commits (~150–200 total)
- `sources/common.py` also holds `IMPROVE_TEMPLATES` (shared request wordings) and `pick()` (stable template choice per id)
- `experiments/code_runner.py` is shared with the code-running sources (`run_python_capture()` also returns stdout)
- Source settings (HF id, config, split, fetch count, filters) live in `CFG.dataset_v2.sources` — never in code
- `explain` records are text only (code blocks removed) — explain mode answers in words
- `generate` records are code only — no fences, no example-usage / test tail
- v1 dataset stays at `data/datasets/train.jsonl`; v2 must be written to a different path

### `data/scripts/build_dataset_v2.py` + `data/scripts/decontaminate.py` (dataset v2, step 5)
- `build_dataset_v2.py` — mixes the per-source files into `CFG.dataset_v2.build["output_dir"]` (`train.jsonl`, `val.jsonl`, `build_report.json`): drops exact duplicates (output or instruction, via `dedupe.hash_code`), benchmark overlap, and records over `max_tokens` (never truncates — a cut answer loses its ending); then a seeded sample of `take[source]` per source; val split per task
- `decontaminate.py` — `BenchmarkIndex`: any 10-word run shared with an MBPP (all configs/splits) or HumanEval problem statement, or identical normalised code to a benchmark solution → record dropped. Keeps the MBPP scoring test honest
- Mix settings (`take`, `max_tokens`, `seed`, `val_share`) live in `CFG.dataset_v2.build`
- Runs on Colab (the source files are on Drive); needs no GPU and runs no code

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
- `dataset_loader.py` — `get_tokenized_dataset(tokenizer)`: reads `CFG.paths.dataset` / `val_dataset`, builds input_ids + labels per record (mode = `metadata.task`), drops over-long records and prints how many; shared by both trainers
- `train_lora_t4.py` — Colab trainer: fresh from plain Phi-2, hyperparameters from `CFG.training`, T4 batch settings (4 × 4) in the script, `--output-dir` (Drive) with automatic resume, eval loss every `eval_steps`
- `train_lora.py` — laptop trainer (GTX 1650), all settings from `CFG.training`; 768-token examples may not fit in 4 GB — train on Colab

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

> RAG is turned off in `configs/config.yaml` (`rag.enabled: false`) since 2026-09-25 — weak dataset matches were copied into answers. Rules below still apply when it is turned back on.

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

## Phase 11 Rules — Long-Term Memory (Planned)

Phase 11 gives Py-V a memory that lasts across all chats. Every message is saved to a local SQLite database, important facts are tagged, and before each answer the memory is searched and the best matches are added to the prompt. Goal: better memory. It does not make the model smarter or faster. All rules below apply when implementing Phase 11.

Decisions (agreed 2026-09-25):

| Topic | Decision |
|-------|----------|
| Scope | Across all chats, not per session |
| What is saved | Every user + assistant message, plus tagged key facts |
| Search | Both: keyword search (SQLite FTS5) + meaning search (embeddings) |
| When used | All modes get short facts; code from memory only in `generate`, `debug`, `refactor` |
| Old or wrong facts | Newest fact wins; user can list and delete memories |
| Storage | SQLite — one local file, Python standard library `sqlite3` |

Planned layout:

- `memory/` — new top-level package, same level as `retrieval/`
- `memory/schema.py` — dataclasses only (`MemoryItem`, `Fact`), no logic
- `memory/store.py` — SQLite access: tables for messages and facts, FTS5 index, save / update / delete / list
- `memory/extractor.py` — rule-based fact tagging (explicit "remember", file names, error messages, versions, decisions) — no model calls
- `memory/search.py` — hybrid search: FTS5 keyword score + embedding similarity + recency boost, merged and ranked
- `configs/config.yaml` — new `memory:` section (`enabled`, `db_path`, `top_k`, `max_prompt_tokens`, `active_code_modes`)
- `inference/engine/chat.py` — search memory before prompt build, save the turn after the reply
- `inference/engine/prompt_builder.py` — `format_memories()`; memory is formatted here only
- `inference/engine/prompt_templates.py` — `{memories}` slot added to templates
- `inference/api/routes.py` — `GET /api/v1/memory` (list) and `DELETE /api/v1/memory/{id}` (delete)

Rules:

- Memory reuses `retrieval/embedder.py` — never add a second embedding model or loader
- All memory settings come from `CFG.memory.*` — never hardcode the DB path, top_k or token budget
- Memory gets a small, fixed prompt budget (a few short items) — the Phi-2 context is only ~2,048 tokens and RAG already uses part of it
- `explain` and `chat` modes receive facts only — never code snippets from memory (same code-bias reason as the RAG rule)
- Facts are short tagged statements, not raw past messages — the existing ban on injecting context history into `explain` / `chat` prompts still applies
- Facts carry a timestamp and a key; a newer fact with the same key replaces the older one (older one marked inactive, not silently lost)
- Fact extraction is rule-based first — Phi-2 is not reliable enough to judge what is important
- If the database is missing or broken, chat continues without memory — no crash (same as RAG)
- Memory runs on CPU / disk only — it must not use VRAM
- The memory database holds private conversations — it must be gitignored and never committed
- SQLite replaces `sessions/*.json` as the store for chat history; `context_manager.py` reads and writes through `memory/store.py`

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
- Inject code snippets from memory into explain or chat mode prompts (Phase 11)
- Duplicate embedding logic — memory reuses `retrieval/embedder.py` (Phase 11)
- Commit the memory database or any chat history to git (Phase 11)
- Use a database server — memory is SQLite only (Phase 11)

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

# Scoring test (loads the model, ~30–50 min for 100 problems — close other heavy apps first)
python -m experiments.eval_mbpp
python -m experiments.eval_mbpp --base
python -m experiments.eval_mbpp --adapter model/lora_v2

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
          Memory DB (Phase 11) ↔ Inference API
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