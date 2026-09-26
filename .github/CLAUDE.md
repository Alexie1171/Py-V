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
| Phase 2 | Base model setup & 4-bit inference (Phi-2; brain upgraded to IBM Granite 3B on 2026-09-26) | Complete |
| Phase 3 | Full data pipeline (scrape → clean → dedupe → format) | Complete |
| Phase 4 | LoRA fine-tuning on Python dataset | Complete |
| Phase 5 | FastAPI inference server | Complete |
| Phase 6 | VS Code extension | Complete |
| Phase 7 | Chat system (context-aware assistant + controller) | Complete |
| Phase 8 | RAG (Retrieval Augmented Generation) | Complete — turned off since 2026-09-25 (see `PROJECT_STATUS.md`) |
| Phase 9 | Multi-LoRA adapters (multi-language support) | Planned |
| Phase 10 | VS Code chat panel (full UI, no terminal) | Planned |
| Phase 11 | Long-term memory (SQLite, across all chats, keyword + meaning search) | Built 2026-09-26 (`memory/`, on by default) — smoke test passes; not yet tried with a trained Granite |

Build order: better training data + GPU retrain (the 8.1.x commits; Granite retrain runs in the one-button GPU pipeline — Kaggle first, Colab as backup) and Phase 11 (built 2026-09-26) → then Phases 9 and 10 and speed work (numbering kept stable on purpose). Current phase: **Phase 11** — commits are numbered 11.x from here ("Phase 11: …", then 11.1, 11.1.1, …)

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
- **Brain: IBM Granite 3B** — `ibm-granite/granite-4.1-3b-base` (config `model.name`), Apache 2.0, 128K context. First brain upgrade on 2026-09-26, replacing Phi-2 (2.7B, 2,048-token limit) — see `PROJECT_STATUS.md`. The chat version `ibm-granite/granite-4.2-3b` (same layers, own chat format, optional thinking mode) is under test (GPU pipeline: both versions tested untrained, trained and tested) and may become the starting point
- Brains live only in the Hugging Face cache (`HF_HOME`, `E:\huggingface Assets`), never in the repo
- Fine-tuned LoRA adapter at `model/lora/` — the adapter inference loads. **None exists for Granite yet**: until the GPU pipeline trains one, the app runs the plain brain (`load_lora_model()` falls back and says so)
- An adapter only fits the brain it was trained on — `load_lora_model()` refuses an adapter whose `base_model_name_or_path` differs from `model.name`
- LoRA layer names depend on the brain: config `training.lora_target_modules` (Granite: q/k/v/o_proj, gate/up/down_proj) — change it together with `model.name`
- Training runs in the GPU pipeline (Kaggle T4, Colab T4 as backup) into a folder named after the brain, `{root}/model_<brain>/lora` (Kaggle: the run's output `PY-V/model_<brain>/lora`; Colab: Drive `MyDrive/PY-V/model_<brain>/lora`); download the finished adapter into `model/lora/`
- Phi-2 history (retired 2026-09-26, all local files deleted): v2 adapter scored 66/100 MBPP on the laptop (plain Phi-2 62), 1/10 long questions; adapters remain on Drive (`MyDrive/PY-V/model_v2/lora`, `MyDrive/Py-V/Py-V/model_t4/lora`)

Rules:
- No training from scratch
- No models >7B parameters
- Always use PEFT / LoRA fine-tuning
- Always load base model with 4-bit BitsAndBytes quantization
- Always resume from checkpoint when one exists (`resolve_checkpoint()` on the laptop, `get_last_checkpoint(output_dir)` on Kaggle/Colab)
- One GPU per job: `load_model()` puts the whole model on the first visible GPU (`device_map={"": 0}`, never `"auto"` — on a 2-GPU machine that splits the model across both, and the batch-size probe only measures GPU 0); `train_lora_t4.py` defaults `CUDA_VISIBLE_DEVICES=0` (else the Trainer wraps the 4-bit model in DataParallel). Use a second GPU by running a second job on it, never by splitting one job
- Every training example = the inference prompt for its mode (`build_training_prompt()`) + answer + end-of-text token; loss on the answer only (prompt labels -100)
- Examples longer than `max_seq_length` are dropped, never truncated — a cut answer has no end token and teaches the model not to stop
- Labels are built in `dataset_loader.py` and padded with -100 (`DataCollatorForSeq2Seq`). Never use `DataCollatorForLanguageModeling`: pad == eos (Phi-2, Granite), so it masks the end token (the Phi-2 v1 adapters never learned to stop because of this)
- Pass `label_names=["labels"]` to `TrainingArguments` — PeftModel hides the labels argument and no eval loss is computed without it

---

## File Responsibility Rules

### `configs/config.yaml`
- Single source of truth for ALL configuration
- Model name, paths, training hyperparameters, RAG settings all live here
- Never duplicate values from here into code

---

### `model/training/config_loader.py`
- Parses `config.yaml` into typed dataclasses (`ModelConfig`, `GenerationConfig`, `TrainingConfig`, `PathsConfig`, `RAGConfig`, `DatasetV2Config`, `EvaluationConfig`)
- Exports a module-level `CFG` singleton
- All other modules import `CFG` from here — never re-parse yaml elsewhere

---

### `model/utils/model_loader.py`
- Single shared model loader for the entire project
- Reads model name from `CFG.model.name`
- Both `inference/engine/` and `model/training/` use this — no duplication
- Tags the loaded model with its prompt format: `model.v_prompt_format` (config `model.prompt_format`: `template` or `native_chat`) and its splitting rules: `model.v_split_rules` (recorded in every test summary as `split_rules`)
- Holds `ADAPTER_META = "v_adapter.json"` — the note `train_lora_t4.py` writes next to every adapter (brain, prompt format, splitting rules, dataset, GPU setup, losses)
- `load_tokenizer(model_name, split_rules_from)` — the ONLY way to load a brain's tokenizer (also used by `build_dataset_v2.py` and `eval_long_context.py`; never `AutoTokenizer.from_pretrained` directly): normalizer, pre-tokenizer and decoder are always taken from a `tokenizer.json`, so text is cut up the same way in every transformers version. transformers **5.0.0** (Kaggle, 2026-09) rebuilds a "GPT2Tokenizer" from vocab + merges with GPT-2's splitting rule: Granite 4.1 base got text in pieces it never learned (same question 558 → 683 tokens, MBPP 69 → 17). Checked on the laptop CPU: with the fix, identical ids in 4.57, 5.0.0 and 5.17
- Whose `tokenizer.json`: `split_rules_source()` = argument, else config `model.split_rules_from`, else the brain's own. Another brain's rules must have the same merges (else ValueError), and a missing file for them is an error, never silently skipped. Reason: Granite 4.2 (chat)'s own file has GPT-2's rule (probably saved with the 5.0 bug) — same merges as 4.1, whose file has the Granite family's rule (numbers in groups of up to 3: 1650 → `165` `0`; 4.2's file: `16` `50`)

---

### `inference/engine/model_loader.py`
- Wraps `model/utils/model_loader.py`
- Also exposes `load_lora_model()` — loads base model then applies LoRA adapter
- This is what the API server calls at startup
- The adapter's `v_adapter.json` sets `model.v_prompt_format` — an adapter is always used with the prompt format it was trained with, whatever config says

---

### `inference/engine/prompt_builder.py`
- Central prompt formatting for the brain (`### Instruction:` / `### Answer:` templates — work for base models; chat versions can use `to_native_chat()`)
- Used by BOTH training (`dataset_loader.py`) and inference (`generator.py`)
- Prompt format: the per-mode templates in `prompt_templates.py` (`### Instruction:\n...\n\n### Answer:\n`)
- `build_training_prompt(mode, instruction)` = `build_prompt(mode, instruction, {})` — training sees exactly what inference sends (no history, no RAG); `build_inference_prompt()` = the generate template, for the stateless `/generate` endpoint
- Any template change means retraining — the adapter learns the exact wording
- `to_native_chat(prompt, tokenizer)` re-wraps a template prompt in the model's own chat format (`apply_chat_template`, `enable_thinking=False`) for chat-tuned brains; unchanged for models without a chat template
- `format_for_model(prompt, model, tokenizer)` — the prompt exactly as the loaded model gets it: template, or `to_native_chat()` when `model.v_prompt_format == "native_chat"`. The generator calls it for every generation; `dataset_loader.py` does the same wrap for training — so app, tests and training always match. Everything else passes plain template prompts
- Never define prompt format in any other file
- Exposes: `build_prompt()`, `build_training_prompt()`, `build_inference_prompt()`, `to_native_chat()`, `format_for_model()`, `format_context()`, `format_retrieved_context()`
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
- `stop_strings` also match across the prompt/answer boundary (the prompt ends in "\n"), so a stop word must never match the start of a legitimate answer — code-mode test stops need a blank line first (`"\n\ndef test_"`); `"\ndef test_"` killed a function named `test_duplicate`
- The current adapter (`model/lora/`) was trained without an end-of-text token, so it does not stop on its own — stop words are its only brake. The v2 adapter stops on its own (end token learned); stop words stay as a safety net
- Retry logic uses temperature 0.5 on second attempt for chat/explain modes
- Repeat settings come from `CFG.generation.for_mode(mode)` (config `generation:` section, `default` for unlisted modes) — never hard-code them here. Code modes (generate/debug/refactor): `repetition_penalty` 1.0 and `no_repeat_ngram_size` 0 — code must copy names and the user's code from the prompt (1.1 renamed `find_Volume` → `find_volume`, 13 of 49 MBPP failures). Prose modes (chat/explain): 1.1 and 4, against loops

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
- Result names always include the brain: `base_<brain>` or `<brain>_<adapter folder>` (files from before 2026-09-26 named `base` / `lora` / `lora_v2` are Phi-2)
- `--base` scores the brain without the LoRA adapter for comparison; `--adapter DIR` scores another adapter (default `CFG.paths.model_output`); `--base --model NAME` scores another base model (tag `base_<name>`)
- Every run also writes `mbpp_{tag}_summary.json`: score + every setting that can change it (base model, adapter md5, decoding settings, benchmark, GPU, library versions). Only compare runs whose settings match
- **Big training and tests run on a cloud T4** (owner rule, 2026-09-26) — the laptop took hours per test. Kaggle is the main place (2026-09-26: 30 GPU h/week, 12 h per background run, 2× T4); Colab is the backup (~4 h/day). Scores compare only within one machine type (the summary records the GPU and library versions): the 2026-09-25 laptop scores are a laptop-only baseline, so any model compared on the T4 gets its own T4 run with the same settings
- The laptop is only for short checks (e.g. "does this fit in 4 GB / how much RAM"), and only after telling the owner. If the cloud GPU quota is used up: wait for the reset, or ask the owner before using the laptop

---

### `experiments/eval_long_context.py`
- Long-question test: "find the bug in this long file" at ~500 / 1,000 / 1,500 / 3,000 / 6,000 tokens (counted with Phi-2's tokenizer when the kept set was built; 2 questions each), debug mode, same generation path as the app, 320-token answer budget
- Modules built from MBPP **full** train/validation/prompt solutions (never the test split `eval_mbpp.py` scores on); one bug planted with `data/scripts/sources/mutations.py`; graded by running the target function's tests with the answer loaded on top of the buggy module
- Questions built once and kept in git as `experiments/longctx_tasks.json` — every model, on the laptop and on the cloud T4, gets the same ones; delete the file only to deliberately make a new question set
- Records prompt tokens, pass/fail and peak GPU memory per question; skips questions longer than the model's context window; catches out-of-memory and skips bigger sizes
- Same flags as `eval_mbpp.py` (`--adapter`, `--base`, `--model`, `--native-chat`); results `longctx_{tag}.jsonl` + `_summary.json`

---

### `experiments/eval_chat.py`
- Short chat test, 8 questions, one skill each: explain a concept, answer from memory notes, admit what it doesn't know, answer from search results, follow up on an earlier turn, ask for a search (`SEARCH: …` line) when it lacks information, follow a format, keep it simple
- Chat mode, same generation path as the app, greedy; keyword checks are a rough signal — every answer is printed in full for a person to judge
- Notes / search results / earlier turns are put into the question text (the app's context slot is off until Phase 11)
- Results `chat_{tag}.jsonl` + `_summary.json`

---

### `experiments/eval_fix.py`
- Fix / improve test — what MBPP doesn't measure. 40 fix questions (bug planted inside the tested function, failing test shown, debug mode; pass = all the problem's tests pass) + 40 improve questions (clumsy rewrite of the tested function via `improve_synthetic.clumsify()`, refactor mode; pass = tests still pass AND fewer syntax-tree nodes than the clumsy version)
- Held-out MBPP problems: sanitized test from problem 101 on + full train/validation/prompt — never in training (decontamination drops all MBPP)
- Graded on top of the code the model was shown, so helpers next to the tested function still exist
- Questions kept in git as `experiments/fix_tasks.json`; results `fix_{tag}.jsonl` + `_summary.json`

---

### `experiments/eval_all.py`
- All four tests (MBPP, long-file, chat, fix/improve) with ONE model load — each script exposes `run(model, tokenizer, tag, args)`; `--skip-done` skips tests whose summary exists; a crashing test doesn't stop the others (non-zero exit)
- Used by the one-button GPU pipeline (`scripts/gpu_pipeline.py`)

---

### `experiments/eval_common.py`
- Shared by the scoring scripts: `add_model_args()` (`--base`, `--adapter`, `--model`, `--native-chat`, `--split-rules-from`), `result_tag()`, `load_for_eval()` → (model, tokenizer, tag)
- The prompt format travels with the loaded model (`model.v_prompt_format`) and the generator applies it — scripts pass plain template prompts. `--native-chat` forces the brain's own chat format (thinking off) for an untrained chat brain; tag gets `_native`. A trained adapter uses the format from its `v_adapter.json`
- `--split-rules-from NAME` gives a base brain another brain's splitting rules (tag gets `_split-<brain>`); a trained adapter always uses the rules in its `v_adapter.json`
- Runs on the cloud T4 (Kaggle / Colab) like every big test. On the T4 (15 GB) it measures ability; how much fits on the laptop (4 GB) is a separate short laptop check
- On Windows the GPU driver spills into system RAM instead of failing when GPU memory is full ("shared GPU memory") — watch system RAM during long questions on the laptop; Qwen3-4B took it to 15.1 of 15.4 GB
- MBPP / HumanEval are for scoring only — never add them to training data

---

### `experiments/code_runner.py`
- `run_python(program, timeout)` — runs code in a separate process, temp dir, timeout
- Guard disables file delete/rename/write, process start, sockets — best-effort, NOT a real sandbox
- Untrusted code from the internet (dataset checks) runs on Kaggle / Colab, not the laptop

---

### `scripts/gpu_pipeline.py`
- The one-button GPU run (owner rule, 2026-09-26: "one button run … we get all the outputs we need to progress further") — same code on Kaggle and Colab; replaced `scripts/colab_pipeline.py` on 2026-09-26. Stages: 1 test untrained Granite chat (own chat format), 2 test untrained Granite plain, 3 training data (sources copied from `--inputs` or an earlier run, missing ones re-made — `old_github` from the v1 dataset found by fingerprint — then the long-file fix examples), 4 build dataset v3, 5 train chat (native_chat), 6 test it, 7 train plain (template), 8 test it, 9 extra: chat with V's template
- Stage 10 (added 2026-09-26): test the chat version with the Granite family's splitting rule (`CHAT_SPLIT_FIX` = the plain version's `tokenizer.json`); stage 5 trains the chat version with the rule of stage 1 or 10 that passed more questions across all four tests (`chat_split_rules()`, ties → its own file) and logs the comparison
- Lanes run at the same time, every job seeing only its own GPU (`CUDA_VISIBLE_DEVICES`): data lane on the CPU (3 → 4; trainings wait for it). Two GPUs (Kaggle T4 x2): chat lane on GPU 0 (1 → 10 → 5 → 6), plain lane on GPU 1 (2 → 7 → 8). One GPU (Colab): 1 → 2 → 10 → 5 → 6 → 7 → 8. Stage 9 goes to whichever GPU lane is free first, also while one waits for the data. Lines are prefixed with the lane name; progress bars are thinned to one a minute per lane
- `scripts/pipeline_redo.json` (in git): one-time re-runs of stages whose saved results are wrong — each entry (id → stages + why) is applied once per root after the earlier run is copied in: test results / training set / long-file file removed, a trained adapter moved to `lora_before_<id>`; applied ids kept in `results/redo_done.json`. First entry `2026-09-26-tokenizer`: stages 2 (plain test) and 4 (training set) from Kaggle run 1
- Everything is saved under `--root` (Kaggle `/kaggle/working/PY-V` = the run's output; Colab `/content/drive/MyDrive/PY-V`): the pipeline links `data/raw/v2` → `results/data_v2`, `data/datasets/v3` → `results/dataset_v3`, `experiments/outputs` → `results/eval` itself (refuses if one is a real non-empty folder); adapters in `model_<brain>/lora`
- `--inputs` (Kaggle: `/kaggle/input`): an earlier run's saved folder (found by its `results/PIPELINE_REPORT.json`) is copied into root first, keeping files already there — that is how a Kaggle run continues; uploaded `<source>.jsonl` files are copied in for missing sources (`.part` + rename)
- Every stage checks its results first and is skipped when done; training resumes from its checkpoint. Stages whose inputs are missing are "blocked", not crashed; a Python error in a stage is caught and marked FAILED, the other lanes go on
- `--stop-after H` (Kaggle: 11 — the limit is 12 h a run): jobs still running then are killed, later stages marked "not started (run time limit)", and the run ends normally so its output is saved. Per-job timeouts: tests 4 h, training 8 h, fetch 3 h, build 1 h (timer-based, fires even when a job prints nothing)
- After every stage change: `{root}/results/PIPELINE_REPORT.md` + `.json` (stage status + where it ran, all scores, training notes); the whole console output is appended to `results/pipeline_log.txt`. Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for its jobs
- New big jobs go into this pipeline as stages, not only into single notebook cells

---

### `Kaggle/py_v_kaggle.ipynb` (git-tracked)
- Main runner for heavy jobs since 2026-09-26 (Colab quota ran out): one code cell **▶ RUN EVERYTHING** — refuses outside Kaggle, clones/pulls the public GitHub repo to `/tmp/Py-V` (code is not part of the output), `pip install -U peft bitsandbytes`, lists `/kaggle/input`, then `gpu_pipeline.py --root /kaggle/working/PY-V --inputs /kaggle/input --stop-after 11`
- Runs as a **background run** (Save Version → Save & Run All): no idle timeout, the laptop can be off; the owner starts it in the browser. Settings: Accelerator GPU T4 x2 (the P100 no longer works with Kaggle's PyTorch since 2026-04), Internet on, input = the private dataset with the training source files (from Drive `results/data_v2` + laptop `data/raw/v2/old_github.jsonl`)
- Kaggle has no Drive: each run starts empty. To continue a stopped run, add this notebook's latest output as an input — the pipeline copies it back
- Results: the version's output folder `PY-V/` (same layout as on Drive). The assistant reads them on the laptop with the Kaggle CLI (`kaggle kernels status / output`, key in `~/.kaggle/kaggle.json` — never in the repo) into `Kaggle downloads/` (gitignored)
- Kaggle also offers a VS Code connection (Run ▸ Kaggle Jupyter Server → VS Code URL); not used for the big run — that session needs the laptop connected and loses `/kaggle/working` when it ends
- Kaggle quota (2026-09-26): 30 GPU h per week, 12 h per run, 2× T4 counts as one GPU hour

---

### `Google Colab/py_v_runner.ipynb` (gitignored)
- Backup runner (one T4, results on Drive) — two code cells only:
  - **▶ RUN EVERYTHING**: its own setup (mount Drive, clone/pull the public GitHub repo, install peft + bitsandbytes), then `scripts/gpu_pipeline.py --root /content/drive/MyDrive/PY-V` (the pipeline links the output folders to Drive itself)
  - **👀 CHECK PROGRESS**: read-only, no GPU (CPU runtime fine) — prints the pipeline report, the end of the log, and each Granite training's saved steps
- The old setup cells 1a–1d and single-job cells (A–H) were removed — every job is a pipeline stage now; their results are recorded in `PROJECT_STATUS.md`
- Code comes from GitHub — local changes must be pushed before running (both runners)
- Results go to Drive `MyDrive/PY-V/results/`; adapters to `MyDrive/PY-V/model_<brain>/lora` — never to an existing adapter folder of another brain
- The assistant writes/updates the runners; the owner runs them. Colab results are read back from the saved notebook file, Kaggle results with the Kaggle CLI
- Heavy jobs (full dataset fetch, training, MBPP scoring, long-question test, brain checks) go to Kaggle or Colab, not the laptop (owner rule, 2026-09-26)
- Kaggle and Colab keep separate results (Kaggle output vs Drive) — finish a job where it started
- Colab's free GPU time runs out after roughly 4 hours of T4 use in a day and resets within ~12–24 h — plan jobs to fit, and put the most important job first

---

### `data/scripts/fetch_sources.py` + `data/scripts/sources/` (dataset v2)
- `fetch_sources.py` — streams each v2 source, converts rows, writes `{CFG.dataset_v2.output_dir}/{source}.jsonl`, reports kept / scanned / rejection reasons (never silent caps)
- `fetch_sources.py` writes each source to a `.part` file and renames it when complete — a source file that exists is always finished (the pipeline relies on this)
- `fetch_sources.py` closes each stream and ends with `os._exit(0)` — safeguard so a half-read Hugging Face stream can't keep a download thread alive after the files are written
- `sources/{name}.py` — one module per source, each `iter_records(cfg, stats)` → PY-V records with `metadata.task` (`generate` / `debug` / `refactor` / `explain`) and `metadata.license`
- `sources/common.py` — shared helpers only (record builder, fenced-code extraction, demo-code trimming, docstring removal)
- `sources/mutations.py` — realistic single bugs (wrong comparison/operator, off-by-one range, name typo, missing cast, missing return, flipped bool, and/or swap, `None` init), spliced into the original text so the fixed code is the untouched original; pure AST work, runs nothing
- `sources/bug_fix.py` — "fix the error" (`debug`) records: OpenCodeInstruct functions that pass their unit tests → one bug → tests re-run to capture the real error or failing check → instruction = what the user saw + broken code, output = original code + one-sentence fix. Skips rows already used for write-code. **Runs internet code: refuses to run outside Kaggle / Colab** unless `PYV_ALLOW_LOCAL_EXEC=1`
- `sources/unit_tests.py` — shared by the code-running sources: `tested_functions()` (OpenCodeInstruct functions that pass their tests upstream AND here, skipping ids used by other record files), `run_tests()` (first failure as JSON, incl. the wrong value for failing `==` asserts), `require_cloud()` (Kaggle via `KAGGLE_KERNEL_RUN_TYPE`, Colab via `COLAB_RELEASE_TAG`)
- `sources/unrefactor.py` — the reverse of refactoring: clean code → clumsy code that should behave the same (comprehension → loop, `sum()` → loop, `return a == b` → if/else, enumerate → `range(len())`, truthiness → `len()`, max/min → if/else, ternary → if/else, `+=` → `x = x + ...`); pure AST work, runs nothing
- `sources/improve_synthetic.py` — "improve this code" (`refactor`) records: clumsy rewrites applied one at a time, tests re-run after each, behaviour-changing rewrites dropped; instruction = request + clumsy code, output = clean original. **Runs internet code: Kaggle / Colab only**
- `sources/long_file_fix.py` — "fix the bug in this file" (`debug`) records: a tested OpenCodeInstruct function hidden among other working functions (earlier functions of the stream), one bug planted INSIDE it, tests re-run inside the file; output = ONLY the corrected function + one-line explanation (teaches "fix just the broken part" on long input). Bug kinds rotated. **Runs internet code: Kaggle / Colab only**
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
- Mix settings (`take`, `max_tokens`, `max_tokens_per_source`, `seed`, `val_share`) live in `CFG.dataset_v2.build`; tokens counted with the config brain's tokenizer
- Dataset v3 (2026-09-26) = v2's mix + 1,200 `long_file_fix` records (limit 930 tokens) → `data/datasets/v3`; v2 stays as it was (Phi-2 was trained on it)
- Runs in the GPU pipeline's data lane on Kaggle / Colab (where the source files are); needs no GPU and runs no code

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
- `dataset_loader.py` — `get_tokenized_dataset(tokenizer, prompt_format)`: reads `CFG.paths.dataset` / `val_dataset`, builds input_ids + labels per record (mode = `metadata.task`; prompt in V's template or, for `native_chat`, re-wrapped exactly like `format_for_model()`), drops over-long records and prints how many; the answer is tokenized without special tokens; shared by both trainers
- `max_seq_length` 1024 (since 2026-09-26, was 768) so long-file examples fit
- `train_lora_t4.py` — cloud T4 trainer (Kaggle / Colab): fresh from the plain brain in config, LoRA layers from `CFG.training.lora_target_modules`, hyperparameters from `CFG.training`, batch size + gradient checkpointing measured on the GPU at start (`pick_batch_setup()`: checkpointing off if it fits, then the biggest batch whose worst case — every example at `max_seq_length` — stays under 85% of GPU memory; effective batch always 16, so results don't depend on the GPU; every try is printed with its peak memory), `--output-dir` with automatic resume, eval loss every `eval_steps`; `--model` / `--prompt-format` / `--split-rules-from` override config (the pipeline trains both Granite versions). At the end writes `v_adapter.json` (brain, prompt format, splitting rules, dataset, steps, losses, GPU setup, minutes) and `training_log.json` (every logged loss) next to the adapter. Kaggle has transformers **5.0.0**, Colab 5.x, the laptop 4.57 — the script must run on all (e.g. `_length_grouping()`: v5 replaced `group_by_length=True` with `train_sampling_strategy="group_by_length"`; checked on the laptop CPU with a tiny Granite under 5.0.0 + peft 0.21: settings accepted, steps + eval loss + checkpoint work)
- `pick_batch_setup()` calls `model.train()` first: Hugging Face only uses gradient checkpointing in training mode, and a freshly loaded model is in eval mode — Kaggle run 1 measured without checkpointing and failed both trainings with "Not even one max-length example fits on this GPU" (checked on the laptop CPU: eval mode → checkpoint not used, train mode → used and recomputed in backward, in 4.57 and 5.0.0)
- `train_lora.py` — laptop trainer (GTX 1650), all settings from `CFG.training`; 768-token examples may not fit in 4 GB — train on Kaggle / Colab

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
- Base model (the brain, 4-bit) is shared — only adapter weights change between languages
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

## Phase 11 Rules — Long-Term Memory (built 2026-09-26)

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

Layout (as built):

- `memory/` — new top-level package, same level as `retrieval/`
- `memory/schema.py` — dataclasses only (`MemoryItem`, `Fact`), no logic
- `memory/store.py` — SQLite access: tables for messages and facts, FTS5 index, save / update / delete / list
- `memory/extractor.py` — rule-based fact tagging (explicit "remember", file names, error messages, versions, decisions) — no model calls
- `memory/search.py` — hybrid search: FTS5 keyword score (stopwords dropped) + embedding similarity (≥ 0.60, bge-small) + recency boost (halves every 30 days), weights 0.45 / 0.45 / 0.10
- `memory/manager.py` — `MemoryManager`, the one entry point: `remember_turn()`, `recall()`, `list_facts()`, `forget()`, session state; embedder loaded lazily on the CPU (keyword-only if it can't load)
- `memory/test_memory.py` — smoke test, no brain needed: `python -m memory.test_memory`
- `configs/config.yaml` — `memory:` section (`enabled`, `db_path` = `data/memory/v_memory.db`, `top_k`, `code_top_k`, `max_prompt_tokens`, `history_turns`, `active_code_modes`, `semantic_search`)
- `inference/engine/chat.py` — search memory before prompt build, save the turn after the reply
- `inference/engine/prompt_builder.py` — `format_memories()`; memory is formatted here only
- `inference/engine/prompt_templates.py` — **unchanged**: the memory block goes into the templates' existing `{context}` slot (a new slot would change the training prompts and void the trained adapters)
- `inference/api/routes.py` — `GET /api/v1/memory` (active facts + counts) and `DELETE /api/v1/memory/{fact_id}` (forget one fact for good); `/chat` replies carry `memories` (items used)
- `inference/engine/context_manager.py` — session state through `MemoryManager` (SQLite) instead of `sessions/*.json`; RAM-only when memory is off
- `retrieval/embedder.py` — `Embedder(device=...)`; memory passes `"cpu"`

Rules:

- Memory reuses `retrieval/embedder.py` — never add a second embedding model or loader
- All memory settings come from `CFG.memory.*` — never hardcode the DB path, top_k or token budget
- Memory gets a small, fixed prompt budget (a few short items) — Granite can read 128K tokens, but on the 4 GB laptop GPU prompts over ~1,500 tokens spill into system RAM and slow down (long-question test); RAG shares the same budget
- `explain` and `chat` modes receive facts only — never code snippets from memory (same code-bias reason as the RAG rule)
- Facts are short tagged statements, not raw past messages — the existing ban on injecting context history into `explain` / `chat` prompts still applies
- Facts carry a timestamp and a key; a newer fact with the same key replaces the older one (older one marked inactive, not silently lost)
- Fact extraction is rule-based first — a 3B brain is not reliable enough to judge what is important (revisit with Granite's chat version)
- If the database is missing or broken, chat continues without memory — no crash (same as RAG)
- Only the user's own messages are read for facts — V's answers can contain made-up facts; statement rules skip questions
- Earlier code comes only from V's answers given in the code modes, and only code modes receive it
- The API server loads the brain once and passes it to `ChatEngine(model, tokenizer)` — never load it twice (it used to, which alone fills a 4 GB GPU)
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

# Brain answers — short chat test (big tests go to Kaggle / Colab)
python -m experiments.eval_chat --adapter model/lora

# Chat system (terminal — Phase 7/8)
python test_chat.py

# Scoring test (big test — run it on Kaggle / Colab via a runner notebook; ~15–25 min on the T4, 30–110 min on the laptop)
python -m experiments.eval_mbpp
python -m experiments.eval_mbpp --base
python -m experiments.eval_mbpp --adapter model/lora_v2

# Long-question test (big test — run it on Kaggle / Colab; on the laptop ~10–40 min per model, watch system RAM)
python -m experiments.eval_long_context --adapter model/lora_v2

# Memory smoke test (no brain, CPU only)
python -m memory.test_memory

# All four tests with one model load (big — Kaggle / Colab); fix/improve test alone
python -m experiments.eval_all --adapter model/lora --skip-done
python -m experiments.eval_fix --adapter model/lora

# One-button GPU run (cloud only, from a runner notebook's RUN EVERYTHING cell)
python -m scripts.gpu_pipeline --root /kaggle/working/PY-V --inputs /kaggle/input --stop-after 11   # Kaggle
python -m scripts.gpu_pipeline --root /content/drive/MyDrive/PY-V                                  # Colab

# Kaggle run status / results on the laptop (Kaggle CLI + key in ~/.kaggle/)
kaggle kernels list --mine
kaggle kernels status <user>/<notebook>
kaggle kernels output <user>/<notebook> -p "Kaggle downloads"

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