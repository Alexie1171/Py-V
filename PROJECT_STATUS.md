# Py-V — Project Status

_Last updated: 2026-09-25_

The rulebook for how code must be written is `.github/CLAUDE.md`. This file is the plain-language picture: where the project is, what is broken or unfinished, where it is going, and the limits we work within.

---

## 1. What Py-V is

A private coding assistant that runs entirely on this laptop.

- **Brain:** Microsoft Phi-2 (a small 2.7B-parameter model), loaded in 4-bit so it fits a 4 GB GPU.
- **Training:** LoRA fine-tuning on Python examples scraped from GitHub and StackOverflow.
- **Server:** FastAPI app that loads the model once and answers requests.
- **Chat:** understands what you want (write / fix / explain / improve code / just chat) and keeps a short session history.
- **RAG:** before answering code questions it looks up similar code from the dataset and this codebase.
- **VS Code extension:** sends requests from the editor to the server.

---

## 2. Where we are

Work paused on **2026-05-04** and resumed on **2026-09-25**.

| Phase | What | Status |
|-------|------|--------|
| 1 | Project structure | Done |
| 2 | Phi-2 setup, 4-bit | Done |
| 3 | Data pipeline (scrape → clean → dedupe → format) | Done |
| 4 | LoRA fine-tuning | Done |
| 5 | FastAPI server | Done |
| 6 | VS Code extension | Done |
| 7 | Chat system (intent routing + session context) | Done |
| 8 | RAG | Done, but **turned off** since 2026-09-25 — see section 5 |
| 9 | Multi-language adapters (JS/TS etc.) | Planned — nothing started |
| 10 | VS Code chat panel | Planned — nothing started |
| 11 | Long-term memory | **Next up** — built before 9 and 10; design agreed (see section 6) |

**Build order:** loose ends (section 4) → **better training data + one Colab retrain** (debug/refactor/explain examples, cleaner v4, end-of-text token) → Phase 11 → Phases 9, 10 and speed work (order of those three not decided yet).

**Things switched off or put off for later** are listed in section 5 — check it before starting new work.

### What happened last (May 2026)

1. First fine-tune (epoch 1) ran locally on the GTX 1650: 115 steps.
2. The next round gave bad quality, so the scrapers were improved and the dataset rebuilt: **5,508 training / 612 validation** examples.
3. Training on the laptop was too slow — the local run stopped at step 50 of 345.
4. Training moved to **Google Colab (T4 GPU)**. That run **finished**: 1,377 steps, continuing from the epoch-1 model. Training loss went from about 0.95 to about 0.77 (noisy, 0.63–1.3 near the end).
5. The Colab adapter was copied into `model/lora/`, so **the app already uses the newest model**.
6. Last change before the pause: cleaner v4 (`data/scripts/cleaner.py`) — rejects code that is mostly non-English text. Not committed, dataset not rebuilt with it.

---

## 3. State of each part

| Part | Works | Problems / unknowns |
|------|-------|---------------------|
| **Data** | Pipeline runs; 5,508 / 612 examples | 99% GitHub functions, ~50% "Write a Python function…"; only 0.5% debug, 0.5% refactor, 0.2% explain examples; some Chinese instructions. Cleaner v4 not applied — it would drop 266 of 5,508 training rows (~5%) |
| **Model** | `model/lora/` = finished Colab adapter (same file as `model_t4/lora/`); knows simple answers | Never tested against the 612 validation examples; can't stay on track (see test results below) |
| **Chat engine** | Intent routing, 5 modes; answers in 10–30 s; explain mode works; clean simple code answers | Debug and refactor don't really do the task (training data gap); history turned off (see section 5); model can't stop on its own (trained without an end marker); no memory |
| **RAG** | Index built 2026-09-25: 5,508 dataset + 160 codebase chunks | **Turned off** (see section 5). Weak matches — "quicksort" returns merge sort and unrelated code; many dataset instructions are auto-generated ("Write a function named `setup` that takes self") |
| **Server** | FastAPI with `/health`, `/generate`, `/chat` | — |
| **Extension** | Compiles; `node_modules` present | Only the basic generate commands; no chat panel yet |
| **Docs** | `.github/CLAUDE.md` updated 2026-09-25 | `README.md` not reviewed for accuracy |

### Environment (checked 2026-09-25)

- Python 3.13 (`C:\Python313`)
- torch 2.7.1 + CUDA 11.8 — **GPU detected**
- transformers 4.57.1, peft 0.19.1, faiss, sentence-transformers 5.4.1, fastapi 0.136.1 — all installed

---

### Test 1 — before fixes (2026-09-25, GTX 1650, RAG on)

Model load: 53.5 s. Real generation speed: ~2.8 tokens/s. GPU only ~31% busy.

| Question | Mode | Time | Result |
|----------|------|------|--------|
| Write a palindrome function | generate | 189 s | Right in 2 lines, then keeps going with junk and misspelled names (`is_palindrom`, `is_PALINODES`) |
| Fix `5 + '3'` TypeError | debug | 183 s | Wrong — unrelated `ratio()` function and tests, likely copied from search results |
| Explain a decorator | explain | 203 s | Useless — one stray line of logging code |
| Improve a loop | refactor | 186 s | Wrong — repeats the `ratio()` function from the previous answer |
| Hello, who are you? | chat | 27 s | Fine, generic |
| What did I ask first? | chat | 219 s | Made up ("your favorite music") — no memory in chat mode |

Causes found:
1. **Never stops early** — generation always runs to the 512-token limit; junk is cut off afterwards (`generator.py` `_apply_stop_words`). Most of each ~3 min is wasted. The one answer that stopped on its own took 27 s.
2. **Misspelled names** — `no_repeat_ngram_size=4` (`generator.py:100`) bans repeating any 4-token sequence; code must repeat names, so the model is forced to misspell them.
3. **Copies old answers** — generate/debug/refactor prompts include the last 3 turns (`prompt_builder.py:107`); the model copies them instead of answering.
4. **Weak RAG matches** — the model copies whatever search returns, even when unrelated.
5. **No memory in chat mode** — by design (no history); it invents answers. Phase 11 addresses this.
6. **Root cause of 1:** training examples have no end-of-text token, and training uses the end token as padding (which is ignored), so the model never learned to stop.

### Fixes made (2026-09-25)

- `generator.py`: stop words are passed to `model.generate()` so it halts early; per-mode stop words (code modes stop at test/demo code, prose modes stop when code starts, all modes stop when the model echoes its own prompt); `no_repeat_ngram_size` off for code modes.
- `prompt_builder.py`: history no longer injected in any mode (see section 5).
- `config.yaml`: RAG turned off (see section 5).

### Test 2 — after the stop-word and repeat fixes (RAG still on, last 2 questions still injected)

| Question | Before | After | Result |
|----------|--------|-------|--------|
| Palindrome function | 189 s | **31 s** | ✅ Correct and clean |
| Fix `5 + '3'` | 183 s | 110 s | ❌ Answered the *previous* question and pasted a search result |
| Explain a decorator | 203 s | **78 s** | ✅ Good explanation (ends when the model starts writing code) |
| Improve a loop | 186 s | 209 s | ❌ Pasted an unrelated search result (parentheses code), ran to the limit |
| Who are you? | 27 s | 52 s | ⚠️ Fine start, then rambled into a new "Answer:" — now also stopped by stop words |
| What did I ask first? | 219 s | **10 s** | ❌ Still made up — Phase 11 |

Total: ~17 min → ~8 min. The two ❌ code answers led to turning off history and RAG.

### Test 3 — history and RAG off (2026-09-25, other VS Code window closed, 8.6 GB RAM free)

Model load: **21 s** (was 51–53 s). Speed ~3 tokens/s. Total for 6 answers: **~2 min** (test 1: ~17 min).

| Question | Test 1 | Test 3 | Result |
|----------|--------|--------|--------|
| Palindrome function | 189 s | 28 s | ✅ Correct, with a good docstring — but the docstring repeats the prompt line "You are a Python expert…" |
| Fix `5 + '3'` | 183 s | 23 s | ❌ Ignores the error; writes an unrelated sum-of-multiples function |
| Explain a decorator | 203 s | 24 s | ✅ Good explanation — starts with a stray `"""` and ends on "Here's a simple example:" with no example |
| Improve a loop | 186 s | 15 s | ⚠️ Valid code, but just wraps the same loop in a function — no real improvement (should be `[i * i for i in range(10)]`) |
| Who are you? | 27 s | 13 s | ✅ Clean |
| What did I ask first? | 219 s | 11 s | ❌ Made up — Phase 11 |

**What's left is the model, not the plumbing.** The training data is 99% GitHub functions; about half the instructions are "Write a Python function…". Only 27 examples (0.5%) are about fixing errors, 25 (0.5%) about improving code, 11 (0.2%) about explaining. The model never learned to debug or refactor. Some instructions are also in Chinese (cleaner v4 checks only the code, not the instruction).

Small cosmetic issues — **fixed 2026-09-25** in `generator.py` (checked against the test-3 answers): prompt text echoed into docstrings (`_strip_prompt_echo()`), stray `"""` at the start of prose answers, dangling "Here's an example:" line when a prose answer is cut at the code.

## 4. Loose ends (clean up before new work)

1. ~~Build the RAG index~~ — done 2026-09-25.
2. ~~Test the current model~~ — done 2026-09-25, see test results above.
3. **Decide on cleaner v4:** rebuild the dataset and retrain, or commit it and use it next time.
4. **Commit the pending changes:**
   - `data/scripts/cleaner.py` (v4)
   - `.gitignore` (ignores the `Google Colab/` folder — the notebook contains personal Google Drive file listings, keep it out of git)
   - `.github/Copilot_Instructions.md` → `.github/CLAUDE.md` rename + updates
   - this file
5. **Remove duplicate backups:** `model/lora_epoch1_backup/`, `model/lora_epoch1_backup_v2/` and `model/lora_new_dataset_checkpoint50/` all hold the same epoch-1 adapter. Keep one.
6. **Minor:** model weights were committed and then removed in May (commits `83df449`, `e7ddbf4`) — they still sit in git history and make the repo bigger.
7. ~~`retrieval/index/` not gitignored~~ — added to `.gitignore` 2026-09-25. Still open: `sessions/` is not gitignored, and its path is hardcoded in `context_manager.py:8` (Phase 11 replaces it).
8. **Model cache settings (owner to do — needs admin):** `HF_HOME` points to `E:\huggingface Assets` (folder re-created; Phi-2 and bge-small re-downloaded 2026-09-25). The old **system-wide** `TRANSFORMERS_CACHE` setting points to `E:\huggingface Assets\transformers`, which makes loading Phi-2 download it a second time. Remove it: Start → "Edit the system environment variables" → Environment Variables → System variables → `TRANSFORMERS_CACHE` → Delete → OK, then restart VS Code.
9. **Windows console encoding:** `indexer.py` crashes on the `→` character when output goes to a file; run with `PYTHONUTF8=1`.

---

## 5. Turned off / postponed — bring back later

Everything switched off, disabled or put off for later goes here (rule in `.github/CLAUDE.md`). Remove a row only when it is back on or finally dropped.

| What | Since | Why | Bring back when / how |
|------|-------|-----|-----------------------|
| **RAG (code search)** | 2026-09-25 | Weak dataset matches were pasted into answers instead of helping (2 of 3 code answers broken in test 2) | After the dataset is improved (cleaner v4 + better instructions). Maybe add a "strong matches only" cut-off. Switch: `rag.enabled: true` in `configs/config.yaml`. The index must be rebuilt first — it now reads dataset v2 (`paths.dataset`) |
| **Chat history in prompts** (all modes) | 2026-09-25 | Model answered the previous question / copied the previous answer instead of the new one | Phase 11 — replaced by short memory facts. `format_context()` in `prompt_builder.py` is kept for this |
| **Retrain with an end-of-text token** | 2026-09-25 | Model never learned to stop; stop words are only a band-aid | Code built (plan step 6). Comes back when notebook Job F has trained the v2 adapter and Job G shows it beats the old one — then it replaces `model/lora/` |
| **Debug / refactor / explain training examples** | 2026-09-25 | Dataset is 99% "write a function"; model can't debug or refactor (test 3) | In dataset v2 now (2,945 fix / 2,062 improve / 1,800 explain). Comes back with the v2 adapter above |
| **Cleaner v4 dataset rebuild** | 2026-05-04 | Not applied yet; drops 266 of 5,508 rows (~5%) | Probably never needed: dataset v2 takes only the best 800 old examples, through the same English filter. Drop this row once the v2 adapter is in use |
| **2nd training pass** | 2026-09-25 | Owner chose 1 pass first, score, then decide | After Job G: if v2 beats the old adapter and the check-set loss was still falling at the end, continue from `model_v2` for a 2nd pass, with a lower learning rate (5e-5 or less — rule in CLAUDE.md) |
| **Scoring test for fixing / improving code** | 2026-09-25 | MBPP only measures writing functions | After the v2 score: small held-out test of broken functions + their tests (the `bug_fix` generator can make it from rows not used in training) |
| **Phase 11 (long-term memory)** | 2026-09-25 | Better training data + retrain chosen first — memory on top of a model that can't debug/refactor adds little | After the dataset improvement and Colab retrain |
| **Speed work** (llama.cpp / GGUF, streaming) | 2026-09-25 | Memory chosen first | After Phase 11 |
| **Phases 9 and 10** | 2026-09-25 | Phase 11 goes first | After Phase 11 |
| **Colibri** | 2026-09-25 | Needs far more RAM, can't run Phi-2 or our LoRA | Only if hardware grows (≥32 GB RAM, bigger GPU) |
| **Removing `TRANSFORMERS_CACHE`** | 2026-09-25 | Needs admin rights | Owner removes it (steps in section 4, item 8) |

---

## 6. Where we are going

Order: better training data + retrain first, then Phase 11, then 9, 10 and speed work.

### Next — better training data (researched 2026-09-25, nothing downloaded yet)

**Work plan (owner approved, done solo, step by step):**
1. Scoring test — **done on Colab T4 (2026-09-25).** "Before" scores on 100 MBPP problems (greedy, generate mode): **current adapter 51/100** (26 min, ~15 s/problem), **base Phi-2 48/100** (14 min, ~8 s/problem). Per problem: both 39, only adapter 12, only base 9, neither 40 → the old fine-tune gave no real gain (+3 is within noise) and makes answers ~2× slower. Detailed results: Drive `MyDrive/PY-V/results/eval/mbpp_{lora,base}.jsonl`. MBPP only measures writing functions — not fixing/improving code
2. Download + convert sources — **built and sample-checked** (`data/scripts/fetch_sources.py` + `data/scripts/sources/`). Sample of 20 per source reviewed by eye; fixes made after review: Glaive answers that still refer to removed code are dropped, CodeSearchNet capped at 2 per repository (dataset is ordered by repo) and functions full of commented-out code skipped, old GitHub data limited to ≤40 lines with score ≥2.5 (~1,030 examples; the top scores were long, project-bound functions). Pass rates seen: OpenCodeInstruct ~43% (all tests passing), Glaive ~6%, CodeSearchNet ~21%. Full fetch: **old GitHub done locally (1,032 records → `data/raw/v2/old_github.jsonl`)**; the 4 online sources run on Colab (runner notebook, Job A). **Done on Colab (2026-09-25):** OpenCodeInstruct **3,000** (6,227 scanned), self-oss-instruct **2,000**, Glaive **1,500**, CodeSearchNet **1,200** — all on Drive `MyDrive/PY-V/results/data_v2/`. With old GitHub (1,032, laptop) that is **8,732 raw records** before mixing. VS Code stopped showing the cell's output after the first two sources (the Colab extension's keep-alive pings were failing), which looked like a hang; the job had in fact finished all four (the per-source reject stats for Glaive/CodeSearchNet were lost with that output). An exit safeguard was added to `fetch_sources.py` anyway. The trained adapter is at `MyDrive/Py-V/Py-V/model_t4/lora` (fingerprint matches the laptop); notebook cell 1d finds it automatically.

**Where jobs run:** heavy jobs (full fetch, scoring test, training) run on Colab through `Google Colab/py_v_runner.ipynb` (gitignored). The notebook downloads the code from GitHub — **push changes first** — and saves results to Drive `MyDrive/PY-V/results/`. Claude writes the notebook; the owner runs it (VS Code Colab kernel or browser) and saves it, so the printed results can be read back.
3. Break-and-fix script for fix-error examples — **built** (`data/scripts/sources/bug_fix.py` + `mutations.py`, run as source `bug_fix`), tested locally on a hand-written function only (9 bug kinds; bugs the tests don't catch are skipped). **Done on Colab (Job C, ~12 min): 3,000 records** (14,665 rows scanned). Bug kinds: missing return 475, name typo 474, compare 429, None init 429, operator 426, off-by-one 313, and/or 162, flipped bool 147, missing cast 145. User sees a failing check (with wrong value) in 1,646, an error message in 1,354. Only 44 functions had no bug the tests caught; 84 originals failed in our runner
4. Improve-code examples — **built**, two sources because real refactor commits are rare (~0.3% of CommitPackFT): `commitpack_refactor` (real commits, ~150–200 expected; sample of 20 reviewed, filters tightened after review; runs nothing) + `improve_synthetic` (clean tested functions rewritten into clumsy-but-equivalent code, tests must still pass; target 2,000; Colab only). Checked locally on hand-written functions. **Done on Colab (Job D, ~45 min):** `improve_synthetic` **2,000** (97,248 rows scanned — only ~3% qualify: 18,118 had fewer than 2 safe rewrites, 6,760 too long, 606 originals failed here); rewrites used: truthiness→len 871, `x += 1`→`x = x + 1` 864, comprehension→loop 760, bool return→if 538, sum→loop 410, ternary→if 371, enumerate→range 277, max/min→if 167. `commitpack_refactor` **202** (all 56,025 commits scanned, 98% not refactors) — as expected; the mix takes what there is, so the improve share is ~19% instead of 20%
5. Mix, clean, dedupe, split to ~12k — **built** (`data/scripts/build_dataset_v2.py` + `decontaminate.py`). Drops duplicates, anything overlapping MBPP/HumanEval (keeps the scoring test honest) and records over 700 tokens; takes each source's share (config `dataset_v2.build.take`); 5% val per task. Local test on 1,138 records: caught all planted duplicates and the planted MBPP copy, plus 18 real duplicates and 1 real MBPP overlap in the old data; median 306 tokens, 95% under 530. **Done on Colab (Job E, 2026-09-25): 11,607 records** (11,027 train + 580 val) → Drive `MyDrive/PY-V/results/dataset_v2/`. Mix: write 41% (4,800) / fix errors 25% (2,945) / improve 18% (2,062) / explain 16% (1,800) — close to plan. Median 339 tokens, 95% under 581. Dropped: 160 overlapping MBPP/HumanEval (127 of them OpenCodeInstruct), 453 too long (OpenCodeInstruct 276, Glaive 122), 199 duplicates (115 in improve_synthetic — OpenCodeInstruct repeats the same small functions). Short vs plan: bug_fix 2,945/3,000, improve_synthetic 1,865/2,000, commitpack 197/400. `old_github.jsonl` rebuilt on Colab from the v1 `train.jsonl` on Drive (fingerprint matched, same 1,032 records)
6. Training fixes — **built (2026-09-25)**, checked locally with the tokenizer only (no model loaded). Owner's choices: start **fresh from plain Phi-2**, **normal strength** (learning rate 2e-4, 4× the v1 run), **1 pass** then score and decide, **learn from answers only**. What changed:
   - every example ends with the end-of-text token, and labels are padded with -100, so the end token is no longer hidden (old cause: pad == eos + `DataCollatorForLanguageModeling`)
   - training uses the same per-mode prompt as the app (`build_training_prompt(mode, ...)` = `build_prompt(...)`, mode from `metadata.task`); the `/generate` endpoint now uses the generate template too
   - loss on the answer only (prompt tokens -100)
   - max_seq_length 384 → 768; longer examples are dropped, never cut (template adds ≤62 tokens, so v2's ≤700-token records all fit)
   - eval loss now actually computed (`eval_strategy="steps"` + `label_names`; the v1 run recorded none), every 100 steps
   - T4 trainer: settings from config, `--output-dir` with automatic resume from the newest checkpoint there (old code looked in the wrong folder), ETA print fixed (was 10× too high)
   - `eval_mbpp.py --adapter DIR` to score any adapter; training data path now `data/datasets/v2/`
7. Retrain on Colab, re-score against step 1 — **ready, not run.** Notebook **Job F** trains → Drive `MyDrive/PY-V/model_v2/lora` (~690 steps, rough guess 2–3 h, checkpoint every 50 steps, resumes after a disconnect); **Job G** scores it on the same 100 MBPP problems → `mbpp_lora_v2.jsonl`. Needs Phase 8.1.12 pushed + 1b re-run. The new adapter replaces `model/lora/` only if it beats the old one (51/100)

**Decided:** ~12,000 clean examples, mix ≈ 40% write / 25% fix errors / 20% improve code / 15% explain, using the sources below; old GitHub data trimmed to its best-scored part. Every code sample run-checked; English only; deduped.

| Source (Hugging Face) | Rows | License | Use for | Notes |
|---|---|---|---|---|
| `nvidia/OpenCodeInstruct` | 1.4M+ | CC-BY-4.0 (credit NVIDIA) | Write code | Each row has unit tests + pass rate → keep only rows that pass all tests |
| `bigcode/self-oss-instruct-sc2-exec-filter-50k` | 50,661 | ODC-By | Write code | Python, already execution-checked; answers include reasoning + tests (trim) |
| `glaiveai/glaive-code-assistant` | 136,109 | Apache-2.0 | Explain, Q&A, some debugging | Real-user-style questions; ~60% Python → filter |
| `code-search-net/code_search_net` (python) | 457,461 | Mixed (original repos) | Explain | Function + its docstring → "explain this code" pairs |
| `bigcode/commitpackft` (python) | ~136 MB file | MIT | Improve / fix code | Real commits (before, after, message); noisy (reverts, test edits) → filter by message |
| **Own generator** (break-and-fix) | unlimited | ours | Fix errors | Break working functions on purpose, run to get the real error, pair with the fix |
| `google-research-datasets/mbpp` (1,401) + `openai/openai_humaneval` (164) | — | CC-BY-4.0 / MIT | **Testing only** | Never train on these — used to score the model |

Licensing to-do: if Py-V or its adapter is ever shared, credit NVIDIA (OpenCodeInstruct, CC-BY-4.0) and BigCode (self-oss-instruct, ODC-By) in the README.

Rejected: Magicoder-OSS-Instruct, CodeAlpaca, CodeFeedback's Evol part (made with OpenAI models — their terms restrict training other models on outputs); HF "bug fixing" sets (tiny, unknown quality, or Java); `vikp/code_with_explanations` (notebook text, not Q&A pairs).

### Phase 9 — Multi-language adapters (planned)
One LoRA adapter per language (Python first, then JS/TS), swapped at runtime on the same base model. Rules in `.github/CLAUDE.md`.

### Phase 10 — VS Code chat panel (planned)
Copilot-style chat panel inside VS Code with streaming answers (words appear as they are generated). Rules in `.github/CLAUDE.md`.

### Phase 11 — Long-term memory (next up — before 9, 10 and speed work)

**Idea:** Py-V keeps a notebook of every chat. It writes down important facts, looks things up before answering, always trusts the newest fact, and lets you erase pages.

**What it improves:** memory — Phi-2 can only see ~2,048 tokens (~1,500 words) at once; the database brings back facts from older messages and older chats.
**What it does not improve:** reasoning (it gives facts, not a smarter model) or speed (extra prompt text makes answers slightly slower).

| Question | Decision |
|----------|----------|
| How long does it remember? | Across all chats |
| What does it save? | Every message + tagged key facts (file names, errors, versions, decisions, "remember this") |
| How does it search? | Both exact words (SQLite FTS5) and similar meaning (existing embedding model) |
| When is memory used? | All modes get short facts; code from memory only when writing, fixing or improving code |
| Old or wrong facts? | Newest fact wins; you can list and delete memories |
| Where is it stored? | SQLite — one file on the laptop, nothing to install |

Planned files and rules: see "Phase 11 Rules" in `.github/CLAUDE.md`.

### Later — Speed work (not yet a phase)
Options, all need measuring on this laptop first:
- **Streaming answers** — same total time, feels much faster (part of Phase 10).
- **Shorter prompts / lower answer length limit.**
- **Faster engine:** merge the LoRA into Phi-2, convert to GGUF, run with llama.cpp or Ollama — often several times faster than transformers + bitsandbytes 4-bit on small GPUs, and keeps the fine-tune.

---

## 7. Decisions log

| Date | Decision | Why |
|------|----------|-----|
| 2026-09-25 | Assistant works solo — no subagents or multi-agent workflows unless the owner explicitly asks | Owner's preference |
| 2026-09-25 | **Colibri** (github.com/Alexie1171/colibri) not adopted for now | Built for huge mixture-of-experts models (35B–2.8T). Cannot run Phi-2, cannot load our LoRA adapters, inference only (no training), recommends ~30 GB RAM for its smallest coding model. Its own docs measure 0.35 tok/s CPU-only for Qwen3.6 and show Ollama beating it on an 8 GB card. Revisit only if hardware grows (≥32 GB RAM, bigger GPU) and we want a much larger model |
| 2026-09-25 | Long-term memory added as **Phase 11** (not renumbering 9 and 10) | Keeps existing phase numbers stable |
| 2026-09-25 | Memory before speed work | Owner's priority |
| 2026-09-25 | Phase 11 built **before** Phases 9 and 10 | Best value for effort: reuses the existing embedder and RAG search code, needs no new training, runs on CPU. Phase 9 needs new scraping + Colab training per language; Phase 10 is mostly UI and can later show memory list/delete |
| 2026-09-25 | Fix generation bugs before Phase 11 | Answers took ~3 min and were unusable; couldn't judge the model |
| 2026-09-25 | `retrieval/index/` kept out of git | 15 MB, rebuildable, goes stale |
| 2026-09-25 | Chat history turned off in all modes until Phase 11 | Model answered old questions instead of the new one |
| 2026-09-25 | RAG turned off until the dataset is better | Weak matches were copied into answers |
| 2026-09-25 | Everything turned off or postponed must be listed in section 5 | So nothing is forgotten |
| 2026-09-25 | Better training data + retrain **before** Phase 11 | Test 3: model is fast but can't debug or refactor — dataset is 99% "write a function". Every feature depends on answer quality |
| 2026-09-25 | Fix the small answer clean-up issues now | Prompt text in docstrings, stray `"""`, dangling "Here's an example:" |
| 2026-09-25 | New dataset: use the recommended source mix (section 6) | Covers all 4 tasks with checked, permissively licensed data |
| 2026-09-25 | Old 5,508 GitHub examples: keep only the best-scored part | Many have weak auto-generated instructions |
| 2026-09-25 | New dataset size: ~12,000 examples | Enough for Phi-2's 4 tasks; should fit one Colab session |
| 2026-09-25 | Heavy jobs run on Colab via a runner notebook; code comes from GitHub, results go to Drive `MyDrive/PY-V/results/` | Laptop has 16 GB RAM and is often in use; Colab T4 is faster; owner has fast internet |
| 2026-09-25 | New training starts **fresh from plain Phi-2**, not from the old adapter | Old adapter gave no real gain (51 vs 48) and learned the never-stop habit |
| 2026-09-25 | Learning rate 2e-4 ("normal strength") | Standard for LoRA; the v1 run's 5e-5 barely changed the model |
| 2026-09-25 | 1 training pass, score, then decide on a 2nd | Early result; a 2nd pass can continue from the first |
| 2026-09-25 | Loss on answers only | Standard for question → answer training; all learning goes into the answers |
| 2026-09-25 | New adapter goes to a new Drive folder (`model_v2/lora`) and replaces `model/lora/` only if it beats the old one on MBPP | The old adapter is never at risk |

---

## 8. Constraints

### Hardware (this laptop)

| | |
|---|---|
| GPU | NVIDIA GTX 1650, 4 GB VRAM |
| CPU | AMD Ryzen 7 3750H, 4 cores / 8 threads |
| RAM | 16 GB |
| Storage | Intel 660p 512 GB NVMe (QLC — slow for heavy disk streaming) + SanDisk 240 GB SATA SSD; ~72 GB free on D: |
| Big training | Google Colab T4 (local GPU too slow) |

**RAM is tight.** Loading Phi-2 needs several GB of RAM for a short time. With two VS Code windows (~4.4 GB), the PHP language server of the other project (~1.5 GB) and Claude Code (~1.1 GB) open, Windows pushed ~5 GB to the page file on 2026-09-25 and the laptop lagged afterwards. Close the other VS Code window before running Py-V.

### Model rules
- Phi-2 base, always 4-bit (BitsAndBytes)
- LoRA / PEFT only — no training from scratch, no full fine-tuning
- No models over 7B parameters
- Always resume from checkpoint when one exists; back up `model/lora/` before training
- Second-epoch learning rate 5e-5 or lower

### Architecture rules (short version — full list in `.github/CLAUDE.md`)
- Every path and setting comes from `configs/config.yaml` via `CFG` — no hardcoded paths
- One job per file; data / training / inference never mixed
- Prompt format lives only in `prompt_builder.py`; templates only in `prompt_templates.py`
- One shared model loader; the model loads once at server startup
- RAG only for `generate` / `debug` / `refactor`; never for `explain` / `chat`
- No history or code injected into `explain` / `chat` prompts (it made the model spit code)
- Extension is TypeScript only; routes stay thin
- Memory (Phase 11): SQLite only, reuse the existing embedder, never commit the database

### Working rules for the AI assistant
- Do the work directly; ask before any subagent or workflow use
- Explain things and ask decision questions in simple terms
- Don't act when the owner asks for decision help only
- Anything turned off or postponed is added to section 5 in the same step

---

## 9. Open questions

1. **Is the Colab model better than epoch 1?** Needs a side-by-side test (less important now — the v2 adapter replaces both if it wins).
2. **Order after Phase 11:** Phase 9, Phase 10 or speed work next?

---

## 10. Handy commands

```bash
# Config check
python -c "from model.training.config_loader import CFG; print(CFG.model.name, CFG.paths.dataset)"

# Build RAG index (after any dataset or code change)
python -m retrieval.indexer

# Test retrieval / model / chat
python -m retrieval.test_rag
python -m experiments.test_phi2
python test_chat.py

# Start the server
uvicorn inference.api.main:app --host 0.0.0.0 --port 8000

# Extension
cd extension && npm install && npm run compile   # then F5 in VS Code
```
