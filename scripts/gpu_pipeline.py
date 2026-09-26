"""
gpu_pipeline.py — PY-V (scripts/)
One-button GPU run: every big job for the Granite brain upgrade. Started by
the RUN EVERYTHING cell of Kaggle/py_v_kaggle.ipynb (background run on
GPU T4 x2) or of Google Colab/py_v_runner.ipynb (one T4, results on Drive).

  1-2  test both untrained Granite versions (chat in its own format, plain)
  10   test the chat version with the Granite family's text-splitting rule
       (its own tokenizer.json has GPT-2's rule — see CHAT_SPLIT_FIX)
  3    training data: every source file (uploaded, from an earlier run, or
       re-made if missing) + the long-file fix examples
  4    rebuild the training set (dataset v3 = v2 + long-file examples)
  5-6  train the chat version (own chat format, with the splitting rule of
       1 or 10 that passed more test questions), then test it
  7-8  train the plain version (V's template), then test it
  9    extra: the chat version with V's template (untrained)

Every test = MBPP (write code) + long-file bug fixes + chat + fix/improve.

Lanes run at the same time, each job seeing only its own GPU (CUDA_VISIBLE_DEVICES):
  data lane, CPU:     3 → 4 (the trainings wait for it)
  two GPUs (Kaggle):  chat lane on GPU 0: 1 → 10 → 5 → 6   plain lane on GPU 1: 2 → 7 → 8
  one GPU (Colab):    1 → 2 → 10 → 5 → 6 → 7 → 8
Stage 9 is done by whichever GPU lane is free first (also while one waits for the data).
scripts/pipeline_redo.json (in git) lists stages to run once more because their
saved results are wrong; each entry is applied once per --root.

Everything is saved under --root: results/ (report, log, data, test answers)
and model_<brain>/lora/ (adapters + checkpoints); the project's output folders
are linked there. A stage whose results are already there is skipped and
training resumes from its last checkpoint, so running again continues where a
run stopped. --inputs (Kaggle: /kaggle/input) is searched for an earlier run's
saved folder (copied into --root first) and for uploaded source files.
--stop-after stops the jobs before a hard session limit (Kaggle: 12 h a run),
so the run ends normally and everything done so far is saved. After every
stage the report is rewritten: {root}/results/PIPELINE_REPORT.md (+ .json);
everything printed is also appended to {root}/results/pipeline_log.txt.

Usage (from repo root):
    python -m scripts.gpu_pipeline --root /kaggle/working/PY-V --inputs /kaggle/input --stop-after 11
    python -m scripts.gpu_pipeline --root /content/drive/MyDrive/PY-V
"""

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.training.config_loader import CFG
from experiments.eval_common import result_tag

BRAINS = {
    "chat":  {"model": "ibm-granite/granite-4.2-3b",      "format": "native_chat"},
    "plain": {"model": "ibm-granite/granite-4.1-3b-base", "format": "template"},
}
# The chat version's own tokenizer.json splits text with GPT-2's rule (probably
# saved with the transformers 5.0 bug); same merges as the plain version, whose
# file has the Granite family's rule. Stage 10 tests the chat version with that
# rule; stage 5 trains it with whichever rule passed more test questions.
CHAT_SPLIT_FIX = BRAINS["plain"]["model"]
TESTS = [("mbpp", "eval_mbpp"), ("longctx", "eval_long_context"), ("chat", "eval_chat"), ("fix", "eval_fix")]
STAGES = {
    1:  "Test Granite chat (untrained, own chat format)",
    2:  "Test Granite plain (untrained)",
    3:  "Training data (missing sources + long-file fix examples)",
    4:  "Build training set v3",
    5:  "Train Granite chat (own chat format, better splitting rule of 1 / 10)",
    6:  "Test trained Granite chat",
    7:  "Train Granite plain (V's template)",
    8:  "Test trained Granite plain",
    9:  "Extra: test Granite chat with V's template (untrained)",
    10: "Test Granite chat with Granite's splitting rule (untrained, own chat format)",
}
TEST_STAGES = {   # stage → (brain, trained, native chat format, splitting rules from)
    1: ("chat", False, True, None),  2: ("plain", False, False, None), 6: ("chat", True, True, None),
    8: ("plain", True, False, None), 9: ("chat", False, False, None),  10: ("chat", False, True, CHAT_SPLIT_FIX),
}
TRAIN_STAGES = {5: "chat", 7: "plain"}
# One-time re-runs (in git): stages whose saved results are wrong. Removed once per
# root, then they run like never done; done ids are kept in results/redo_done.json
REDO_FILE = Path(__file__).with_name("pipeline_redo.json")
OUTPUTS  = CFG.evaluation.output_dir
DATA_DIR = CFG.dataset_v2.output_dir
LINKS    = {DATA_DIR: "results/data_v2",                             # project folder → saved folder under root
            CFG.dataset_v2.build["output_dir"]: "results/dataset_v3",
            OUTPUTS: "results/eval"}
V1_MD5  = "4028f2d3a010abf0aa36d52c498031e9"   # the v1 dataset (source of old_github) — laptop data/datasets/train.jsonl
ENV     = {**os.environ, "PYTHONUNBUFFERED": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}   # less fragmentation → more of the T4 usable
BAR       = re.compile(r"\d+%\|")   # a tqdm progress-bar line
BAR_EVERY = 60                      # seconds between progress-bar lines per lane


@dataclasses.dataclass
class Lane:
    name: str             # shown before every line it prints
    gpu:  str             # CUDA_VISIBLE_DEVICES for its jobs; "" = CPU only
    last_bar: float = 0.0

    @property
    def where(self) -> str:
        return f"GPU {self.gpu}" if self.gpu else "CPU"


def count_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return 0
    return sum(line.startswith("GPU ") for line in out.splitlines())


def find(bases: list, name: str, depth: int = 6) -> list:
    """Files called `name` at most `depth` folders below any of `bases`."""
    found = []
    for base in bases:
        for k in range(depth + 1):
            found += sorted(Path(base).glob("*/" * k + name))
    return found


class Pipeline:
    def __init__(self, root: Path, inputs: list, gpus: int, stop_after: float = None):
        self.root        = root
        self.results     = root / "results"
        self.log         = self.results / "pipeline_log.txt"
        self.inputs      = inputs
        self.gpus        = gpus
        self.deadline    = time.time() + stop_after * 3600 if stop_after else None
        self.out_of_time = False
        self.status      = {n: ("waiting", "", "") for n in STAGES}   # n → (status, where, minutes)
        self.extra       = [9]                  # done by whichever GPU lane is free first
        self.jobs        = self._jobs()
        self.lock        = threading.Lock()     # console + log
        self.report_lock = threading.Lock()
        self.logf        = None

    # ─── output ───────────────────────────────────────────────────────────────

    def write(self, text: str, lane: Lane = None, stamp: bool = False):
        if lane is not None:
            text = "\n".join(f"[{lane.name}] {line}" for line in text.split("\n"))
        with self.lock:
            print(text, flush=True)
            if self.logf is None:
                self.results.mkdir(parents=True, exist_ok=True)
                self.logf = open(self.log, "a", encoding="utf-8", buffering=1)
            self.logf.write((f"[{datetime.datetime.now():%H:%M:%S}] " if stamp else "") + text + "\n")

    def say(self, text: str, lane: Lane = None):
        self.write(text, lane, stamp=True)

    def echo(self, lane: Lane, line: str):
        """A job's output line — progress bars at most once a minute per lane
        (they would bury everything else, especially with lanes side by side)."""
        if BAR.search(line) and "100%" not in line:
            now = time.time()
            if now - lane.last_bar < BAR_EVERY:
                return
            lane.last_bar = now
        self.write(line, lane)

    # ─── running jobs ─────────────────────────────────────────────────────────

    def run(self, lane: Lane, args: list, timeout_h: float) -> bool:
        """Run a module (python -m ...) on the lane's GPU, streaming its output.
        True if it succeeded. Killed after timeout_h, or at --stop-after."""
        limit = timeout_h * 3600
        if self.deadline:
            limit = min(limit, self.deadline - time.time())
        if limit <= 0:
            self.out_of_time = True
            return False
        self.say("$ python -m " + " ".join(args), lane)
        proc = subprocess.Popen([sys.executable, "-m", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1,
                                env={**ENV, "CUDA_VISIBLE_DEVICES": lane.gpu})
        killed = threading.Event()

        def kill():
            killed.set()
            proc.kill()

        timer = threading.Timer(limit, kill)
        timer.daemon = True
        timer.start()
        try:
            for line in proc.stdout:
                self.echo(lane, line.rstrip("\n"))
            code = proc.wait()
        finally:
            timer.cancel()
        if killed.is_set():
            if self.deadline and time.time() >= self.deadline - 5:
                self.out_of_time = True
                self.say("STOPPED - run time limit (--stop-after) reached; run again to continue", lane)
            else:
                self.say(f"TIMEOUT after {timeout_h} h", lane)
            return False
        return code == 0

    def stage(self, n: int, lane: Lane):
        """Run one stage unless its results already exist; record what happened."""
        name  = f"{n}. {STAGES[n]}"
        done, work, needs = self.jobs[n]
        start = time.time()
        try:
            if self.out_of_time:
                status = "not started (run time limit) - run again"
            elif needs is not None and not needs():
                status = "blocked (an earlier stage is missing)"
            elif done():
                status = "done earlier - skipped"
            else:
                self.record(n, "running", lane, start)
                self.say(f"\n{'=' * 70}\n=== {name}  [{lane.where}]\n{'=' * 70}", lane)
                ok = work(lane)
                if ok and done():
                    status = "done"
                elif self.out_of_time:
                    status = "stopped (run time limit) - run again to continue"
                else:
                    status = "FAILED"
        except Exception:
            self.say(traceback.format_exc(), lane)
            status = "FAILED (pipeline error)"
        self.record(n, status, lane, start)
        self.say(f"--- {name}: {status} ({(time.time() - start) / 60:.0f} min)", lane)

    def record(self, n: int, status: str, lane: Lane, start: float):
        self.status[n] = (status, lane.where, round((time.time() - start) / 60, 1))
        self.write_report()

    def next_extra(self):
        with self.lock:
            return self.extra.pop(0) if self.extra else None

    # ─── the jobs ─────────────────────────────────────────────────────────────

    def _jobs(self) -> dict:
        """stage → (done, work(lane), needs)"""
        chat_tested = self.tests_done(*TEST_STAGES[1])
        jobs = {n: (self.tests_done(*spec), self.run_tests(*spec),
                    self.trained(spec[0]) if spec[1] else None) for n, spec in TEST_STAGES.items()}
        jobs.update({
            3: (self.data_done, self.make_data, None),
            4: (self.dataset_done, lambda lane: self.run(lane, ["data.scripts.build_dataset_v2"], timeout_h=1),
                self.data_done),
            5: (self.trained("chat"), self.train("chat"), lambda: self.dataset_done() and chat_tested()),
            7: (self.trained("plain"), self.train("plain"), self.dataset_done),
        })
        return jobs

    def adapter_dir(self, brain: str) -> Path:
        return self.root / f"model_{BRAINS[brain]['model'].split('/')[-1]}" / "lora"

    def test_args(self, brain: str, trained: bool, native: bool, split: str = None) -> list:
        args = ["--model", BRAINS[brain]["model"]]
        if trained:
            args += ["--adapter", str(self.adapter_dir(brain))]
        else:
            args += ["--base"] + (["--native-chat"] if native else []) + (["--split-rules-from", split] if split else [])
        return args

    def test_tag(self, brain: str, trained: bool, native: bool, split: str = None) -> str:
        args = argparse.Namespace(base=not trained, model=BRAINS[brain]["model"], adapter=str(self.adapter_dir(brain)),
                                  native_chat=native and not trained, split_rules_from=split)
        return result_tag(args)

    def tests_done(self, brain, trained, native, split=None):
        tag = self.test_tag(brain, trained, native, split)
        return lambda: all((OUTPUTS / f"{prefix}_{tag}_summary.json").exists() for prefix, _ in TESTS)

    def run_tests(self, brain, trained, native, split=None):
        """All four tests with one model load (eval_all), skipping tests already saved."""
        return lambda lane: self.run(lane, ["experiments.eval_all", *self.test_args(brain, trained, native, split),
                                            "--skip-done"], timeout_h=4)

    def trained(self, brain):
        return lambda: all((self.adapter_dir(brain) / f).exists()
                           for f in ("adapter_model.safetensors", "v_adapter.json"))

    def train(self, brain):
        b = BRAINS[brain]

        def work(lane):
            split = self.chat_split_rules(lane) if brain == "chat" else None
            return self.run(lane, ["model.training.train_lora_t4", "--model", b["model"], "--prompt-format", b["format"],
                                   *(["--split-rules-from", split] if split else []),
                                   "--output-dir", str(self.adapter_dir(brain))], timeout_h=8)
        return work

    def chat_split_rules(self, lane: Lane):
        """Splitting rules for training the chat version: Granite's rule (stage 10)
        only if it passed more test questions than its own file (stage 1)."""
        own   = self.total_passed(self.test_tag(*TEST_STAGES[1]))
        fixed = self.total_passed(self.test_tag(*TEST_STAGES[10]))
        use   = CHAT_SPLIT_FIX if fixed is not None and own is not None and fixed > own else None
        self.say(f"Chat splitting rule: own file {own} vs Granite's rule {fixed} test questions passed -> "
                 f"training with {'Granite' if use else 'its own'}'s rule", lane)
        return use

    def summaries(self, tag: str) -> dict:
        """The saved test summaries of one setup: test prefix → summary."""
        got = {}
        for prefix, _ in TESTS:
            path = OUTPUTS / f"{prefix}_{tag}_summary.json"
            if path.exists():
                got[prefix] = json.load(open(path, encoding="utf-8"))
        return got

    def total_passed(self, tag: str):
        """Questions passed across all four tests (MBPP + long-file + chat + fix + improve), None if any is missing."""
        got = self.summaries(tag)
        if len(got) < len(TESTS):
            return None
        return (got["mbpp"]["score"] + got["longctx"]["passed"] + got["chat"]["passed"]
                + got["fix"]["fix"]["passed"] + got["fix"]["improve"]["passed"])

    def _missing_sources(self) -> list:
        """Sources the training set mixes whose file is not there yet
        (fetch_sources writes a file only when it is complete)."""
        return [n for n in CFG.dataset_v2.build["take"] if not (DATA_DIR / f"{n}.jsonl").exists()]

    def data_done(self):
        return not self._missing_sources()

    def make_data(self, lane: Lane) -> bool:
        self._import_sources(lane)
        ok = True
        for name in self._missing_sources():
            if name == "old_github" and not self._restore_v1(lane):
                ok = False
                continue
            ok = self.run(lane, ["data.scripts.fetch_sources", "--only", name], timeout_h=3) and ok
        return ok

    def _import_sources(self, lane: Lane):
        """Source files uploaded to --inputs (Kaggle: a private dataset made from
        Drive results/data_v2) — the same data as before, no time spent re-making it."""
        for name in self._missing_sources():
            found = find(self.inputs, f"{name}.jsonl")
            if found:
                target = DATA_DIR / f"{name}.jsonl"
                part   = target.with_name(target.name + ".part")   # a source file that exists is complete
                shutil.copy(found[0], part)
                part.replace(target)
                self.say(f"{name}: copied from {found[0]}", lane)

    def _restore_v1(self, lane: Lane) -> bool:
        """The v1 dataset is not on GitHub: find it by fingerprint in --inputs or on Drive."""
        target = Path(CFG.dataset_v2.sources["old_github"]["path"])
        if target.exists():
            return True
        found = find(self.inputs, "train.jsonl")
        for depth in ("*", "*/*", "*/*/*"):         # Colab: anywhere on Drive as <folder>/data/datasets/train.jsonl
            found += sorted(self.root.parent.glob(f"{depth}/data/datasets/train.jsonl"))
        for path in found:
            if hashlib.md5(path.read_bytes()).hexdigest() == V1_MD5:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, target)
                self.say(f"v1 dataset copied from {path}", lane)
                return True
        self.say("v1 dataset NOT FOUND - add the laptop file D:/Py-V/data/raw/v2/old_github.jsonl to the Kaggle "
                 f"data upload (Colab: upload it to {self.results / 'data_v2'}) and run again", lane)
        return False

    def dataset_done(self):
        report = CFG.dataset_v2.build["output_dir"] / "build_report.json"
        return report.exists() and "long_file_fix" in json.load(open(report, encoding="utf-8"))

    # ─── saved folders ────────────────────────────────────────────────────────

    def seed_from_inputs(self) -> list:
        """Every Kaggle run starts empty: copy an earlier run's saved folder (its
        output attached as input) into root, keeping files already there."""
        notes = []
        for report in find(self.inputs, "PIPELINE_REPORT.json"):
            earlier = report.parent.parent
            if report.parent.name != "results" or earlier.resolve() == self.root.resolve():
                continue
            copied = 0
            for src in sorted(earlier.rglob("*")):
                dst = self.root / src.relative_to(earlier)
                if src.is_file() and not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    copied += 1
            notes.append(f"Earlier run found in {earlier}: {copied} files copied into {self.root}")
        return notes

    def apply_redo(self):
        """One-time re-runs listed in scripts/pipeline_redo.json: the saved results
        of those stages are removed (a trained adapter is moved aside), once per root."""
        if not REDO_FILE.exists():
            return
        done_file = self.results / "redo_done.json"
        done      = json.load(open(done_file, encoding="utf-8")) if done_file.exists() else []
        for redo_id, entry in json.load(open(REDO_FILE, encoding="utf-8")).items():
            if redo_id in done:
                continue
            for n in entry["stages"]:
                self.say(f"redo {redo_id}: stage {n} - {self.clear_stage(n, redo_id)} ({entry['why']})")
            done.append(redo_id)
            with open(done_file, "w", encoding="utf-8") as f:
                json.dump(done, f, indent=1)

    def clear_stage(self, n: int, redo_id: str) -> str:
        if n in TEST_STAGES:
            tag   = self.test_tag(*TEST_STAGES[n])
            files = [OUTPUTS / f"{prefix}_{tag}{end}" for prefix, _ in TESTS for end in (".jsonl", "_summary.json")]
        elif n in TRAIN_STAGES:
            folder = self.adapter_dir(TRAIN_STAGES[n])
            if not folder.exists():
                return "nothing saved yet"
            folder.rename(folder.with_name(f"lora_before_{redo_id}"))
            return f"adapter moved to {folder.with_name(f'lora_before_{redo_id}')}"
        elif n == 3:
            files = [DATA_DIR / "long_file_fix.jsonl"]
        elif n == 4:
            files = [CFG.dataset_v2.build["output_dir"] / name for name in ("build_report.json", "train.jsonl", "val.jsonl")]
        else:
            raise ValueError(f"stage {n} cannot be redone")
        removed = [path for path in files if path.exists()]
        for path in removed:
            path.unlink()
        return f"{len(removed)} saved files removed"

    def link_outputs(self):
        """Point the project's output folders into root, so everything the jobs
        write is saved there (Drive on Colab, the run's output on Kaggle)."""
        for project, saved in LINKS.items():
            project, target = Path(project), self.root / saved
            target.mkdir(parents=True, exist_ok=True)
            if project.is_symlink():
                if project.resolve() == target.resolve():
                    continue
                project.unlink()                            # linked to another root before
            elif project.is_dir() and not any(project.iterdir()):
                project.rmdir()
            elif project.exists():
                raise SystemExit(f"{project} is a real folder - what the jobs write there would not be saved "
                                 f"under {self.root}. Move it away and run again.")
            project.parent.mkdir(parents=True, exist_ok=True)
            project.symlink_to(target.resolve(), target_is_directory=True)
            self.say(f"linked {project} -> {target}")

    # ─── report ───────────────────────────────────────────────────────────────

    def write_report(self):
        with self.report_lock:
            self._write_report()

    def _write_report(self):
        rows = [("Granite chat — untrained, own chat format",                   1),
                ("Granite chat — untrained, own chat format, Granite's splitting rule", 10),
                ("Granite plain — untrained",                                   2),
                ("Granite chat — untrained, V's template",                      9),
                ("Granite chat — trained (own chat format)",                    6),
                ("Granite plain — trained (V's template)",                      8)]
        scores = {}
        lines  = [f"# PY-V pipeline report", f"_Updated {datetime.datetime.now():%Y-%m-%d %H:%M}_", "",
                  "## Stages", "| Stage | Status | Where | Minutes |", "|---|---|---|---|"]
        lines += [f"| {n}. {STAGES[n]} | {s} | {w} | {m} |" for n, (s, w, m) in sorted(self.status.items())]
        lines += ["", "## Scores", "| Brain | MBPP /100 | Long-file /10 | Chat /8 | Fix /40 | Improve /40 |",
                  "|---|---|---|---|---|---|"]
        for label, stage in rows:
            tag = self.test_tag(*TEST_STAGES[stage])
            got = scores[tag] = self.summaries(tag)
            cell = lambda p, key="score": str(got[p].get(key, "")) if p in got else "—"
            fix  = got.get("fix", {})
            lines.append(f"| {label} | {cell('mbpp')} | {cell('longctx', 'passed')} | {cell('chat', 'passed')} | "
                         f"{fix.get('fix', {}).get('passed', '—')} | {fix.get('improve', {}).get('passed', '—')} |")
        lines += ["", "## Training"]
        training = {}
        for brain in BRAINS:
            meta = self.adapter_dir(brain) / "v_adapter.json"
            if meta.exists():
                m = json.load(open(meta, encoding="utf-8"))
                training[brain] = m
                lines.append(f"- **{brain}** ({m['base_model']}, {m['prompt_format']}, splitting rules of "
                             f"{m.get('split_rules_from', m['base_model'])}): {m['steps']} steps, "
                             f"{m['minutes']} min, train loss {m['train_loss']}, check-set loss {m['eval_loss']}, "
                             f"GPU: {m['gpu']}")
            else:
                lines.append(f"- **{brain}**: not trained yet")
        lines += ["", "Detailed answers per question: `results/eval/` (`mbpp_`, `longctx_`, `chat_`, `fix_` files). "
                  "Training logs: `model_<brain>/lora/training_log.json`. Full console log: `results/pipeline_log.txt`."]
        self.results.mkdir(parents=True, exist_ok=True)
        (self.results / "PIPELINE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with open(self.results / "PIPELINE_REPORT.json", "w", encoding="utf-8") as f:
            json.dump({"stages": [{"stage": n, "name": STAGES[n], "status": s, "where": w, "minutes": m}
                                  for n, (s, w, m) in sorted(self.status.items())],
                       "scores": scores, "training": training}, f, indent=1)

    # ─── order ────────────────────────────────────────────────────────────────

    def go(self):
        started = time.time()
        notes   = self.seed_from_inputs()           # before the log opens: an earlier run's log continues
        self.say(f"\n##### PY-V pipeline started {datetime.datetime.now():%Y-%m-%d %H:%M} - "
                 f"{self.gpus} GPU lane(s) + data lane #####")
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip() if count_gpus() else "no GPU found"
        self.say(gpus)
        for note in notes:
            self.say(note)
        self.link_outputs()
        self.apply_redo()
        self.write_report()

        data_ready = threading.Event()
        data = Lane("data", "")
        if self.gpus >= 2:
            gpu_lanes = [(Lane("chat", "0"), (1, 10), (5, 6)), (Lane("plain", "1"), (2,), (7, 8))]
        else:
            gpu_lanes = [(Lane("gpu", "0"), (1, 2, 10), (5, 6, 7, 8))]

        def data_lane():
            try:
                for n in (3, 4):
                    self.stage(n, data)
            finally:
                data_ready.set()

        def gpu_lane(lane, before, after):
            for n in before:
                self.stage(n, lane)
            while not data_ready.is_set() and (n := self.next_extra()) is not None:
                self.stage(n, lane)                 # use the wait for the training data
            data_ready.wait()
            for n in after:
                self.stage(n, lane)
            while (n := self.next_extra()) is not None:
                self.stage(n, lane)

        threads = [threading.Thread(target=data_lane)]
        threads += [threading.Thread(target=gpu_lane, args=spec) for spec in gpu_lanes]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.say(f"\n##### Pipeline finished {datetime.datetime.now():%H:%M} after "
                 f"{(time.time() - started) / 3600:.1f} h - report: {self.results / 'PIPELINE_REPORT.md'} #####")
        print((self.results / "PIPELINE_REPORT.md").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="PY-V one-button GPU run (Kaggle or Colab)")
    parser.add_argument("--root", "--drive", dest="root", required=True,
                        help="where everything is saved: Kaggle /kaggle/working/PY-V, Colab /content/drive/MyDrive/PY-V")
    parser.add_argument("--inputs", nargs="*", default=[],
                        help="read-only folders with uploaded source files and/or an earlier run's saved folder "
                             "(Kaggle: /kaggle/input)")
    parser.add_argument("--gpus", type=int, default=None,
                        help="GPU lanes (default: every GPU found, at most 2 - one per Granite version)")
    parser.add_argument("--stop-after", type=float, default=None,
                        help="hours; stop the jobs by then so the run ends normally (Kaggle: 11, its limit is 12 h)")
    args = parser.parse_args()
    gpus = args.gpus if args.gpus is not None else count_gpus()
    Pipeline(Path(args.root), [Path(p) for p in args.inputs], max(1, min(gpus, 2)), args.stop_after).go()


if __name__ == "__main__":
    main()
