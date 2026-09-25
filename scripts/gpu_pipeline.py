"""
gpu_pipeline.py — PY-V (scripts/)
One-button GPU run: every big job for the Granite brain upgrade. Started by
the RUN EVERYTHING cell of Kaggle/py_v_kaggle.ipynb (background run on
GPU T4 x2) or of Google Colab/py_v_runner.ipynb (one T4, results on Drive).

  1-2  test both untrained Granite versions (chat in its own format, plain)
  3    training data: every source file (uploaded, from an earlier run, or
       re-made if missing) + the long-file fix examples
  4    rebuild the training set (dataset v3 = v2 + long-file examples)
  5-6  train the chat version (own chat format), then test it
  7-8  train the plain version (V's template), then test it
  9    extra: the chat version with V's template (untrained)

Every test = MBPP (write code) + long-file bug fixes + chat + fix/improve.

Lanes run at the same time, each job seeing only its own GPU (CUDA_VISIBLE_DEVICES):
  data lane, CPU:     3 → 4 (the trainings wait for it)
  two GPUs (Kaggle):  chat lane on GPU 0: 1 → 5 → 6   plain lane on GPU 1: 2 → 7 → 8
  one GPU (Colab):    1 → 2 → 5 → 6 → 7 → 8
Stage 9 is done by whichever GPU lane is free first (also while one waits for the data).

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
TESTS = [("mbpp", "eval_mbpp"), ("longctx", "eval_long_context"), ("chat", "eval_chat"), ("fix", "eval_fix")]
STAGES = {
    1: "Test Granite chat (untrained, own chat format)",
    2: "Test Granite plain (untrained)",
    3: "Training data (missing sources + long-file fix examples)",
    4: "Build training set v3",
    5: "Train Granite chat (own chat format)",
    6: "Test trained Granite chat",
    7: "Train Granite plain (V's template)",
    8: "Test trained Granite plain",
    9: "Extra: test Granite chat with V's template (untrained)",
}
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
        return {
            1: (self.tests_done("chat", False, True),   self.run_tests("chat", False, True),   None),
            2: (self.tests_done("plain", False, False), self.run_tests("plain", False, False), None),
            3: (self.data_done, self.make_data, None),
            4: (self.dataset_done, lambda lane: self.run(lane, ["data.scripts.build_dataset_v2"], timeout_h=1),
                self.data_done),
            5: (self.trained("chat"), self.train("chat"), self.dataset_done),
            6: (self.tests_done("chat", True, True), self.run_tests("chat", True, True), self.trained("chat")),
            7: (self.trained("plain"), self.train("plain"), self.dataset_done),
            8: (self.tests_done("plain", True, False), self.run_tests("plain", True, False), self.trained("plain")),
            9: (self.tests_done("chat", False, False), self.run_tests("chat", False, False), None),
        }

    def adapter_dir(self, brain: str) -> Path:
        return self.root / f"model_{BRAINS[brain]['model'].split('/')[-1]}" / "lora"

    def test_args(self, brain: str, trained: bool, native: bool) -> list:
        args = ["--model", BRAINS[brain]["model"]]
        if trained:
            args += ["--adapter", str(self.adapter_dir(brain))]
        else:
            args += ["--base"] + (["--native-chat"] if native else [])
        return args

    def test_tag(self, brain: str, trained: bool, native: bool) -> str:
        args = argparse.Namespace(base=not trained, model=BRAINS[brain]["model"],
                                  adapter=str(self.adapter_dir(brain)), native_chat=native and not trained)
        return result_tag(args)

    def tests_done(self, brain, trained, native):
        tag = self.test_tag(brain, trained, native)
        return lambda: all((OUTPUTS / f"{prefix}_{tag}_summary.json").exists() for prefix, _ in TESTS)

    def run_tests(self, brain, trained, native):
        """All four tests with one model load (eval_all), skipping tests already saved."""
        return lambda lane: self.run(lane, ["experiments.eval_all", *self.test_args(brain, trained, native),
                                            "--skip-done"], timeout_h=4)

    def trained(self, brain):
        return lambda: all((self.adapter_dir(brain) / f).exists()
                           for f in ("adapter_model.safetensors", "v_adapter.json"))

    def train(self, brain):
        b = BRAINS[brain]
        return lambda lane: self.run(lane, ["model.training.train_lora_t4", "--model", b["model"],
                                            "--prompt-format", b["format"],
                                            "--output-dir", str(self.adapter_dir(brain))], timeout_h=8)

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
        rows = [("Granite chat — untrained, own chat format", "chat", False, True),
                ("Granite plain — untrained",                 "plain", False, False),
                ("Granite chat — untrained, V's template",    "chat", False, False),
                ("Granite chat — trained (own chat format)",  "chat", True, True),
                ("Granite plain — trained (V's template)",    "plain", True, False)]
        scores = {}
        lines  = [f"# PY-V pipeline report", f"_Updated {datetime.datetime.now():%Y-%m-%d %H:%M}_", "",
                  "## Stages", "| Stage | Status | Where | Minutes |", "|---|---|---|---|"]
        lines += [f"| {n}. {STAGES[n]} | {s} | {w} | {m} |" for n, (s, w, m) in sorted(self.status.items())]
        lines += ["", "## Scores", "| Brain | MBPP /100 | Long-file /10 | Chat /8 | Fix /40 | Improve /40 |",
                  "|---|---|---|---|---|---|"]
        for label, brain, trained, native in rows:
            tag = self.test_tag(brain, trained, native)
            got = {}
            for prefix, _ in TESTS:
                path = OUTPUTS / f"{prefix}_{tag}_summary.json"
                if path.exists():
                    got[prefix] = json.load(open(path, encoding="utf-8"))
            scores[tag] = got
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
                lines.append(f"- **{brain}** ({m['base_model']}, {m['prompt_format']}): {m['steps']} steps, "
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
        self.write_report()

        data_ready = threading.Event()
        data = Lane("data", "")
        if self.gpus >= 2:
            gpu_lanes = [(Lane("chat", "0"), (1,), (5, 6)), (Lane("plain", "1"), (2,), (7, 8))]
        else:
            gpu_lanes = [(Lane("gpu", "0"), (1, 2), (5, 6, 7, 8))]

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
