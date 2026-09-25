"""
colab_pipeline.py — PY-V (scripts/)
One-button Colab run: every big job for the Granite brain upgrade, one after
the other, most important first. Started by the "RUN EVERYTHING" cell of
Google Colab/py_v_runner.ipynb (which mounts Drive and links the output
folders first).

  1-2  test both untrained Granite versions (chat in its own format, plain)
  3    training data: every source file on Drive (re-made if missing — the old
       GitHub one from the v1 dataset on Drive) + the long-file fix examples
  4    rebuild the training set (dataset v3 = v2 + long-file examples)
  5-6  train the chat version (own chat format), then test it
  7-8  train the plain version (V's template), then test it
  9    extra: the chat version with V's template (untrained)

Every test = MBPP (write code) + long-file bug fixes + chat + fix/improve.
A stage whose results are already on Drive is skipped, and training resumes
from its last checkpoint — so when Colab disconnects or the day's GPU time
runs out, running the cell again continues where it stopped. After every
stage the report is rewritten: {drive}/results/PIPELINE_REPORT.md (+ .json),
and everything printed is also appended to {drive}/results/pipeline_log.txt.

Usage (from repo root, on Colab):
    python -m scripts.colab_pipeline --drive /content/drive/MyDrive/PY-V
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.training.config_loader import CFG
from experiments.eval_common import result_tag

BRAINS = {
    "chat":  {"model": "ibm-granite/granite-4.2-3b",      "format": "native_chat"},
    "plain": {"model": "ibm-granite/granite-4.1-3b-base", "format": "template"},
}
TESTS = [("mbpp", "eval_mbpp"), ("longctx", "eval_long_context"), ("chat", "eval_chat"), ("fix", "eval_fix")]
OUTPUTS = Path("experiments/outputs")          # linked to Drive results/eval by the notebook
V1_MD5  = "4028f2d3a010abf0aa36d52c498031e9"   # the v1 dataset (source of old_github) — laptop data/datasets/train.jsonl
ENV     = {**os.environ, "PYTHONUNBUFFERED": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}   # less fragmentation → more of the T4 usable


class Pipeline:
    def __init__(self, drive: Path):
        self.drive   = drive
        self.results = drive / "results"
        self.log     = self.results / "pipeline_log.txt"
        self.stages  = []                       # (name, status, minutes)
        self.results.mkdir(parents=True, exist_ok=True)

    # ─── helpers ──────────────────────────────────────────────────────────────

    def say(self, text: str):
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(text, flush=True)
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {text}\n")

    def run(self, args: list, timeout_h: float = 6) -> bool:
        """Run a module (python -m ...), streaming its output. True if it succeeded."""
        self.say("$ python -m " + " ".join(args))
        proc = subprocess.Popen([sys.executable, "-m", *args], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=ENV, bufsize=1)
        deadline = time.time() + timeout_h * 3600
        with open(self.log, "a", encoding="utf-8") as log:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                if time.time() > deadline:
                    proc.kill()
                    self.say(f"TIMEOUT after {timeout_h} h")
                    return False
        return proc.wait() == 0

    def stage(self, name: str, done, work, needs=None):
        """Run one stage unless its results already exist; record what happened."""
        start = time.time()
        if needs is not None and not needs():
            status = "blocked (an earlier stage is missing)"
        elif done():
            status = "done earlier - skipped"
        else:
            self.say(f"\n{'=' * 70}\n=== {name}\n{'=' * 70}")
            status = "done" if work() and done() else "FAILED"
        minutes = (time.time() - start) / 60
        self.stages.append((name, status, round(minutes, 1)))
        self.say(f"--- {name}: {status} ({minutes:.0f} min)")
        self.write_report()

    # ─── the jobs ─────────────────────────────────────────────────────────────

    def adapter_dir(self, brain: str) -> Path:
        return self.drive / f"model_{BRAINS[brain]['model'].split('/')[-1]}" / "lora"

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
        """All four tests with one model load (eval_all), skipping tests already on Drive."""
        return lambda: self.run(["experiments.eval_all", *self.test_args(brain, trained, native), "--skip-done"],
                                timeout_h=4)

    def trained(self, brain):
        return lambda: all((self.adapter_dir(brain) / f).exists()
                           for f in ("adapter_model.safetensors", "v_adapter.json"))

    def train(self, brain):
        b = BRAINS[brain]
        return lambda: self.run(["model.training.train_lora_t4", "--model", b["model"],
                                 "--prompt-format", b["format"], "--output-dir", str(self.adapter_dir(brain))],
                                timeout_h=8)

    def _missing_sources(self) -> list:
        """Sources the training set mixes whose file is not on Drive (fetch_sources
        writes a file only when it is complete)."""
        return [n for n in CFG.dataset_v2.build["take"] if not (CFG.dataset_v2.output_dir / f"{n}.jsonl").exists()]

    def data_done(self):
        return not self._missing_sources()

    def make_data(self) -> bool:
        ok = True
        for name in self._missing_sources():
            if name == "old_github" and not self._restore_v1():
                ok = False
                continue
            ok = self.run(["data.scripts.fetch_sources", "--only", name], timeout_h=3) and ok
        return ok

    def _restore_v1(self) -> bool:
        """The v1 dataset is not on GitHub: copy it from Drive (matched by fingerprint)."""
        target = Path(CFG.dataset_v2.sources["old_github"]["path"])
        if target.exists():
            return True
        found = []
        for depth in ("*", "*/*", "*/*/*"):
            found += glob.glob(str(self.drive.parent / depth / "data" / "datasets" / "train.jsonl"))
        for path in found:
            if hashlib.md5(open(path, "rb").read()).hexdigest() == V1_MD5:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, target)
                self.say(f"v1 dataset copied from {path}")
                return True
        self.say("v1 dataset NOT FOUND on Drive - upload the laptop file D:/Py-V/data/raw/v2/old_github.jsonl "
                 f"to {self.results / 'data_v2'} and run again")
        return False

    def dataset_done(self):
        report = CFG.dataset_v2.build["output_dir"] / "build_report.json"
        return report.exists() and "long_file_fix" in json.load(open(report, encoding="utf-8"))

    # ─── report ───────────────────────────────────────────────────────────────

    def write_report(self):
        rows = [("Granite chat — untrained, own chat format", "chat", False, True),
                ("Granite plain — untrained",                 "plain", False, False),
                ("Granite chat — untrained, V's template",    "chat", False, False),
                ("Granite chat — trained (own chat format)",  "chat", True, True),
                ("Granite plain — trained (V's template)",    "plain", True, False)]
        scores = {}
        lines  = [f"# PY-V pipeline report", f"_Updated {datetime.datetime.now():%Y-%m-%d %H:%M}_", "",
                  "## Stages", "| Stage | Status | Minutes |", "|---|---|---|"]
        lines += [f"| {n} | {s} | {m} |" for n, s, m in self.stages]
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
        (self.results / "PIPELINE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with open(self.results / "PIPELINE_REPORT.json", "w", encoding="utf-8") as f:
            json.dump({"stages": self.stages, "scores": scores, "training": training}, f, indent=1)

    # ─── order ────────────────────────────────────────────────────────────────

    def go(self):
        self.say(f"\n##### PY-V pipeline started {datetime.datetime.now():%Y-%m-%d %H:%M} #####")
        subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"], env=ENV)

        self.stage("1. Test Granite chat (untrained, own chat format)",
                   self.tests_done("chat", False, True), self.run_tests("chat", False, True))
        self.stage("2. Test Granite plain (untrained)",
                   self.tests_done("plain", False, False), self.run_tests("plain", False, False))
        self.stage("3. Training data (missing sources + long-file fix examples)",
                   self.data_done, self.make_data)
        self.stage("4. Build training set v3",
                   self.dataset_done, lambda: self.run(["data.scripts.build_dataset_v2"], timeout_h=1),
                   needs=self.data_done)
        self.stage("5. Train Granite chat (own chat format)",
                   self.trained("chat"), self.train("chat"), needs=self.dataset_done)
        self.stage("6. Test trained Granite chat",
                   self.tests_done("chat", True, True), self.run_tests("chat", True, True),
                   needs=self.trained("chat"))
        self.stage("7. Train Granite plain (V's template)",
                   self.trained("plain"), self.train("plain"), needs=self.dataset_done)
        self.stage("8. Test trained Granite plain",
                   self.tests_done("plain", True, False), self.run_tests("plain", True, False),
                   needs=self.trained("plain"))
        self.stage("9. Extra: test Granite chat with V's template (untrained)",
                   self.tests_done("chat", False, False), self.run_tests("chat", False, False))

        self.say(f"\n##### Pipeline finished {datetime.datetime.now():%H:%M} — report: "
                 f"{self.results / 'PIPELINE_REPORT.md'} #####")
        print((self.results / "PIPELINE_REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", required=True, help="Drive project folder, e.g. /content/drive/MyDrive/PY-V")
    Pipeline(Path(parser.parse_args().drive)).go()
