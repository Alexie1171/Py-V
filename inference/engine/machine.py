"""
machine.py — PY-V (inference/engine/)
V knows the computer she runs on: what it has (GPU, RAM, CPU — read once) and
how busy it is right now (free RAM, free GPU memory, CPU load — read before
every answer). Three levels: free / busy / tight (config machine.busy / .tight,
any one resource past its line counts). When it's busy she takes on less
(machine.busy_work / tight_work: shorter chat history, fewer memories, shorter
prose answers), runs gently (lower process priority, half the CPU threads) so
other programs stay smooth, and says so casually — not every message, in
different words each time. Never uses the brain.
"""

import os
import platform
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil
import torch

from model.training.config_loader import CFG, WorkLimits

GB = 1024 ** 3
LEVELS = ("free", "busy", "tight")
MIN_GAP_SECONDS = 120   # never two heads-ups closer than this, even when it gets worse


@dataclass
class Snapshot:
    ram_total_gb: float
    ram_free_gb:  float
    cpu_percent:  float
    gpu_total_gb: Optional[float] = None   # None = no GPU
    gpu_free_gb:  Optional[float] = None   # free on the card + what V's brain has cached and can reuse
    level:        str  = "free"
    reasons:      list = field(default_factory=list)   # what is past the line: "ram", "gpu", "cpu" (worst first)


def _cpu_name() -> str:
    try:
        if os.name == "nt":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown CPU"


class Machine:

    def __init__(self, cfg=None):
        self.cfg      = cfg or CFG.machine
        self.specs    = self._read_specs()
        self._threads = torch.get_num_threads()
        self._gentle  = False
        self._said    = None    # level of the last heads-up (None = nothing said / all clear)
        self._said_at = 0.0
        psutil.cpu_percent(interval=None)   # first call starts the measurement

    # ─── what the machine has ─────────────────────────────────────────────────

    @staticmethod
    def _read_specs() -> dict:
        specs = {"cpu": _cpu_name(),
                 "cores": psutil.cpu_count(logical=False) or 0,
                 "threads": psutil.cpu_count(logical=True) or 0,
                 "ram_gb": psutil.virtual_memory().total / GB,
                 "gpu": None, "gpu_gb": None}
        if torch.cuda.is_available():
            specs["gpu"]    = torch.cuda.get_device_name(0)
            specs["gpu_gb"] = torch.cuda.get_device_properties(0).total_memory / GB
        return specs

    # ─── how busy it is ───────────────────────────────────────────────────────

    def check(self) -> Snapshot:
        ram  = psutil.virtual_memory()
        snap = Snapshot(ram_total_gb=ram.total / GB, ram_free_gb=ram.available / GB,
                        cpu_percent=psutil.cpu_percent(interval=None))
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            cached = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            snap.gpu_total_gb, snap.gpu_free_gb = total / GB, (free + cached) / GB

        for level in ("tight", "busy"):
            lines   = getattr(self.cfg, level)
            reasons = [name for name, past in (
                ("ram", snap.ram_free_gb < lines["ram_free_gb"]),
                ("gpu", snap.gpu_free_gb is not None and snap.gpu_free_gb < lines["gpu_free_gb"]),
                ("cpu", snap.cpu_percent > lines["cpu_percent"])) if past]
            if reasons:
                snap.level, snap.reasons = level, reasons
                break
        self._pace(snap.level != "free")
        return snap

    def work(self, snap: Optional[Snapshot]) -> WorkLimits:
        """The most V takes on for this answer: the normal settings, capped when busy / tight."""
        normal = WorkLimits(history_turns=CFG.memory.history_turns, memory_facts=CFG.memory.top_k,
                            memory_code=CFG.memory.code_top_k, prose_max_tokens=None,
                            file_chars=CFG.files.max_prompt_chars)
        if snap is None or snap.level == "free":
            return normal
        cap = self.cfg.tight_work if snap.level == "tight" else self.cfg.busy_work
        return WorkLimits(history_turns    = min(normal.history_turns, cap.history_turns),
                          memory_facts     = min(normal.memory_facts,  cap.memory_facts),
                          memory_code      = min(normal.memory_code,   cap.memory_code),
                          prose_max_tokens = min(CFG.model.max_tokens, cap.prose_max_tokens),
                          file_chars       = min(normal.file_chars, cap.file_chars or normal.file_chars))

    def _pace(self, gentle: bool):
        """Gentle = lower process priority (Windows; elsewhere it could not be raised
        back without admin rights) and half the CPU threads — V answers a bit
        slower, other programs stay smooth."""
        gentle = gentle and self.cfg.gentle
        if gentle == self._gentle:
            return
        try:
            if os.name == "nt":
                psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if gentle else psutil.NORMAL_PRIORITY_CLASS)
            torch.set_num_threads(max(1, self._threads // 2) if gentle else self._threads)
            self._gentle = gentle
        except Exception:
            pass

    # ─── what V says about it ─────────────────────────────────────────────────

    def heads_up(self, snap: Snapshot) -> Optional[str]:
        """A casual note for the user, or None. Said when the machine gets busier,
        again after heads_up_minutes at the same level, and once when it is free
        again after a note."""
        now = time.monotonic()
        if snap.level == "free":
            if self._said is None:
                return None
            self._said = None
            return random.choice(_ALL_CLEAR)

        if self._said is not None and LEVELS.index(snap.level) < LEVELS.index(self._said):
            self._said = snap.level   # eased off a little: nothing to say
            return None
        worse   = self._said is None or LEVELS.index(snap.level) > LEVELS.index(self._said)
        overdue = now - self._said_at >= self.cfg.heads_up_minutes * 60
        if not (worse or overdue) or now - self._said_at < MIN_GAP_SECONDS:
            return None
        self._said, self._said_at = snap.level, now
        return _phrase(snap)


# ─── V's words — picked from parts so it never reads the same twice ──────────

_OPENERS = {"busy":  ["Heads up,", "Quick note,", "Psst,", "Just so you know,", "FYI,"],
            "tight": ["Whoa,", "Okay, heads up:", "Yikes,", "Hey, real quick:"]}

_WHAT = {
    "ram": ["your RAM's pretty full ({ram_free:.1f} GB free)",
            "memory's running low ({ram_used:.1f} of {ram_total:.1f} GB used)",
            "your laptop's RAM is nearly used up",
            "there's not much RAM left ({ram_free:.1f} GB)"],
    "gpu": ["the graphics card's memory is nearly full ({gpu_free:.1f} GB free)",
            "the GPU is pretty packed right now",
            "there's only {gpu_free:.1f} GB of GPU memory left"],
    "cpu": ["your CPU's working hard ({cpu:.0f}%)",
            "the processor's pretty busy right now",
            "the CPU is running at {cpu:.0f}%"],
}

_DID = {"busy":  ["so I kept things a bit lighter.", "so I went easy this time.",
                  "so I trimmed my notes a little.", "so I'm taking it a bit slower."],
        "tight": ["so I kept this one short so nothing freezes.", "so I'm going extra easy on your laptop.",
                  "so I cut back to keep things from freezing.", "so I kept it short and slow."]}

_ASK = ["Closing a tab or two would really help.", "If you can close something, I'll be back to full speed.",
        "Maybe close an app you're not using?", ""]

_ALL_CLEAR = ["All good now, your laptop's breathing again. Back to full speed!",
              "Things calmed down, so I'm back to normal.",
              "Looks like there's room again, back to full speed.",
              "Okay, much better now. Back to my usual self!"]


def _phrase(snap: Snapshot) -> str:
    numbers = {"ram_free": snap.ram_free_gb, "ram_used": snap.ram_total_gb - snap.ram_free_gb,
               "ram_total": snap.ram_total_gb, "gpu_free": snap.gpu_free_gb or 0.0, "cpu": snap.cpu_percent}
    parts = [random.choice(_OPENERS[snap.level]),
             random.choice(_WHAT[snap.reasons[0]]).format(**numbers) + ",",
             random.choice(_DID[snap.level])]
    if snap.level == "tight":
        parts.append(random.choice(_ASK))
    return " ".join(p for p in parts if p)


if __name__ == "__main__":   # python -m inference.engine.machine — what V sees (no brain)
    from inference.engine.prompt_builder import format_machine
    machine = Machine()
    time.sleep(1)            # CPU load is measured between two readings
    snap = machine.check()
    print(format_machine(machine.specs, snap))
    print(f"Level: {snap.level} {snap.reasons or ''} -> {machine.work(snap)}")
