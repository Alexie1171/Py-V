"""
eval_file_context.py — PY-V (experiments/)
File reading test for the chat panel (Phase 10.2) — instant, no brain.

Runs messages through ChatEngine._prepare() with a stand-in brain and an open
file, and checks the mode, the answer's language and what of the file went
into the prompt (inference/engine/file_context.py). Plus checks of the piece
splitting on long Python / Java files.

Usage (from repo root):
    python -m experiments.eval_file_context
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.training.config_loader import CFG

CFG.memory.enabled  = False   # no database, no embedder
CFG.machine.enabled = False   # the normal budget, whatever the laptop is doing

from inference.engine import file_context as fc
from inference.engine.chat import ChatEngine
from inference.engine.context_manager import ContextManager
from inference.engine.controller import Controller


# ─── Sample files ─────────────────────────────────────────────────────────────

def _python_file() -> str:
    parts = ['"""Shop tools."""', "", "import json", "from pathlib import Path", ""]
    for i in range(40):
        parts += ["", f"def helper_{i}(items):", f'    """Helper number {i}."""', "    total = 0",
                  "    for item in items:", f"        total += item * {i}", "    return total", ""]
    parts += ["", "def parse_config(path):", "    data = json.loads(Path(path).read_text())",
              "    return data['shop']", ""]
    return "\n".join(parts)


PY      = _python_file()
PY_LINE = PY.split("\n").index("def helper_6(items):") + 1
SEL     = "\n".join(PY.split("\n")[PY_LINE - 1:PY_LINE + 6])
TS      = "export function add(a: number, b: number) {\n  return a - b;\n}\n"


def _java_file() -> list:
    lines = ["public class Shop {", "    private int total;", ""]
    for i in range(8):
        lines += [f"    // method {i}", f"    public int m{i}(int x) {{", "        int y = x * 2;", "        if (y > 10) {",
                  "            y = y - 1;", "        }", "", "        return y + total;", "    }", ""]
    return lines + ["}"]


py     = {"name": "src/shop.py", "language_id": "python", "content": PY, "cursor_line": PY_LINE + 2}
py_sel = dict(py, selection=SEL, selection_line=PY_LINE)
ts     = {"name": "web/app.ts", "language_id": "typescript", "content": TS, "cursor_line": 2}
yml    = {"name": "configs/app.yaml", "language_id": "yaml", "content": "model:\n  name: x\n", "cursor_line": 1}

# (message, open file, mode, answer language (None = Python), what of the file goes in: "sel" / "file" / None)
CASES = [
    ("fix this function",                                 py_sel, "debug",    None,         "sel"),
    ("hmm",                                               py_sel, "explain",  None,         "sel"),
    ("explain",                                           py_sel, "explain",  None,         "sel"),
    ("why is this slow?",                                 py_sel, "explain",  None,         "sel"),
    ("what does total mean here",                         py_sel, "explain",  None,         "sel"),
    ("what is a decorator?",                              py_sel, "explain",  None,         None),
    ("how are you?",                                      py_sel, "chat",     None,         None),
    ("fix this:\n```python\ndef add(a, b): return a - b\n```", py_sel, "debug", None,       None),
    ("refactor this function to be more readable",        py,     "refactor", None,         "file"),
    ("clean up the code",                                 py,     "refactor", None,         "file"),
    ("what does this file do?",                           py,     "explain",  None,         "file"),
    ("what does parse_config do?",                        py,     "explain",  None,         "file"),
    ("what is a closure?",                                py,     "explain",  None,         None),
    ("write a function that reverses a string",           py,     "generate", None,         None),
    ("fix the bug",                                       ts,     "debug",    "TypeScript", "file"),
    ("write a function that reverses a string",           ts,     "generate", "TypeScript", None),
    ("write a function that reverses a string in python", ts,     "generate", None,         None),
    ("write a function that reverses a string",           yml,    "generate", None,         None),
    ("add a logging section to this file",                yml,    "generate", "YAML",       "file"),
]


class _Tokenizer:
    chat_template = "stand-in"

    def apply_chat_template(self, messages, **kw):
        return "\n".join(f"[{m['role']}] {m['content']}" for m in messages)


class _Brain:
    v_prompt_format = "native_chat"


def _engine() -> ChatEngine:
    engine = ChatEngine.__new__(ChatEngine)   # no brain loaded
    engine.controller      = Controller()
    engine.memory          = None
    engine.context_manager = ContextManager(None)
    engine.machine         = None
    engine.retriever       = None
    engine.model, engine.tokenizer = _Brain(), _Tokenizer()
    return engine


def main():
    engine, passed, total = _engine(), 0, 0

    print("Messages with an open file:")
    for message, file, mode, language, goes_in in CASES:
        turn = engine._prepare("eval-file", message, file)
        got_lang = turn.language["name"] if turn.language else None
        got_in   = None
        if turn.file_label:
            got_in = "sel" if "Selected code from" in turn.prompt else "file"
        ok = (turn.intent.mode, got_lang, got_in) == (mode, language, goes_in)
        passed += ok
        total  += 1
        print(f"  {'ok  ' if ok else 'MISS'} {message[:48]!r:52} {file['name']:18} mode={turn.intent.mode:8} "
              f"lang={got_lang} read={turn.file_label}"
              + ("" if ok else f"   (want {mode} / {language} / {goes_in})"))

    print("\nLong files in pieces:")
    checks = []
    turn = engine._prepare("eval-file", "what does parse_config do?", py)
    checks.append(("the piece that defines parse_config goes in", "def parse_config" in turn.prompt))
    checks.append(("the imports at the top go in", "import json" in turn.prompt))
    checks.append(("left-out lines are marked", "left out)" in turn.prompt))
    turn = engine._prepare("eval-file", "refactor this function to be more readable", py)
    checks.append(("the function at the cursor goes in", "def helper_6" in turn.prompt))
    checks.append(("improve gets only that function (no neighbours, no imports)",
                   "def helper_5" not in turn.prompt and "import json" not in turn.prompt))
    turn = engine._prepare("eval-file", "fix the bug in this function", py)
    checks.append(("fix also gets the neighbours and imports",
                   "def helper_5" in turn.prompt and "import json" in turn.prompt))
    checks.append(("improve gets no more than fits in one answer",
                   len(turn.prompt) < CFG.model.max_tokens * fc.CODE_BUDGET_CHARS_PER_TOKEN + 1200))
    java   = _java_file()
    pieces = fc.split_pieces(java, 20)
    checks.append(("a Java class splits into its methods", len(pieces) >= 8 and max(b - a for a, b in pieces) <= 13))
    code, _ = fc._cut("\n".join(java), 1, java.index("    public int m5(int x) {"), "fix m5", 600, 20)
    checks.append(("the Java method at the cursor goes in", "public int m5" in code and "public int m2" not in code))
    lines = PY.split("\n")
    pieces = fc.split_pieces(lines, 40)
    checks.append(("pieces cover the whole file, in order",
                   pieces[0][0] == 0 and pieces[-1][1] == len(lines)
                   and all(pieces[i][1] == pieces[i + 1][0] for i in range(len(pieces) - 1))))
    for name, ok in checks:
        passed += ok
        total  += 1
        print(f"  {'ok  ' if ok else 'MISS'} {name}")

    print(f"\nFILE READING score: {passed}/{total}")
    return passed == total


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
