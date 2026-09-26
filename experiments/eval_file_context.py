"""
eval_file_context.py — PY-V (experiments/)
File reading, updating and project search test for the chat panel (Phases
10.2, 10.3, 10.5) — instant, no brain.

Reading: runs messages through ChatEngine._prepare() with a stand-in brain and
an open file, and checks the mode, the answer's language and what of the file
went into the prompt (inference/engine/file_context.py), plus the piece
splitting on long Python / Java files.
Updating: where a code block from an answer goes in the file ("Apply to
file", inference/engine/file_update.py) — selection, named functions /
methods / classes, whole file, at the cursor; Python, TypeScript, Java, Go.
Project search: a small project on disk (retrieval/project.py, keyword search
— the embedder isn't loaded here): what gets indexed, what a question finds,
updates after edits, and the pieces in the prompt through ChatEngine.

Usage (from repo root):
    python -m experiments.eval_file_context
"""

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.training.config_loader import CFG

CFG.memory.enabled          = False   # no database, no embedder
CFG.memory.semantic_search  = False   # project search: keyword only (no embedder loaded)
CFG.machine.enabled         = False   # the normal budget, whatever the laptop is doing
CFG.learning.enabled        = False

from inference.engine import file_context as fc
from inference.engine.file_update import propose
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
    engine.learning        = None
    engine.projects, engine._projects_lock = {}, threading.Lock()
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

    print("\nApply to file (where the code goes):")
    for name, ok in _update_checks():
        passed += ok
        total  += 1
        print(f"  {'ok  ' if ok else 'MISS'} {name}")

    print("\nProject search:")
    for name, ok in _project_checks(engine):
        passed += ok
        total  += 1
        print(f"  {'ok  ' if ok else 'MISS'} {name}")

    print(f"\nPANEL FILES score: {passed}/{total}")
    return passed == total


SHOP = '''"""Shop."""
import json


def add(a, b):
    return a - b


class Shop:
    def __init__(self):
        self.items = []

    @property
    def total(self):
        t = 0
        for i in self.items:
            t = t + i
        return t


def main():
    print(add(1, 2))
'''


PROJECT = {
    "config/loader.py": "import yaml\n\n\ndef load_config(path):\n    \"\"\"Read the settings file.\"\"\"\n"
                        "    with open(path) as f:\n        return yaml.safe_load(f)\n",
    "app/main.py":      "from config.loader import load_config\n\n\ndef main():\n    cfg = load_config('app.yaml')\n"
                        "    print(cfg['name'])\n",
    "app/utils.py":     "import re\n\n\ndef slugify(text):\n    return re.sub(r'[^a-z0-9]+', '-', text.lower())\n",
    "README.md":        "# Shop\n\nA small shop app. Run it with python -m app.main and the settings in app.yaml.\n",
    "node_modules/lib/index.js": "module.exports = function load_config() {}\n",
}


def _project_checks(engine) -> list:
    from retrieval.project import ProjectIndex, format_hits

    checks = []
    tmp    = Path(tempfile.mkdtemp(prefix="pyv-project-"))
    root   = tmp / "shop"
    for rel, text in PROJECT.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "big.json").write_text("[" + ",".join(["1"] * 40000) + "]", encoding="utf-8")

    index = ProjectIndex(str(root), tmp / "index", embed=None)
    index.update()
    files = {r["path"] for r in index._db.execute("SELECT DISTINCT path FROM pieces")}
    checks.append(("code + docs indexed; node_modules and a big JSON file skipped",
                   files == {"config/loader.py", "app/main.py", "app/utils.py", "README.md"}))

    hits = index.search("where is the config loaded from the settings file?")
    checks.append(("'where is the config loaded' finds the loader", hits and hits[0].path == "config/loader.py"))
    hits = index.search("what does slugify() do?")
    checks.append(("a function name finds its definition", hits and hits[0].path == "app/utils.py"))
    hits = index.search("where is the config loaded from the settings file?", skip_path="config/loader.py")
    checks.append(("the open file is skipped (it goes in through file reading)",
                   all(h.path != "config/loader.py" for h in hits)))
    checks.append(("an unrelated question finds nothing", not index.search("how do I bake bread at home")))
    block, labels = format_hits(index.search("what does slugify() do?"), 200)
    checks.append(("found pieces fit the budget, with file and lines", block.startswith("Related code") and
                   labels == ["app/utils.py:1-6"] and len(block) < 400))

    time.sleep(0.05)
    (root / "app" / "utils.py").write_text("def make_slug(text):\n    return text.strip().lower().replace(' ', '-')\n",
                                           encoding="utf-8")
    (root / "README.md").unlink()
    index.update()
    checks.append(("after edits: the new function is found, the old one and a deleted file are gone",
                   index.search("what does make_slug() do?")[0].path == "app/utils.py"
                   and not any(h.name == "slugify" for h in index.search("slugify"))
                   and "README.md" not in {r["path"] for r in index._db.execute("SELECT path FROM files")}))

    # Through ChatEngine: a project question → explain, the loader's code in the prompt
    CFG.project.index_dir = tmp / "engine-index"
    engine.projects, engine._projects_lock = {}, threading.Lock()
    engine._answering, engine._level = threading.Event(), ("free", 0.0)
    engine.memory = None
    engine.project_index(str(root))
    for _ in range(100):
        state = engine.projects and next(iter(engine.projects.values())).state
        if state and state["status"] == "ready":
            break
        time.sleep(0.1)
    turn = engine._prepare("eval-project", "where is the config loaded?", None, str(root))
    checks.append(("'where is the config loaded?' → explain with the loader's code from the project",
                   turn.intent.mode == "explain" and "Related code from the user's project" in turn.prompt
                   and "def load_config" in turn.prompt and turn.project_files[0].startswith("config/loader.py")))
    turn = engine._prepare("eval-project", "write a function that reverses a string", None, str(root))
    checks.append(("a normal code request doesn't search the project", not turn.project_files))
    return checks


def _update_checks() -> list:
    checks = []
    fixed = "def add(a, b):\n    return a + b"

    p = propose(SHOP, fixed + "\n\nprint(add(1, 2))", "python")
    checks.append(("a fixed function replaces its namesake, example lines left out",
                   p.how == "blocks" and "return a + b" in p.content and "return a - b" not in p.content
                   and p.content.count("print(add(1, 2))") == 1 and "class Shop" in p.content))

    p = propose(SHOP, "@property\ndef total(self):\n    return sum(self.items)", "python")
    checks.append(("a method is replaced inside its class, decorator and indent kept",
                   "    @property\n    def total(self):\n        return sum(self.items)" in p.content
                   and "t = t + i" not in p.content and p.content.count("@property") == 1))

    p = propose(SHOP, "import os\n\ndef add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b", "python")
    checks.append(("a new function goes after the replaced one, a missing import to the imports",
                   "import json\nimport os" in p.content and p.content.index("def sub") < p.content.index("class Shop")))

    moved = "# a new line on top\n" + SHOP
    p = propose(moved, fixed, "python", selection="def add(a, b):\n    return a - b\n", selection_line=5)
    checks.append(("the selection it read is replaced, even after it moved a line",
                   p.how == "selection" and "return a + b" in p.content and moved.count("\n") == p.content.count("\n")))

    p = propose(SHOP, fixed, "python", selection="def add(a, b):\n    return a * b\n", selection_line=5)
    checks.append(("a selection that changed since falls back to the function's name", p.how == "blocks"))

    whole = SHOP.replace("return a - b", "return a + b").replace("t = t + i", "t += i")
    p = propose(SHOP, whole, "python")
    checks.append(("a whole-file answer replaces the file", p.how == "whole" and p.content == whole))

    p = propose(SHOP, "x = compute()\ny = x * 2", "python", cursor_line=6)
    checks.append(("code with nothing to replace goes after the cursor line",
                   p.how == "insert" and p.content.split("\n")[7] == "x = compute()"))

    crlf = SHOP.replace("\n", "\r\n")
    p = propose(crlf, fixed, "python")
    checks.append(("Windows line endings are kept", "\r\n" in p.content and "\n" not in p.content.replace("\r\n", "")))

    ts = ("import { x } from './x';\n\nexport function add(a: number, b: number) {\n  return a - b;\n}\n\n"
          "function mul(a, b) {\n  if (a) {\n    return a * b;\n  }\n  return 0;\n}\n")
    p = propose(ts, "export function add(a: number, b: number) {\n  return a + b;\n}", "typescript")
    checks.append(("TypeScript: a function replaced by braces, the next one untouched",
                   p.how == "blocks" and "return a + b" in p.content and "if (a) {" in p.content
                   and p.content.count("function") == 2))

    java = ("public class Shop {\n    private int total;\n\n    public int add(int x) {\n        if (x > 0) {\n"
            "            return total - x;\n        }\n        return 0;\n    }\n\n"
            "    public int get() {\n        return total;\n    }\n}\n")
    p = propose(java, "public int add(int x) {\n    return total + x;\n}", "java")
    checks.append(("Java: a method replaced inside its class at its indent",
                   "    public int add(int x) {\n        return total + x;\n    }" in p.content
                   and "public int get()" in p.content and p.content.rstrip().endswith("}")))

    go = "package main\n\nfunc add(a int, b int) int {\n\treturn a - b\n}\n\nfunc main() {\n\tadd(1, 2)\n}\n"
    p = propose(go, "func add(a int, b int) int {\n\treturn a + b\n}", "go")
    checks.append(("Go: a func replaced", p.how == "blocks" and "return a + b" in p.content and "func main()" in p.content))

    try:
        propose(SHOP, "   \n", "python")
        checks.append(("an empty code block is refused", False))
    except ValueError:
        checks.append(("an empty code block is refused", True))
    return checks


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
