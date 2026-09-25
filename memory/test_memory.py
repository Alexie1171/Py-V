"""
test_memory.py — PY-V (memory/)
Smoke test for Phase 11 memory, no brain needed: fact tagging, newest-wins,
keyword + meaning search (embedder on the CPU), code only in code modes, the
prompt block and its budget, forgetting, session state through the
ContextManager. Uses a throwaway database.

Usage (from repo root):
    python -m memory.test_memory
"""

import dataclasses
import shutil
import tempfile
from pathlib import Path

from model.training.config_loader import CFG
from memory.extractor import extract_facts
from memory.manager import MemoryManager
from inference.engine.context_manager import ContextManager
from inference.engine.prompt_builder import build_prompt, format_memories

FAILED = []


def check(name: str, ok: bool, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def test_extractor():
    print("extractor")
    keys = lambda text: dict(extract_facts(text))
    check("name", keys("Hi, my name is Sam.").get("user:name") == "The user's name is Sam.")
    check("name at sentence start", keys("My name is Sam.").get("user:name") == "The user's name is Sam.")
    check("questions are not facts", keys("What is my name?") == {})
    check("remember", any(k.startswith("note:") for k in keys("Remember that the API runs on port 8000.")))
    check("preference", keys("I prefer tabs over spaces.").get("preference:tabs-over-spaces")
          == "The user prefers tabs over spaces.")
    check("decision", any(k.startswith("decision:") for k in keys("We decided to use Granite as the brain.")))
    check("laptop", "user:laptop" in keys("My laptop has a GTX 1650 with 4 GB."))
    mixed = keys("I get TypeError: unsupported operand in utils/helpers.py with Python 3.13")
    check("error + file + version", {"error:typeerror", "file:utils/helpers.py", "version:python"} <= set(mixed),
          sorted(mixed))


def test_memory(tmp: Path, semantic: bool):
    print(f"memory ({'keyword + meaning' if semantic else 'keyword only'} search)")
    cfg = dataclasses.replace(CFG.memory, db_path=tmp / f"mem_{semantic}.db", semantic_search=semantic)
    mem = MemoryManager(cfg)

    mem.remember_turn("s1", "chat", "My name is Sam.", "Nice to meet you, Sam!")
    mem.remember_turn("s1", "chat", "Actually my name is Alex.", "Got it, Alex.")
    names = [f.text for f in mem.list_facts() if f.key == "user:name"]
    check("newest fact wins", names == ["The user's name is Alex."], names)
    check("older fact kept inactive", len([f for f in mem.store.facts(active_only=False) if f.key == "user:name"]) == 2)

    hits = mem.recall("what is my name", "chat")["facts"]
    check("recall by keyword", bool(hits) and "Alex" in hits[0].text, [h.text for h in hits])

    mem.remember_turn("s2", "chat", "My laptop has a GTX 1650 with 4 GB of memory.", "Noted.")
    if semantic:
        hits = mem.recall("which graphics card do I own", "chat")["facts"]
        check("recall by meaning (no shared words)", any("1650" in h.text for h in hits), [h.text for h in hits])
        check("embedder runs on the CPU", str(mem._encoder.model.device) == "cpu", mem._encoder.model.device)

    code = "```python\ndef add(a, b):\n    return a + b\n```"
    mem.remember_turn("s3", "generate", "Write a function to add two numbers", code)
    check("code recalled in code modes", bool(mem.recall("add two numbers function", "generate")["code"]))
    check("never code in chat mode", mem.recall("add two numbers function", "chat")["code"] == [])

    gen  = format_memories(mem.recall("add two numbers function", "generate"), "generate")
    chat = format_memories(mem.recall("add two numbers function", "generate"), "chat")
    check("prompt block: code in generate", "def add" in gen, gen)
    check("prompt block: no code in chat", "def add" not in chat, chat)
    prompt = build_prompt("chat", "what is my name", {}, memories=mem.recall("what is my name", "chat"))
    check("block goes into the template's context slot", "Things V remembers" in prompt and "Alex" in prompt)
    check("no memory → template unchanged", build_prompt("chat", "hi", {}) == build_prompt("chat", "hi", {}, memories={}))

    for i in range(40):
        mem.store.add_fact(f"note:long-{i}", "A long remembered sentence about the project " * 3 + str(i))
    block = format_memories({"facts": mem.search.facts("long remembered sentence project", 40)}, "chat")
    check("block stays within the token budget", len(block) <= CFG.memory.max_prompt_tokens * 4 + 1, len(block))

    fact = mem.list_facts()[0]
    check("forget", mem.forget(fact.id) and fact.id not in [f.id for f in mem.list_facts()])

    cm = ContextManager(mem)
    ctx = cm.load("s9")
    cm.append_history(ctx, "hello", "hi there", "chat")
    cm.update(ctx, mode="debug", current_task="fix the parser")
    again = ContextManager(MemoryManager(cfg)).load("s9")
    check("session state survives a restart", again.mode == "debug" and again.current_task == "fix the parser"
          and [t.content for t in again.history] == ["hello", "hi there"])


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        test_extractor()
        test_memory(tmp, semantic=False)
        test_memory(tmp, semantic=True)
        print("\nContextManager without memory:")
        cm = ContextManager(None)
        ctx = cm.load("x")
        cm.append_history(ctx, "a", "b")
        check("chat works without memory", [t.content for t in cm.load("x").history] == ["a", "b"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASSED' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")


if __name__ == "__main__":
    main()
