"""
eval_learning.py — PY-V (experiments/)
Phase 13 learning test — instant, no brain, no internet (the web is faked).

  yes / no     "Want me to look that up online?" answered in many wordings
  lookups      "look up X", "look it up" (the previous question), when she offers one
  study        "learn about <topic> for <time>" in many wordings, note parsing,
               a whole session with a fake web: notes, progress, what's next,
               a later related session picking up and linking the topics
  store        notes search, approved answers, export → training records,
               export / import between machines (Kaggle → laptop)
  chat flow    through ChatEngine with a stand-in brain: she doesn't know →
               offers → "yeah" → looks it up (sources, prompt holds what she
               found); "nah" → no lookup; a study request and "stop studying"
               need no brain

Usage (from repo root):
    python -m experiments.eval_learning
"""

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.training.config_loader import CFG

CFG.memory.enabled  = False
CFG.machine.enabled = False

from learning import web
from learning.export import answer_records, topic_records
from learning.lookup import answer_kind, asked_to_look_up, build_query, offer_lookup
from learning.store import LearningStore
from learning.study import StudySession, is_stop_request, parse_note, parse_study_request

YES = ["yes", "Yes!", "yea", "yeah", "yeah sure", "yep", "yup", "sure", "sure thing", "ok", "okay go ahead",
       "go ahead", "go for it", "do it", "please", "yes please", "yeah look it up", "look it up", "search it",
       "google it", "of course", "why not", "alright", "y", "k", "absolutely", "sounds good"]
NO = ["no", "No.", "nah", "nope", "no thanks", "no thank you", "not now", "don't", "dont bother", "never mind",
      "nevermind", "skip it", "leave it", "forget it", "no need", "i'm good", "n", "don't look it up",
      "no, don't search"]
NEITHER = ["please explain more about generators", "what about lists?", "how does it work", "tell me a joke",
           "write a function that sorts a list", "kotlin is nice", "node.js question", "yes but first explain what "
           "a decorator is and then show me how the wrapper function gets called"]

STUDY = [("learn about asyncio for 30 minutes", ("asyncio", 30)), ("study rust for 2 hours", ("rust", 120)),
         ("learn python decorators for an hour", ("python decorators", 60)),
         ("can you learn about sql joins for half an hour", ("sql joins", 30)),
         ("spend 45 minutes learning about numpy", ("numpy", 45)),
         ("Learn about Python code for 2 hours", ("Python code", 120)),
         ("please study the rust borrow checker for 20 min", ("the rust borrow checker", 20)),
         ("I want to learn python", None), ("learn about decorators", None), ("what did you learn?", None),
         ("explain asyncio for beginners", None)]

NOTE = """- asyncio runs coroutines on an event loop in one thread.
- `await` hands control back to the loop until the result is ready.
- asyncio.gather runs several coroutines at once and
  collects their results in order.
NEXT: tasks and cancellation, asyncio.Queue, timeouts"""


def fake_web(pages: dict):
    """search / read_page / online that never go online. pages: url → text."""
    def search(query, max_results=6):
        return [{"title": f"Page about {query}", "url": url, "snippet": text[:80]} for url, text in pages.items()]

    def read_page(url, max_chars=20000):
        return {"url": url, "title": url.rsplit("/", 1)[-1], "text": pages[url]} if url in pages else None

    web.search, web.read_page, web.online = search, read_page, lambda: True


def main():
    checks = []
    add = lambda name, ok: checks.append((name, bool(ok)))

    # ─── yes / no ─────────────────────────────────────────────────────────────
    add(f"yes in {len(YES)} wordings", all(answer_kind(t) == "yes" for t in YES))
    misses = [t for t in YES if answer_kind(t) != "yes"] + [t for t in NO if answer_kind(t) != "no"] + \
             [t for t in NEITHER if answer_kind(t) is not None]
    add(f"no in {len(NO)} wordings", all(answer_kind(t) == "no" for t in NO))
    add(f"{len(NEITHER)} other messages are neither", all(answer_kind(t) is None for t in NEITHER))
    if misses:
        print("  yes/no misses:", misses)

    # ─── lookups ──────────────────────────────────────────────────────────────
    add("'look up the latest pandas version' → its words",
        asked_to_look_up("look up the latest pandas version") == "latest pandas version")
    add("'search the web for rust async traits'", asked_to_look_up("search the web for rust async traits") == "rust async traits")
    add("'can you look it up?' → the previous question",
        asked_to_look_up("can you look it up?", "What is the newest version of Zig?") == "newest version Zig")
    add("'google python 3.14 release date'", asked_to_look_up("google python 3.14 release date") == "python 3.14 release date")
    add("a normal question is not a lookup request", asked_to_look_up("how do I sort a list?") is None)
    add("offers when she doesn't know", offer_lookup("who maintains the foo library?", "I'm not sure who maintains it.", "chat"))
    add("offers on questions that need new facts", offer_lookup("what is the latest version of numpy?", "It is 1.26.", "explain"))
    add("no offer on normal answers", not offer_lookup("what is the current directory in python?",
                                                        "It's the folder your program runs in.", "explain"))
    add("no offer in code modes", not offer_lookup("latest numpy version?", "I don't know", "generate"))
    add("search words: no code, no chat words",
        build_query("Can you tell me what ```python\nx = 1\n``` the newest Django release is?") == "newest Django release")

    # ─── study requests and notes ─────────────────────────────────────────────
    misses = [(m, parse_study_request(m), want) for m, want in STUDY if parse_study_request(m) != want]
    add(f"study requests ({len(STUDY)} wordings)", not misses)
    if misses:
        print("  study misses:", misses)
    add("'stop studying' / 'stop the study session'", is_stop_request("stop studying") and
        is_stop_request("please stop the study session") and not is_stop_request("stop the server"))
    notes, following = parse_note(NOTE)
    add("a note: 3 lines (a wrapped line joined), 3 next subtopics",
        notes and notes.count("\n- ") == 2 and "collects their results" in notes
        and following == ["tasks and cancellation", "asyncio.Queue", "timeouts"])
    add("'NOTHING USEFUL' → no note", parse_note("NOTHING USEFUL") == (None, []))

    # ─── a whole study session (fake web) + the store ─────────────────────────
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:   # SQLite files stay open on Windows
        store = LearningStore(Path(tmp) / "v.db", embed=None)
        pages = {f"https://example.com/asyncio-{i}": ("asyncio event loop coroutines await tasks. " * 30) for i in range(40)}
        fake_web(pages)
        written = []

        def write(topic, sub, title, excerpt, covered, stop):
            written.append(sub)
            return NOTE

        session = StudySession(store, "asyncio", 0.02, write, say=lambda t: None)
        session.run()
        t = store.topic("asyncio")
        add("the session wrote notes with sources and saved its time",
            session.notes >= 3 and len(store.notes(t["id"])) == session.notes and t["minutes"] > 0
            and all(n["url"] for n in store.notes(t["id"])))
        add("covered starts with the overview; the page's leads are next",
            t["covered"][0] == "overview" and "tasks and cancellation" in t["covered"] + t["next"])
        add("a page is never read twice", len(set(n["url"] for n in store.notes(t["id"]))) == session.notes)

        again = StudySession(store, "asyncio tasks", 0.01, write, say=lambda t: None)
        add("a later session on a related topic finds the earlier one", any(r["topic"] == "asyncio" for r in again.related))
        again.run()
        add("…links them and carries on with its 'what's next'",
            any(x["topic"] == "asyncio" for x in store.linked(again.row["id"])) and bool(again.carried))

        hits = store.search_notes("how does the asyncio event loop run coroutines")
        add("notes found for a related question (keywords)", hits and hits[0]["topic"] in ("asyncio", "asyncio tasks"))
        add("no notes for an unrelated question", not store.search_notes("how do I bake bread"))

        a1 = store.approve_answer("write a function that adds", "def add(a, b):\n    return a + b", "generate")
        store.approve_answer("hi", "Hey!", "chat")
        store.approve_answer("add in rust", "fn add() {}", "generate", "Rust")
        records, skipped = answer_records(store)
        add("approved answers → training records (chat / other languages left out)",
            len(records) == 1 and skipped == 2 and records[0]["metadata"]["task"] == "generate")
        add("unapprove removes it", store.unapprove_answer(a1) and store.approved_count() == 2)
        add("study notes train only after the topic is approved", topic_records(store) == [])
        store.approve_topic(t["id"], True)
        topics = topic_records(store)
        add("…approved topic → explain records in sentences",
            topics and topics[0]["metadata"]["task"] == "explain" and "- " not in topics[0]["output"])

        path = Path(tmp) / "export.jsonl"
        count = store.export_topics(path, [t["id"]])
        other = LearningStore(Path(tmp) / "laptop.db", embed=None)
        added = other.import_topics(path)
        mine  = other.topic("asyncio")
        add("export → import (Kaggle → laptop): notes, covered, next, time",
            count == added == session.notes and mine["covered"] == t["covered"] and mine["minutes"] > 0)
        add("importing twice adds nothing", other.import_topics(path) == 0)

        # ─── the chat flow (stand-in brain, fake web) ─────────────────────────
        checks += chat_flow(Path(tmp))

    for name, ok in checks:
        print(f"  {'ok  ' if ok else 'MISS'} {name}")
    passed = sum(ok for _, ok in checks)
    print(f"\nLEARNING score: {passed}/{len(checks)}")
    return passed == len(checks)


def chat_flow(tmp: Path) -> list:
    import inference.engine.chat as chat_mod
    from inference.engine.context_manager import ContextManager
    from inference.engine.controller import Controller
    from learning.manager import LearningManager

    checks  = []
    add     = lambda name, ok: checks.append((name, bool(ok)))
    prompts = []
    answers = iter(["I'm not sure, I don't have information about the newest Zig release.",
                    "Zig 0.14 is the newest release, from March.",
                    "Generators produce values lazily."])

    def fake_generate(model, tokenizer, prompt, **kw):
        prompts.append(prompt)
        return next(answers)

    class Tok:
        chat_template = "stand-in"

        def apply_chat_template(self, messages, **kw):
            return "\n".join(f"[{m['role']}] {m['content']}" for m in messages)

    class Brain:
        v_prompt_format = "native_chat"

    fake_web({"https://ziglang.org/news": "Zig 0.14.0 was released in March. The newest Zig release brings ... " * 20})
    chat_mod.generate_from_prompt = fake_generate
    engine = chat_mod.ChatEngine.__new__(chat_mod.ChatEngine)
    engine.controller, engine.memory, engine.machine, engine.retriever = Controller(), None, None, None
    engine.context_manager = ContextManager(None)
    engine.learning = LearningManager(tmp / "chat.db", embed=None)
    engine.model, engine.tokenizer = Brain(), Tok()
    engine.projects, engine._projects_lock = {}, __import__("threading").Lock()
    engine._answering, engine._last_user, engine._level = __import__("threading").Event(), 0.0, ("free", 0.0)

    r1 = engine.chat("s1", "What is the newest Zig release?")
    add("she doesn't know → asks 'Want me to look that up online?' with quick replies",
        r1["response"].endswith(chat_mod._OFFER) and r1["asks"] == chat_mod._ASKS)
    r2 = engine.chat("s1", "yeah")
    add("'yeah' → she looks it up: sources shown, what she found is in her prompt",
        r2["sources"] and r2["sources"][0]["url"] == "https://ziglang.org/news" and "Zig 0.14.0" in prompts[-1]
        and "What is the newest Zig release?" in prompts[-1] and "Want me" not in r2["response"])
    answers_no = iter(["I'm not sure about that one."])
    chat_mod.generate_from_prompt = lambda *a, **k: next(answers_no)
    r3 = engine.chat("s3", "who maintains the foo library?")
    r4 = engine.chat("s3", "nah")
    add("'nah' → no lookup, no brain", r4["flags"] == ["no_brain"] and r4["response"] in chat_mod._LOOKUP_NO
        and engine.context_manager.load("s3").pending_lookup is None)

    def no_brain(*a, **k):
        raise AssertionError("the brain was used")

    chat_mod.generate_from_prompt = no_brain
    fake_web({f"https://example.com/p{i}": "numpy arrays broadcasting dtype " * 40 for i in range(5)})
    r5 = engine.chat("s4", "learn about numpy for 1 minute")
    add("a study request starts a session without the brain",
        r5["flags"] == ["no_brain"] and "numpy" in r5["response"] and r5["study"]["running"])
    r6 = engine.chat("s4", "stop studying")
    time.sleep(0.5)
    add("'stop studying' stops it", "stopped" in r6["response"].lower() and not engine.learning.study_status()["running"])
    return checks


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
