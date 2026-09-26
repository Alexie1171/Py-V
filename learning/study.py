"""
study.py — PY-V (learning/)
Study sessions (Phase 13, owner's design): "learn about <topic> for <time>".
The request itself is the permission to go online for that topic. V reads web
pages on it until the time is up and writes short notes in her own words,
saving her progress per topic (notes + sources, what's covered, what's next,
time spent — learning/store.py). A later session on the same or a related
topic, in any chat, picks up from "what's next" and links the topics.

One page = search (the topic + a subtopic — only these words go online) →
read the best page not read before → the part that matches → the brain
writes 3-6 notes + up to 3 subtopics to study next → saved.

Pacing (machine awareness, Phase 12): she waits while the user is chatting
(the brain answers the user first — a note being written is stopped and done
later) and while the laptop is busy; time keeps running (wall clock).

Run from the panel ("learn about asyncio for 30 minutes") or headless — the
big Kaggle run's study stage:
    python -m learning.study --topic "Python code" --minutes 120 --db results/study/v_study.db \\
        --export results/study/python_code.jsonl
Back on the laptop (adds the notes to V's memory database):
    python -m learning.study --import "Kaggle downloads/.../python_code.jsonl"
"""

import argparse
import logging
import os
import re
import sys
import threading
import time
from typing import Callable, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from learning import web
from learning.lookup import build_query
from learning.store import LearningStore, clean_topic

logger = logging.getLogger(__name__)

SEEDS = ["overview", "key concepts", "examples", "common mistakes", "best practices", "advanced features"]
MAX_PAGE_TRIES = 3        # pages tried per subtopic before moving on
OFFLINE_AFTER  = 3        # failed searches in a row → check the connection, stop if offline
EXCERPT_CHARS  = 3000     # of a page given to the brain (~850 tokens)
MAX_MINUTES    = 600

_UNITS = r"(?P<unit>minutes?|mins?|m|hours?|hrs?|h)"
_STUDY = [
    re.compile(rf"^\s*(?:please\s+|v,?\s+|can you\s+|could you\s+|go\s+)?(?:learn|study|research|read up)\s+(?:about\s+|on\s+|up on\s+)?"
               rf"(?P<topic>.+?)\s+for\s+(?:about\s+|around\s+)?(?P<n>\d+(?:\.\d+)?|an?|half an?)\s*{_UNITS}\b", re.I),
    re.compile(rf"^\s*(?:please\s+)?spend\s+(?P<n>\d+(?:\.\d+)?|an?|half an?)\s*{_UNITS}\s+(?:learning|studying|reading)\s+"
               rf"(?:about\s+|on\s+|up on\s+)?(?P<topic>.+?)\s*[.!]?\s*$", re.I),
]
_STOP_STUDY = re.compile(r"^\s*(?:please\s+|v,?\s+)?(?:stop|end|quit|finish|cancel)\s+(?:the\s+)?(?:studying|study(?: session)?|learning)\b", re.I)


def parse_study_request(message: str) -> Optional[Tuple[str, float]]:
    """("asyncio", 30.0) for "learn about asyncio for 30 minutes"; None for anything else."""
    for pattern in _STUDY:
        match = pattern.search(message.strip())
        if not match:
            continue
        topic = clean_topic(match.group("topic"))
        n     = match.group("n").lower()
        value = 0.5 if n.startswith("half") else 1.0 if n in ("a", "an") else float(n)
        unit  = match.group("unit").lower()
        minutes = value * 60 if unit.startswith("h") else value
        if topic and 0 < minutes:
            return topic, min(minutes, MAX_MINUTES)
    return None


def is_stop_request(message: str) -> bool:
    return bool(_STOP_STUDY.search(message))


def parse_note(raw: str) -> Tuple[Optional[str], List[str]]:
    """The brain's note → (notes as "- " lines or None, next subtopics)."""
    if not raw or "NOTHING USEFUL" in raw.upper():
        return None, []
    body, _, nxt = raw.partition("NEXT:")
    lines = []
    for line in body.split("\n"):
        line = line.strip()
        if re.match(r"^([-*•]|\d+[.)])\s+", line):
            lines.append("- " + re.sub(r"^([-*•]|\d+[.)])\s+", "", line))
        elif lines and line and not line.lower().startswith(("page", "notes", "here are")):
            lines[-1] += " " + line               # a note running onto the next line
    lines = [l for l in lines if len(l) > 8][:8]
    subtopics = []
    for item in re.split(r"[,\n;]", nxt):
        item = re.sub(r"^([-*•]|\d+[.)])\s*", "", item).strip(" .\"'")
        if 2 < len(item) <= 60:
            subtopics.append(item)
    return ("\n".join(lines) if lines else None), subtopics[:3]


class StudySession:
    """One study session. write(topic, subtopic, title, excerpt, covered, stop) → the brain's raw note, or None
    when stopped (the user needed the brain). pause() → True while she should wait (user chatting, laptop busy)."""

    def __init__(self, store: LearningStore, topic: str, minutes: float, write: Callable,
                 pause: Callable[[], bool] = None, say: Callable[[str], None] = None):
        self.store    = store
        self.topic    = clean_topic(topic)
        self.minutes  = min(max(minutes, 0.1), MAX_MINUTES)
        self.write    = write
        self.pause    = pause or (lambda: False)
        self.say      = say or (lambda text: logger.info(text))
        self.stopped  = threading.Event()
        self.writing  = threading.Event()     # set by the chat engine: stop the note being written, the user is waiting
        self.state    = "starting"
        self.notes    = 0
        self.last     = ""
        self.started  = time.time()
        self.deadline = self.started + self.minutes * 60
        self.thread   = None
        self.row      = self.store.topic(self.topic, create=True)
        self.related  = self.store.related_topics(self.topic)
        self.carried  = []                    # "what's next" from earlier, related sessions

    # ─── control ──────────────────────────────────────────────────────────────

    def start(self) -> "StudySession":
        self.thread = threading.Thread(target=self.run, daemon=True, name=f"study-{self.topic[:20]}")
        self.thread.start()
        return self

    def stop(self):
        self.stopped.set()
        self.writing.set()

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def status(self) -> dict:
        return {"running": self.running, "topic": self.topic, "minutes": round(self.minutes, 1),
                "minutes_left": max(0.0, round((self.deadline - time.time()) / 60, 1)) if self.running else 0.0,
                "notes": self.notes, "state": self.state, "last": self.last, "paused": self.state == "paused"}

    # ─── the loop ─────────────────────────────────────────────────────────────

    def plan(self) -> List[str]:
        """What to study, in order: this topic's "what's next", then related topics' next items, then the seeds."""
        t       = self.row
        covered = set(s.lower() for s in t["covered"])
        queue   = list(t["next"])
        for r in self.related:
            self.store.link(t["id"], r["id"])
            for item in r["next"][:3]:
                if item.lower() not in covered and item not in queue:
                    queue.append(item)
                    self.carried.append(item)
        if not t["covered"]:
            queue = ["overview"] + [s for s in queue if s != "overview"]
        for seed in SEEDS:
            if seed.lower() not in covered and seed not in queue:
                queue.append(seed)
        return queue

    def run(self):
        covered   = list(self.row["covered"])
        queue     = self.plan()
        saved_at  = time.time()
        failures  = 0
        tries     = {}
        first     = True
        self.say(f"Studying {self.topic} for {self.minutes:g} minutes"
                 + (f" (picking up: {', '.join(queue[:3])})" if self.row["covered"] else ""))
        try:
            while time.time() < self.deadline and not self.stopped.is_set():
                while self.pause() and time.time() < self.deadline and not self.stopped.is_set():
                    self.state = "paused"
                    self.stopped.wait(3)             # wakes at once on Stop
                if self.stopped.is_set() or time.time() >= self.deadline:
                    break
                if not queue:
                    queue = [f"{s} (more)" for s in SEEDS]
                sub = queue.pop(0)
                self.last, self.state = sub, "searching"
                words   = build_query(f"{self.topic} {sub}" if sub.lower() not in self.topic.lower() else self.topic)
                results = web.search(words)
                if not results:
                    failures += 1
                    if failures >= OFFLINE_AFTER and not web.online():
                        self.state = "offline"
                        self.say("No internet - study session stopped.")
                        break
                    continue
                failures = 0
                page = next((r for r in results if not self.store.page_read(self.row["id"], r["url"])), None)
                if page is None:
                    continue                              # read everything it found on this before
                self.state = "reading"
                doc = web.read_page(page["url"])
                if not doc:
                    self.store.mark_page(self.row["id"], page["url"], False)
                    tries[sub] = tries.get(sub, 0) + 1
                    if tries[sub] < MAX_PAGE_TRIES:
                        queue.insert(0, sub)
                    continue
                excerpt = web.best_part(doc["text"], (self.topic + " " + sub).split(), EXCERPT_CHARS)
                self.state = "writing"
                self.writing.clear()
                raw = self.write(self.topic, sub, doc["title"], excerpt, covered[-12:], self.writing)
                if raw is None:                            # stopped: the user needed the brain — this page again later
                    queue.insert(0, sub)
                    continue
                notes, following = parse_note(raw)
                self.store.mark_page(self.row["id"], page["url"], notes is not None)
                if notes:
                    self.store.add_note(self.row["id"], sub, notes, page["url"], doc["title"])
                    self.notes += 1
                    if sub not in covered:
                        covered.append(sub)
                    self.say(f"  note {self.notes}: {sub} ({doc['title'][:60]})")
                for item in following:
                    if item.lower() not in {c.lower() for c in covered} and item not in queue:
                        queue.insert(min(len(queue), 3), item)   # the page's own leads come soon
                now = time.time()
                self.store.save_progress(self.row["id"], now - saved_at, covered, queue, new_session=first)
                saved_at, first = now, False
        finally:
            self.store.save_progress(self.row["id"], time.time() - saved_at, covered, queue, new_session=first)
            if self.state not in ("offline",):
                self.state = "done"
            self.say(f"Study session on {self.topic} ended: {self.notes} notes, "
                     f"{(time.time() - self.started) / 60:.0f} min")


# ─── headless (Kaggle study stage, or a terminal) ─────────────────────────────

def _brain_writer(model, tokenizer):
    """write() for a session that owns the brain (no chat to wait for)."""
    from inference.engine.prompt_builder import build_study_prompt
    from inference.engine.generator import generate_from_prompt
    from model.training.config_loader import CFG

    def write(topic, sub, title, excerpt, covered, stop):
        prompt = build_study_prompt(topic, sub, title, excerpt, covered, tokenizer)
        text   = generate_from_prompt(model, tokenizer, prompt, mode="study", max_tokens=CFG.learning.note_tokens,
                                      temperature=0.3, formatted=True, adapter=False, stop=stop)
        return None if stop.is_set() else (text or None)
    return write


def main():
    parser = argparse.ArgumentParser(description="V's study session without the panel (Kaggle stage / terminal)")
    parser.add_argument("--topic")
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--db", default=None, help="study database (default: V's memory database)")
    parser.add_argument("--export", default=None, help="write the topic + notes as JSON lines here when done")
    parser.add_argument("--import", dest="import_path", default=None,
                        help="add exported notes (e.g. from the Kaggle run) to V's memory database, then stop")
    args = parser.parse_args()

    from model.training.config_loader import CFG
    store = LearningStore(args.db or CFG.memory.db_path, embed=_embed_or_none())
    if args.import_path:
        added = store.import_topics(args.import_path)
        print(f"Imported {added} notes from {args.import_path}")
        return
    if not args.topic:
        parser.error("--topic is required (or --import)")

    if not web.online():
        print("No internet - a study session needs it (Kaggle: Settings > Internet on).")
        sys.exit(2)
    from inference.engine.model_loader import load_model
    model, tokenizer = load_model()          # notes are words: the plain brain, no adapter
    model.eval()
    session = StudySession(store, args.topic, args.minutes, _brain_writer(model, tokenizer), say=print)
    session.run()
    if args.export:
        os.makedirs(os.path.dirname(os.path.abspath(args.export)), exist_ok=True)
        count = store.export_topics(args.export, [session.row["id"]])
        print(f"Exported {count} notes on {session.topic} to {args.export}")
    if session.notes == 0:
        sys.exit(1)


def _embed_or_none():
    try:
        from retrieval.embedder import cpu_embedder
        encoder = cpu_embedder()
        return encoder.encode
    except Exception as e:
        logger.warning(f"No embedder ({e}) - keyword search only")
        return None


if __name__ == "__main__":
    main()
