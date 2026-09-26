"""
manager.py — PY-V (learning/)
LearningManager — what the chat engine and the API use for Phase 13:
  start_study() / stop_study() / study_status() / yield_brain()   study sessions (one at a time)
  lookup()                  a web lookup the user said yes to → what she found + sources
  recall_notes()            study notes relevant to a question (added to the prompt)
  topics() / overview()     what she has studied (memory view, "what have you learned?")
  approve() / unapprove()   "Good answer" → training examples (learning/export.py)

Nothing here changes the brain. Settings: config learning:.
"""

import logging
import random
import threading
from typing import Callable, List, Optional

from model.training.config_loader import CFG
from learning import web
from learning.lookup import build_query
from learning.store import LearningStore
from learning.study import StudySession, clean_topic

logger = logging.getLogger(__name__)


def _embed_fn():
    """The shared CPU embedder's encode(), loaded on first use; None → keyword search only."""
    state = {"encoder": None, "failed": False}

    def embed(texts):
        if state["failed"]:
            return None
        if state["encoder"] is None:
            try:
                from retrieval.embedder import cpu_embedder
                state["encoder"] = cpu_embedder()
            except Exception as e:
                logger.warning(f"Learning: embedding model unavailable ({e}) - keyword search only")
                state["failed"] = True
                return None
        return state["encoder"].encode([t[:1200] for t in texts])
    return embed


class LearningManager:

    def __init__(self, db_path=None, embed: Callable = None):
        self.cfg     = CFG.learning
        self.store   = LearningStore(db_path or CFG.memory.db_path, embed if embed is not None else _embed_fn())
        self.session: Optional[StudySession] = None
        self._lock   = threading.Lock()

    # ─── study sessions ───────────────────────────────────────────────────────

    def start_study(self, topic: str, minutes: float, write: Callable, pause: Callable[[], bool]) -> str:
        """Start a session in the background; returns what V says about it (no brain needed)."""
        topic = clean_topic(topic)
        with self._lock:
            if self.session and self.session.running:
                s = self.session.status()
                return (f"I'm already studying {s['topic']} ({s['minutes_left']:g} min left). "
                        "Say \"stop studying\" first if you want me to switch.")
            if not self.cfg.web:
                return "My web access is switched off (config learning.web), so I can't study online right now."
            minutes = min(minutes, self.cfg.study_max_minutes)
            session = StudySession(self.store, topic, minutes, write, pause)
            self.session = session.start()
        return self._study_reply(session, minutes)

    def _study_reply(self, session: StudySession, minutes: float) -> str:
        row, when = session.row, f"{minutes:g} minute{'s' if minutes != 1 else ''}"
        if minutes >= 60 and minutes % 60 == 0:
            when = f"{minutes / 60:g} hour{'s' if minutes != 60 else ''}"
        if row["covered"]:
            text = (f"Back to {row['topic']} for {when}. Last time ({row['minutes']:g} min) I covered "
                    f"{_list(row['covered'][-3:])}, so I'll pick up with {_list(row['next'][:2]) or 'the next parts'}.")
        else:
            text = random.choice([f"Okay, I'll study {row['topic']} for {when}.",
                                  f"Sure, {row['topic']} for {when} it is.",
                                  f"On it: {when} of {row['topic']}."])
        if session.related:
            r = session.related[0]
            text += f" It ties in with {r['topic']}, which I studied before, so I'll build on that."
        return text + " You can keep chatting; I pause while I answer you."

    def stop_study(self) -> str:
        with self._lock:
            session = self.session
        if not session or not session.running:
            return "I'm not studying anything right now."
        session.stop()
        return f"Okay, I stopped studying {session.topic}. I keep what I learned ({session.notes} notes so far)."

    def study_status(self) -> Optional[dict]:
        session = self.session
        return session.status() if session else None

    def yield_brain(self):
        """The user is waiting: stop the study note being written (done again later)."""
        session = self.session
        if session and session.running:
            session.writing.set()

    # ─── web lookup ───────────────────────────────────────────────────────────

    def lookup(self, query: str, question: str) -> dict:
        """{"found": page text for the prompt, "sources": [{"title", "url"}]} — empty when offline / nothing found."""
        if not self.cfg.web or not query.strip():
            return {"found": "", "sources": []}
        results = web.search(query)
        words   = (query + " " + build_query(question)).split()
        parts, sources = [], []
        per_page = max(600, self.cfg.lookup_chars // max(1, self.cfg.lookup_pages))
        for r in results:
            doc = web.read_page(r["url"])
            if not doc:
                continue
            parts.append(f"[{doc['title']}]\n{web.best_part(doc['text'], words, per_page)}")
            sources.append({"title": doc["title"], "url": doc["url"]})
            if len(sources) >= self.cfg.lookup_pages:
                break
        if not parts and results:     # pages wouldn't load: the search snippets at least
            for r in results[:4]:
                if r["snippet"]:
                    parts.append(f"[{r['title']}] {r['snippet']}")
                    sources.append({"title": r["title"], "url": r["url"]})
        return {"found": "\n\n".join(parts)[:self.cfg.lookup_chars], "sources": sources}

    # ─── what she knows ───────────────────────────────────────────────────────

    def recall_notes(self, query: str, top_k: int = None) -> List[dict]:
        k = self.cfg.notes_top_k if top_k is None else top_k
        if k <= 0:
            return []
        try:
            return self.store.search_notes(query, k)
        except Exception as e:
            logger.warning(f"Study notes search failed: {e}")
            return []

    def topics(self) -> List[dict]:
        return self.store.topics()

    def overview(self) -> dict:
        return {"topics": self.store.topics(), "approved": self.store.approved_count(), "study": self.study_status()}

    # ─── approved answers ─────────────────────────────────────────────────────

    def approve(self, question: str, answer: str, mode: str = None, language: str = None, session_id: str = None) -> str:
        return self.store.approve_answer(question, answer, mode, language, session_id)

    def unapprove(self, answer_id: str) -> bool:
        return self.store.unapprove_answer(answer_id)


def _list(items: List[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + " and " + items[-1]
