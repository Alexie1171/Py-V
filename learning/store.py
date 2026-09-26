"""
store.py — PY-V (learning/)
What V learned (Phase 13), in the same local SQLite file as her memory
(config memory.db_path — private, gitignored, never committed), own tables:

  topics     one per study topic: time spent, what's covered, what's next,
             whether the owner approved it for training
  links      related topics ("asyncio" ↔ "python async tasks")
  notes      short notes in V's own words, each with its source (address,
             title, date) — FTS5 keyword index + meaning vectors
  pages      addresses already read per topic (not read twice)
  approved   chat answers the owner marked "Good answer" — training examples
             for the next Kaggle retrain (learning/export.py)

Nothing here changes the brain; training stays a Kaggle run the owner starts.
"""

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np

from memory.store import query_words, text_words

_SCHEMA = """
CREATE TABLE IF NOT EXISTS study_topics (
    id INTEGER PRIMARY KEY, topic TEXT NOT NULL UNIQUE, created REAL NOT NULL, updated REAL NOT NULL,
    seconds REAL NOT NULL DEFAULT 0, covered TEXT NOT NULL DEFAULT '[]', next TEXT NOT NULL DEFAULT '[]',
    approved INTEGER NOT NULL DEFAULT 0, sessions INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS study_links (a INTEGER NOT NULL, b INTEGER NOT NULL, PRIMARY KEY (a, b));
CREATE TABLE IF NOT EXISTS study_notes (
    id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, subtopic TEXT NOT NULL, text TEXT NOT NULL,
    url TEXT, title TEXT, created REAL NOT NULL);
CREATE INDEX IF NOT EXISTS study_notes_by_topic ON study_notes(topic_id);
CREATE TABLE IF NOT EXISTS study_note_vectors (note_id INTEGER PRIMARY KEY, vector BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS study_pages (
    topic_id INTEGER NOT NULL, url TEXT NOT NULL, read REAL NOT NULL, useful INTEGER NOT NULL,
    PRIMARY KEY (topic_id, url));
CREATE TABLE IF NOT EXISTS approved_answers (
    id TEXT PRIMARY KEY, session_id TEXT, question TEXT NOT NULL, answer TEXT NOT NULL, mode TEXT,
    language TEXT, created REAL NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS study_notes_fts USING fts5(text, subtopic, content='study_notes', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS study_notes_ai AFTER INSERT ON study_notes BEGIN
    INSERT INTO study_notes_fts(rowid, text, subtopic) VALUES (new.id, new.text, new.subtopic); END;
CREATE TRIGGER IF NOT EXISTS study_notes_ad AFTER DELETE ON study_notes BEGIN
    INSERT INTO study_notes_fts(study_notes_fts, rowid, text, subtopic) VALUES ('delete', old.id, old.text, old.subtopic); END;
"""

MIN_SIMILARITY = 0.62   # a note / topic close enough in meaning (bge-small)
MIN_WORD_SHARE = 0.5


def clean_topic(topic: str) -> str:
    return re.sub(r"\s+", " ", topic).strip(" .,:;!?\"'").strip()


class LearningStore:

    def __init__(self, db_path, embed=None):
        """embed: callable(texts) → normalized vectors (the shared CPU embedder), or None = keyword search only."""
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embed = embed
        self._lock = threading.Lock()
        self._db   = sqlite3.connect(str(db_path), check_same_thread=False, timeout=15)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    # ─── topics ───────────────────────────────────────────────────────────────

    def topic(self, topic: str, create: bool = False) -> Optional[dict]:
        topic = clean_topic(topic)
        with self._lock:
            row = self._db.execute("SELECT * FROM study_topics WHERE lower(topic) = lower(?)", (topic,)).fetchone()
            if row is None and create:
                now = time.time()
                self._db.execute("INSERT INTO study_topics (topic, created, updated) VALUES (?, ?, ?)", (topic, now, now))
                self._db.commit()
                row = self._db.execute("SELECT * FROM study_topics WHERE lower(topic) = lower(?)", (topic,)).fetchone()
        return self._topic(row) if row else None

    def topic_by_id(self, topic_id: int) -> Optional[dict]:
        with self._lock:
            row = self._db.execute("SELECT * FROM study_topics WHERE id = ?", (topic_id,)).fetchone()
        return self._topic(row) if row else None

    def topics(self) -> List[dict]:
        with self._lock:
            rows  = self._db.execute("SELECT * FROM study_topics ORDER BY updated DESC").fetchall()
            count = dict(self._db.execute("SELECT topic_id, COUNT(*) FROM study_notes GROUP BY topic_id").fetchall())
        out = []
        for r in rows:
            t = self._topic(r)
            t["notes"] = count.get(t["id"], 0)
            out.append(t)
        return out

    def save_progress(self, topic_id: int, seconds: float, covered: List[str], next_up: List[str], new_session=False):
        with self._lock:
            self._db.execute(
                "UPDATE study_topics SET seconds = seconds + ?, covered = ?, next = ?, updated = ?, "
                "sessions = sessions + ? WHERE id = ?",
                (seconds, json.dumps(covered[-200:]), json.dumps(next_up[:50]), time.time(), 1 if new_session else 0,
                 topic_id))
            self._db.commit()

    def approve_topic(self, topic_id: int, approved: bool) -> bool:
        with self._lock:
            cur = self._db.execute("UPDATE study_topics SET approved = ? WHERE id = ?", (1 if approved else 0, topic_id))
            self._db.commit()
            return cur.rowcount > 0

    def forget_topic(self, topic_id: int) -> bool:
        with self._lock:
            ids = [r[0] for r in self._db.execute("SELECT id FROM study_notes WHERE topic_id = ?", (topic_id,))]
            if ids:
                marks = ",".join("?" * len(ids))
                self._db.execute(f"DELETE FROM study_note_vectors WHERE note_id IN ({marks})", ids)
                self._db.execute(f"DELETE FROM study_notes WHERE id IN ({marks})", ids)
            self._db.execute("DELETE FROM study_pages WHERE topic_id = ?", (topic_id,))
            self._db.execute("DELETE FROM study_links WHERE a = ? OR b = ?", (topic_id, topic_id))
            cur = self._db.execute("DELETE FROM study_topics WHERE id = ?", (topic_id,))
            self._db.commit()
            return cur.rowcount > 0

    def link(self, a: int, b: int):
        if a == b:
            return
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO study_links (a, b) VALUES (?, ?)", (min(a, b), max(a, b)))
            self._db.commit()

    def linked(self, topic_id: int) -> List[dict]:
        with self._lock:
            ids = [r[0] for r in self._db.execute(
                "SELECT CASE WHEN a = ? THEN b ELSE a END FROM study_links WHERE a = ? OR b = ?",
                (topic_id, topic_id, topic_id))]
        return [t for t in (self.topic_by_id(i) for i in ids) if t]

    def related_topics(self, topic: str, limit: int = 3) -> List[dict]:
        """Earlier topics close to this one: shared words ("python asyncio" ~ "asyncio tasks") or close meaning."""
        topic  = clean_topic(topic)
        wanted = set(query_words(topic))
        scored = []
        others = [t for t in self.topics() if t["topic"].lower() != topic.lower()]
        vectors = self.embed([topic] + [t["topic"] for t in others]) if (self.embed and others) else None
        for i, t in enumerate(others):
            share = len(wanted & set(query_words(t["topic"]))) / len(wanted) if wanted else 0
            sim   = float(vectors[i + 1] @ vectors[0]) if vectors is not None else 0
            if share >= MIN_WORD_SHARE or sim >= 0.75:
                scored.append((share + sim, t))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [t for _, t in scored[:limit]]

    @staticmethod
    def _topic(r) -> dict:
        return {"id": r["id"], "topic": r["topic"], "created": r["created"], "updated": r["updated"],
                "minutes": round(r["seconds"] / 60, 1), "covered": json.loads(r["covered"]),
                "next": json.loads(r["next"]), "approved": bool(r["approved"]), "sessions": r["sessions"]}

    # ─── pages and notes ──────────────────────────────────────────────────────

    def page_read(self, topic_id: int, url: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM study_pages WHERE topic_id = ? AND url = ?",
                                    (topic_id, url)).fetchone() is not None

    def mark_page(self, topic_id: int, url: str, useful: bool):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO study_pages (topic_id, url, read, useful) VALUES (?, ?, ?, ?)",
                             (topic_id, url, time.time(), 1 if useful else 0))
            self._db.commit()

    def add_note(self, topic_id: int, subtopic: str, text: str, url: str = None, title: str = None) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO study_notes (topic_id, subtopic, text, url, title, created) VALUES (?, ?, ?, ?, ?, ?)",
                (topic_id, subtopic, text, url, title, time.time()))
            self._db.commit()
            note_id = cur.lastrowid
        if self.embed is not None:
            vectors = self.embed([f"{subtopic}\n{text}"[:1200]])
            if vectors is not None:
                with self._lock:
                    self._db.execute("INSERT OR REPLACE INTO study_note_vectors (note_id, vector) VALUES (?, ?)",
                                     (note_id, np.asarray(vectors[0], dtype=np.float32).tobytes()))
                    self._db.commit()
        return note_id

    def notes(self, topic_id: int) -> List[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM study_notes WHERE topic_id = ? ORDER BY id", (topic_id,)).fetchall()
        return [dict(r) for r in rows]

    def search_notes(self, query: str, top_k: int = 3) -> List[dict]:
        """Notes relevant to a question: half its meaningful words, or close meaning. Best first."""
        wanted = set(query_words(query))
        scores = {}
        with self._lock:
            if wanted:
                try:
                    rows = self._db.execute(
                        "SELECT n.id, n.text, n.subtopic FROM study_notes_fts f JOIN study_notes n ON n.id = f.rowid "
                        "WHERE study_notes_fts MATCH ? LIMIT 50", (" OR ".join(f'"{w}"' for w in wanted),)).fetchall()
                except sqlite3.OperationalError:
                    rows = []
                for r in rows:
                    share = len(wanted & text_words(r["text"] + " " + r["subtopic"])) / len(wanted)
                    if share >= MIN_WORD_SHARE:
                        scores[r["id"]] = 0.5 * share
            vec_rows = self._db.execute("SELECT note_id, vector FROM study_note_vectors").fetchall() if self.embed else []
        if vec_rows:
            q = self.embed([query])
            if q is not None:
                matrix = np.frombuffer(b"".join(r["vector"] for r in vec_rows), dtype=np.float32).reshape(len(vec_rows), -1)
                for r, sim in zip(vec_rows, matrix @ np.asarray(q[0], dtype=np.float32)):
                    if sim >= MIN_SIMILARITY:
                        scores[r["note_id"]] = scores.get(r["note_id"], 0) + 0.5 * float(sim)
        best = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out  = []
        with self._lock:
            for note_id, score in best:
                r = self._db.execute("SELECT n.*, t.topic FROM study_notes n JOIN study_topics t ON t.id = n.topic_id "
                                     "WHERE n.id = ?", (note_id,)).fetchone()
                if r:
                    out.append({**dict(r), "score": round(score, 3)})
        return out

    # ─── approved chat answers ────────────────────────────────────────────────

    def approve_answer(self, question: str, answer: str, mode: str = None, language: str = None,
                       session_id: str = None) -> str:
        answer_id = uuid.uuid4().hex[:16]
        with self._lock:
            self._db.execute("INSERT INTO approved_answers (id, session_id, question, answer, mode, language, created) "
                             "VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (answer_id, session_id, question, answer, mode, language, time.time()))
            self._db.commit()
        return answer_id

    def unapprove_answer(self, answer_id: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM approved_answers WHERE id = ?", (answer_id,))
            self._db.commit()
            return cur.rowcount > 0

    def approved_answers(self) -> List[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT * FROM approved_answers ORDER BY created").fetchall()]

    def approved_count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM approved_answers").fetchone()[0]

    # ─── moving notes between machines (Kaggle study session → laptop) ────────

    def export_topics(self, path, topic_ids: List[int] = None) -> int:
        """Topics + their notes as JSON lines (one topic per line). Returns how many notes."""
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for t in self.topics():
                if topic_ids and t["id"] not in topic_ids:
                    continue
                notes = self.notes(t["id"])
                count += len(notes)
                f.write(json.dumps({"topic": t, "notes": [{k: n[k] for k in ("subtopic", "text", "url", "title", "created")}
                                                          for n in notes]}, ensure_ascii=False) + "\n")
        return count

    def import_topics(self, path) -> int:
        """Merge exported topics into this store (notes already there are skipped). Returns notes added."""
        added = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item  = json.loads(line)
                t     = item["topic"]
                mine  = self.topic(t["topic"], create=True)
                have  = {(n["subtopic"], n["text"]) for n in self.notes(mine["id"])}
                for n in item["notes"]:
                    if (n["subtopic"], n["text"]) in have:
                        continue
                    self.add_note(mine["id"], n["subtopic"], n["text"], n.get("url"), n.get("title"))
                    if n.get("url"):
                        self.mark_page(mine["id"], n["url"], True)
                    added += 1
                covered = list(dict.fromkeys(mine["covered"] + t.get("covered", [])))
                next_up = [s for s in dict.fromkeys(mine["next"] + t.get("next", [])) if s not in covered]
                self.save_progress(mine["id"], t.get("minutes", 0) * 60, covered, next_up, new_session=True)
        return added
