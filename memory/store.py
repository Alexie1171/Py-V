"""
store.py — PY-V (memory/)
SQLite access for V's long-term memory: one local file, Python's built-in
sqlite3, no server. Tables:
  messages  — every user + assistant message, all sessions
  facts     — short tagged statements; newest per key is active
  vectors   — meaning-search embeddings (float32) for facts and messages
  sessions  — small per-session state (mode, language, ...) — replaces sessions/*.json
plus FTS5 keyword indexes over messages and facts, kept in sync by triggers.
Thread-safe (FastAPI serves sync routes from a thread pool).
"""

import json
import re
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from memory.schema import Fact, Message

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
    mode TEXT, text TEXT NOT NULL, created REAL NOT NULL);
CREATE INDEX IF NOT EXISTS messages_by_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY, key TEXT NOT NULL, text TEXT NOT NULL,
    source_message_id INTEGER, created REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1);
CREATE INDEX IF NOT EXISTS facts_by_key ON facts(key, active);

CREATE TABLE IF NOT EXISTS vectors (
    kind TEXT NOT NULL, item_id INTEGER NOT NULL, vector BLOB NOT NULL,
    PRIMARY KEY (kind, item_id));

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, state TEXT NOT NULL, updated REAL NOT NULL);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(text, content='messages', content_rowid='id');
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts    USING fts5(text, content='facts',    content_rowid='id');

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text) VALUES ('delete', old.id, old.text); END;
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text); END;
"""

_WORD = re.compile(r"[A-Za-z0-9_]{2,}")

# Words too common to say anything — without this every fact matches every question
_STOPWORDS = set("""
a an and are as at be but by can could did do does for from had has have how i if in into is it its
me my no not of on or our please should so than that the their them then there these they this to
was we were what when where which who why will with would you your yours v us am been being also just
""".split())


def _fts_query(text: str) -> str:
    """User text → a safe FTS5 query: its meaningful words, any of them may match."""
    words = [w.lower() for w in _WORD.findall(text)]
    words = list(dict.fromkeys(w for w in words if w not in _STOPWORDS))[:20]
    return " OR ".join(f'"{w}"' for w in words)


class MemoryStore:

    def __init__(self, db_path):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db   = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    # ─── messages ─────────────────────────────────────────────────────────────

    def add_message(self, session_id: str, role: str, mode: str, text: str) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO messages (session_id, role, mode, text, created) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, mode, text, time.time()))
            self._db.commit()
            return cur.lastrowid

    def recent_messages(self, session_id: str, limit: int) -> list:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit)).fetchall()
        return [self._message(r) for r in reversed(rows)]

    def messages(self, ids) -> list:
        ids = list(ids)
        if not ids:
            return []
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM messages WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()
        return [self._message(r) for r in rows]

    def delete_message(self, message_id: int) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM messages WHERE id = ?", (message_id,))
            self._db.execute("DELETE FROM vectors WHERE kind = 'message' AND item_id = ?", (message_id,))
            self._db.commit()
            return cur.rowcount > 0

    # ─── facts ────────────────────────────────────────────────────────────────

    def add_fact(self, key: str, text: str, source_message_id: int = None) -> int:
        """Save a fact; the newest per key wins (older active ones are marked inactive).
        Returns the id — the existing one if the same fact is already active."""
        with self._lock:
            same = self._db.execute(
                "SELECT id FROM facts WHERE key = ? AND text = ? AND active = 1", (key, text)).fetchone()
            if same:
                return same["id"]
            self._db.execute("UPDATE facts SET active = 0 WHERE key = ? AND active = 1", (key,))
            cur = self._db.execute(
                "INSERT INTO facts (key, text, source_message_id, created, active) VALUES (?, ?, ?, ?, 1)",
                (key, text, source_message_id, time.time()))
            self._db.commit()
            return cur.lastrowid

    def facts(self, active_only: bool = True, ids=None) -> list:
        sql, args = "SELECT * FROM facts", []
        where = []
        if active_only:
            where.append("active = 1")
        if ids is not None:
            ids = list(ids)
            if not ids:
                return []
            where.append(f"id IN ({','.join('?' * len(ids))})")
            args += ids
        if where:
            sql += " WHERE " + " AND ".join(where)
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY created DESC", args).fetchall()
        return [self._fact(r) for r in rows]

    def delete_fact(self, fact_id: int) -> bool:
        """Forget a fact completely (the user asked) — not just deactivate it."""
        with self._lock:
            cur = self._db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            self._db.execute("DELETE FROM vectors WHERE kind = 'fact' AND item_id = ?", (fact_id,))
            self._db.commit()
            return cur.rowcount > 0

    # ─── search helpers ───────────────────────────────────────────────────────

    def keyword_search(self, kind: str, text: str, limit: int) -> dict:
        """{id: relevance} by FTS5 bm25 — higher is better. kind: "fact" (active only) or "message"."""
        query = _fts_query(text)
        if not query:
            return {}
        if kind == "fact":
            sql = ("SELECT f.id, bm25(facts_fts) AS rank FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid "
                   "WHERE facts_fts MATCH ? AND f.active = 1 ORDER BY rank LIMIT ?")
        else:
            sql = ("SELECT rowid AS id, bm25(messages_fts) AS rank FROM messages_fts "
                   "WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?")
        with self._lock:
            rows = self._db.execute(sql, (query, limit)).fetchall()
        return {r["id"]: -r["rank"] for r in rows}      # bm25: lower = better → negate

    def set_vector(self, kind: str, item_id: int, vector):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO vectors (kind, item_id, vector) VALUES (?, ?, ?)",
                             (kind, item_id, np.asarray(vector, dtype=np.float32).tobytes()))
            self._db.commit()

    def vectors(self, kind: str, ids=None) -> dict:
        sql, args = "SELECT item_id, vector FROM vectors WHERE kind = ?", [kind]
        if ids is not None:
            ids = list(ids)
            if not ids:
                return {}
            sql += f" AND item_id IN ({','.join('?' * len(ids))})"
            args += ids
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return {r["item_id"]: np.frombuffer(r["vector"], dtype=np.float32) for r in rows}

    def code_message_ids(self, modes) -> list:
        """Assistant answers given in the code modes — the only messages memory may quote as code."""
        modes = list(modes)
        with self._lock:
            rows = self._db.execute(
                f"SELECT id FROM messages WHERE role = 'assistant' AND mode IN ({','.join('?' * len(modes))})",
                modes).fetchall()
        return [r["id"] for r in rows]

    # ─── session state (replaces sessions/*.json) ─────────────────────────────

    def load_session(self, session_id: str):
        with self._lock:
            row = self._db.execute("SELECT state FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return json.loads(row["state"]) if row else None

    def save_session(self, session_id: str, state: dict):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO sessions (session_id, state, updated) VALUES (?, ?, ?)",
                             (session_id, json.dumps(state), time.time()))
            self._db.commit()

    def stats(self) -> dict:
        with self._lock:
            one = lambda sql: self._db.execute(sql).fetchone()[0]
            return {"messages": one("SELECT COUNT(*) FROM messages"),
                    "facts":    one("SELECT COUNT(*) FROM facts WHERE active = 1"),
                    "sessions": one("SELECT COUNT(DISTINCT session_id) FROM messages")}

    # ─── rows → dataclasses ───────────────────────────────────────────────────

    @staticmethod
    def _message(r) -> Message:
        return Message(r["id"], r["session_id"], r["role"], r["mode"], r["text"], r["created"])

    @staticmethod
    def _fact(r) -> Fact:
        return Fact(r["id"], r["key"], r["text"], r["source_message_id"], r["created"], bool(r["active"]))
