"""
manager.py — PY-V (memory/)
MemoryManager — what the chat engine and the API use:
  remember_turn()  save both messages, tag facts from the user's message
  remember_fact()  a fact from a web lookup, with its source (Phase 13)
  recall()         facts (every mode) + earlier code answers (code modes only)
  list_facts() / forget()   for GET / DELETE /api/v1/memory
  load_session() / save_session() / history()   session state (replaces sessions/*.json)

Settings from CFG.memory. The embedding model (retrieval/embedder.py) runs on
the CPU — memory never uses the GPU — and is loaded on first use; if it can't
load, search falls back to keywords only.
"""

import logging

from model.training.config_loader import CFG
from memory.extractor import extract_facts
from memory.search import MemorySearch, has_code
from memory.store import MemoryStore

logger = logging.getLogger(__name__)

EMBED_CHARS = 1200   # text embedded per item — the question plus the start of the answer


class MemoryManager:

    def __init__(self, cfg=None):
        self.cfg      = cfg or CFG.memory
        self.store    = MemoryStore(self.cfg.db_path)
        self._encoder = None
        self._failed  = False
        self.search   = MemorySearch(self.store, self._embed if self.cfg.semantic_search else None)

    # ─── embeddings (CPU, lazy) ───────────────────────────────────────────────

    def _embed(self, texts):
        if self._failed:
            return None
        if self._encoder is None:
            try:
                from retrieval.embedder import cpu_embedder
                self._encoder = cpu_embedder()   # shared with project search and study notes
            except Exception as e:
                logger.warning(f"Memory: embedding model unavailable ({e}) - keyword search only")
                self._failed = True
                return None
        return self._encoder.encode([t[:EMBED_CHARS] for t in texts])

    def _index(self, kind: str, item_id: int, text: str):
        if not self.cfg.semantic_search:
            return
        vectors = self._embed([text])
        if vectors is not None:
            self.store.set_vector(kind, item_id, vectors[0])

    # ─── chat turns ───────────────────────────────────────────────────────────

    def remember_turn(self, session_id: str, mode: str, user_text: str, assistant_text: str) -> list:
        """Save both messages and the facts in the user's message. Returns the new facts' ids."""
        user_id      = self.store.add_message(session_id, "user", mode, user_text)
        assistant_id = self.store.add_message(session_id, "assistant", mode, assistant_text)

        fact_ids = []
        for key, fact in extract_facts(user_text):
            fact_id = self.store.add_fact(key, fact, user_id)
            self._index("fact", fact_id, fact)
            fact_ids.append(fact_id)

        # earlier code answers are searchable for later code questions
        if mode in self.cfg.active_code_modes and has_code(assistant_text):
            self._index("message", assistant_id, f"{user_text}\n{assistant_text}")
        return fact_ids

    def remember_fact(self, key: str, text: str) -> int:
        """A fact that doesn't come from the user's message — what a web lookup found, with its source
        (Phase 13). Same key → the newer one wins."""
        fact_id = self.store.add_fact(key, text)
        self._index("fact", fact_id, text)
        return fact_id

    def recall(self, query: str, mode: str, top_k: int = None, code_top_k: int = None) -> dict:
        """{"facts": [MemoryHit], "code": [MemoryHit]} — code only in the code modes.
        top_k / code_top_k: fewer items when the machine is busy (None = config)."""
        top_k      = self.cfg.top_k      if top_k      is None else top_k
        code_top_k = self.cfg.code_top_k if code_top_k is None else code_top_k
        facts = self.search.facts(query, top_k) if top_k > 0 else []
        code  = (self.search.code(query, self.cfg.active_code_modes, code_top_k)
                 if mode in self.cfg.active_code_modes and code_top_k > 0 else [])
        return {"facts": facts, "code": code}

    # ─── the user's view ──────────────────────────────────────────────────────

    def list_facts(self) -> list:
        return self.store.facts(active_only=True)

    def forget(self, fact_id: int) -> bool:
        return self.store.delete_fact(fact_id)

    def stats(self) -> dict:
        return self.store.stats()

    # ─── session state ────────────────────────────────────────────────────────

    def load_session(self, session_id: str):
        return self.store.load_session(session_id)

    def save_session(self, session_id: str, state: dict):
        self.store.save_session(session_id, state)

    def history(self, session_id: str) -> list:
        return self.store.recent_messages(session_id, self.cfg.history_turns)
