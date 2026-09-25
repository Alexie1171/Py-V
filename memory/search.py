"""
search.py — PY-V (memory/)
Hybrid memory search: keyword (SQLite FTS5 / bm25) + meaning (embeddings,
retrieval/embedder.py on the CPU) + a small recency boost, merged and ranked.

  facts(query, top_k)        — active facts; used in every mode
  code(query, modes, top_k)  — earlier assistant answers that contain code,
                               given in the code modes; only code modes ask
"""

import re
import time

from memory.schema import MemoryHit

KEYWORD_WEIGHT = 0.45
MEANING_WEIGHT = 0.45
RECENCY_WEIGHT = 0.10
MIN_SIMILARITY = 0.60    # bge-small: unrelated texts still score ~0.4–0.55 — below this, no meaning match
RECENCY_DAYS   = 30      # the recency boost halves every 30 days
CANDIDATES     = 30

_CODE = re.compile(r"```|^\s*(?:def|class|import|from)\s", re.MULTILINE)


def has_code(text: str) -> bool:
    return bool(_CODE.search(text or ""))


class MemorySearch:

    def __init__(self, store, embed=None):
        """embed: callable(list of texts) → normalized vectors, or None for keyword-only search."""
        self.store = store
        self.embed = embed

    def facts(self, query: str, top_k: int) -> list:
        active = {f.id: f for f in self.store.facts()}
        if not active:
            return []
        keyword = {i: s for i, s in self.store.keyword_search("fact", query, CANDIDATES).items() if i in active}
        meaning = self._meaning(query, "fact", active.keys())
        items   = {i: (f.text, f.created) for i, f in active.items()}
        return self._merge(keyword, meaning, items, "fact", top_k)

    def code(self, query: str, modes, top_k: int) -> list:
        allowed = set(self.store.code_message_ids(modes))
        if not allowed:
            return []
        keyword = {i: s for i, s in self.store.keyword_search("message", query, CANDIDATES * 3).items()
                   if i in allowed}
        meaning = self._meaning(query, "message", allowed)
        found   = {m.id: m for m in self.store.messages(set(keyword) | set(meaning)) if has_code(m.text)}
        items   = {i: (m.text, m.created) for i, m in found.items()}
        return self._merge(keyword, meaning, items, "code", top_k)

    # ─── internals ────────────────────────────────────────────────────────────

    def _meaning(self, query: str, kind: str, ids) -> dict:
        if self.embed is None:
            return {}
        vectors = self.store.vectors(kind, ids)
        if not vectors:
            return {}
        q = self.embed([query])
        if q is None:
            return {}
        q = q[0]
        scores = {i: float(v @ q) for i, v in vectors.items()}
        return {i: s for i, s in scores.items() if s >= MIN_SIMILARITY}

    @staticmethod
    def _merge(keyword: dict, meaning: dict, items: dict, kind: str, top_k: int) -> list:
        top_keyword = max(keyword.values(), default=0) or 1
        now, hits   = time.time(), []
        for i in (set(keyword) | set(meaning)) & set(items):
            text, created = items[i]
            score = (KEYWORD_WEIGHT * keyword.get(i, 0) / top_keyword
                     + MEANING_WEIGHT * ((meaning[i] - MIN_SIMILARITY) / (1 - MIN_SIMILARITY) if i in meaning else 0)
                     + RECENCY_WEIGHT * 0.5 ** ((now - created) / 86400 / RECENCY_DAYS))
            hits.append(MemoryHit(kind, i, text, round(score, 4), created))
        hits.sort(key=lambda h: (h.score, h.created), reverse=True)
        return hits[:top_k]
