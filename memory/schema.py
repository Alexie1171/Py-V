"""
schema.py — PY-V (memory/)
Data shapes for V's long-term memory. Dataclasses only, no logic.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Message:
    """One saved chat message (user or assistant), from any session."""
    id:         int
    session_id: str
    role:       str            # "user" or "assistant"
    mode:       Optional[str]  # generate / debug / refactor / explain / chat
    text:       str
    created:    float          # unix time


@dataclass
class Fact:
    """A short tagged statement. A newer fact with the same key replaces the
    older one (the older one stays in the database, marked inactive)."""
    id:                int
    key:               str     # e.g. "user:name", "preference:tabs-over-spaces", "error:typeerror"
    text:              str     # e.g. "The user's name is Sam."
    source_message_id: Optional[int]
    created:           float
    active:            bool


@dataclass
class MemoryHit:
    """A search result: a fact, or an earlier code answer (code modes only)."""
    kind:    str               # "fact" or "code"
    id:      int
    text:    str
    score:   float
    created: float
