"""
context_manager.py — PY-V (inference/engine/)
Session state for the chat engine. Since Phase 11 it reads and writes through
V's memory (memory/manager.py → SQLite) instead of sessions/*.json: every turn
is saved there (and its facts tagged), the session's small state (mode,
language, ...) too. With memory disabled or broken, sessions live in RAM only
and chat carries on.
"""

import logging

from inference.engine.context_schema import SessionContext, ChatTurn

logger = logging.getLogger(__name__)

_STATE_FIELDS = ("language", "mode", "current_task", "entities", "errors_seen",
                 "functions_touched", "last_summary")


class ContextManager:

    def __init__(self, memory=None):
        self.memory = memory       # MemoryManager or None
        self._ram   = {}           # session_id -> SessionContext when there is no memory

    def load(self, session_id: str) -> SessionContext:
        if self.memory is None:
            return self._ram.setdefault(session_id, SessionContext(session_id=session_id))
        try:
            state   = self.memory.load_session(session_id) or {}
            context = SessionContext(session_id=session_id,
                                     **{k: v for k, v in state.items() if k in _STATE_FIELDS})
            context.history = [ChatTurn(m.role, m.text) for m in self.memory.history(session_id)]
            return context
        except Exception as e:
            logger.warning(f"Memory: could not load session {session_id} ({e}) - starting fresh")
            return SessionContext(session_id=session_id)

    def save(self, context: SessionContext):
        if self.memory is None:
            self._ram[context.session_id] = context
            return
        try:
            state = {k: getattr(context, k) for k in _STATE_FIELDS}
            self.memory.save_session(context.session_id, state)
        except Exception as e:
            logger.warning(f"Memory: could not save session {context.session_id} ({e})")

    def append_history(self, context: SessionContext, user: str, assistant: str, mode: str = None):
        context.history.append(ChatTurn("user", user))
        context.history.append(ChatTurn("assistant", assistant))
        if self.memory is not None:
            try:
                self.memory.remember_turn(context.session_id, mode or context.mode, user, assistant)
            except Exception as e:
                logger.warning(f"Memory: could not save the turn ({e})")
        self.save(context)

    def update(self, context: SessionContext, **kwargs):
        for k, v in kwargs.items():
            if hasattr(context, k):
                setattr(context, k, v)
        self.save(context)
