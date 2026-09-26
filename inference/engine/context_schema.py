# inference/engine/context_schema.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ChatTurn:
    role: str  # "user" or "assistant"
    content: str
    mode: Optional[str] = None  # the mode the turn was answered in (chat history leaves code-mode turns' code out)


@dataclass
class SessionContext:

    session_id: str

    # core state
    language: str = "python"
    mode: str = "chat"

    # current task tracking
    current_task: Optional[str] = None

    # memory
    history: List[ChatTurn] = field(default_factory=list)

    # extracted intelligence
    entities: List[str] = field(default_factory=list)
    errors_seen: List[str] = field(default_factory=list)
    functions_touched: List[str] = field(default_factory=list)

    # system hints
    last_summary: Optional[str] = None

    # Phase 13: V asked "Want me to look that up online?" — the search words and the question, until answered
    pending_lookup: Optional[str] = None
    pending_question: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "language": self.language,
            "mode": self.mode,
            "current_task": self.current_task,
            "history": [
                {"role": h.role, "content": h.content, "mode": h.mode}
                for h in self.history
            ],
            "entities": self.entities,
            "errors_seen": self.errors_seen,
            "functions_touched": self.functions_touched,
            "last_summary": self.last_summary,
            "pending_lookup": self.pending_lookup,
            "pending_question": self.pending_question,
        }