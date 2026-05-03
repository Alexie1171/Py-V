# inference/engine/context_schema.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ChatTurn:
    role: str  # "user" or "assistant"
    content: str


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "language": self.language,
            "mode": self.mode,
            "current_task": self.current_task,
            "history": [
                {"role": h.role, "content": h.content}
                for h in self.history
            ],
            "entities": self.entities,
            "errors_seen": self.errors_seen,
            "functions_touched": self.functions_touched,
            "last_summary": self.last_summary
        }