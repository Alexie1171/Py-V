import json
import os
from inference.engine.context_schema import SessionContext, ChatTurn


class ContextManager:

    def __init__(self, storage_path="sessions"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    def load(self, session_id: str) -> SessionContext:
        path = f"{self.storage_path}/{session_id}.json"

        if not os.path.exists(path):
            return SessionContext(session_id=session_id)

        with open(path, "r") as f:
            data = json.load(f)

        return self._from_dict(data)

    def save(self, context: SessionContext):
        path = f"{self.storage_path}/{context.session_id}.json"

        with open(path, "w") as f:
            json.dump(context.to_dict(), f, indent=2)

    def append_history(self, context: SessionContext, user: str, assistant: str):
        context.history.append(ChatTurn("user", user))
        context.history.append(ChatTurn("assistant", assistant))
        self.save(context)

    def update(self, context: SessionContext, **kwargs):
        for k, v in kwargs.items():
            if hasattr(context, k):
                setattr(context, k, v)
        self.save(context)

    def _from_dict(self, data: dict) -> SessionContext:
        ctx = SessionContext(
            session_id=data["session_id"],
            language=data.get("language", "python"),
            mode=data.get("mode", "chat"),
            current_task=data.get("current_task"),
            entities=data.get("entities", []),
            errors_seen=data.get("errors_seen", []),
            functions_touched=data.get("functions_touched", []),
            last_summary=data.get("last_summary")
        )

        history = data.get("history", [])
        ctx.history = [ChatTurn(h["role"], h["content"]) for h in history]

        return ctx