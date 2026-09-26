"""
schemas.py — PY-V (inference/api/)
Pydantic request and response models for the FastAPI layer.
No business logic here — types only.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    instruction: str = Field(
        ...,
        min_length  = 5,
        description = "Natural language description of the Python code to generate.",
        example     = "Write a function that checks if a string is a palindrome.",
    )
    max_tokens: int = Field(
        default     = 512,
        ge          = 16,
        le          = 1024,
        description = "Maximum number of new tokens to generate.",
    )
    temperature: float = Field(
        default     = 0.2,
        ge          = 0.0,
        le          = 1.0,
        description = "Sampling temperature. Lower = more deterministic.",
    )


class GenerateResponse(BaseModel):
    instruction: str = Field(..., description="The original instruction echoed back.")
    code:        str = Field(..., description="Generated Python code.")
    tokens_used: int = Field(..., description="Approximate number of tokens in the output.")


class OpenFile(BaseModel):
    """The file open in the editor, sent by the chat panel (Phase 10.2) — see inference/engine/file_context.py."""
    name:           str = Field(..., min_length=1, description="Path relative to the workspace, e.g. src/app.py.")
    language_id:    str = Field("", description="VS Code's languageId (python, typescript, ...).")
    content:        str = Field("", max_length=400_000, description="The file's text (a window around the cursor for a huge file).")
    first_line:     int = Field(1, ge=1, description="Line number of the first line of content.")
    selection:      str = Field("", max_length=400_000, description="Selected text, empty when nothing is selected.")
    selection_line: int = Field(0, ge=0, description="First selected line (1-based); 0 = nothing selected.")
    cursor_line:    int = Field(0, ge=0, description="Cursor line (1-based); 0 = unknown.")


class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length  = 1,
        description = "Stable identifier for this conversation session.",
        example     = "user-abc123",
    )
    message: str = Field(
        ...,
        min_length  = 1,
        description = "The user's message or instruction.",
        example     = "Write a quicksort implementation.",
    )
    file: Optional[OpenFile] = Field(
        None,
        description = "The file open in the editor (chat panel). Its code goes into the prompt when the message is about it.",
    )
    project: Optional[str] = Field(
        None,
        description = "The workspace folder (chat panel). Searched when a question is about the project (Phase 10.5).",
    )


class ChatResponse(BaseModel):
    session_id:  str   = Field(..., description="Echoed session ID.")
    response:    str   = Field(..., description="Model response.")
    mode:        str   = Field(..., description="Detected intent mode (generate/debug/explain/refactor/chat).")
    confidence:  float = Field(..., description="Controller confidence score for the detected mode.")
    rag_chunks:  int   = Field(..., description="Number of RAG chunks injected into the prompt.")
    memories:    int   = Field(0,   description="Number of memory items (facts / earlier code) added to the prompt.")
    language:    Optional[str] = Field(None, description="Programming language of the answer when not Python (e.g. TypeScript).")
    load:        Optional[str] = Field(None, description="How busy the computer was: free / busy / tight (V takes on less when busy).")
    note:        Optional[str] = Field(None, description="V's casual heads-up about the computer, when she has one.")
    file:        Optional[str] = Field(None, description="What of the open file V read, e.g. 'app.py, lines 10-24'; None = nothing.")
    notes:       int           = Field(0, description="Study notes added to the prompt (Phase 13).")
    project:     int           = Field(0, description="Pieces of the user's project added to the prompt (Phase 10.5).")
    project_files: List[str]   = Field(default_factory=list, description="Which ones, e.g. 'src/a.py:10-40'.")
    sources:     Optional[List[dict]] = Field(None, description="Pages a web lookup read: [{title, url}] (Phase 13).")
    asks:        Optional[List[str]]  = Field(None, description="Quick replies the panel offers, e.g. to 'Want me to look that up online?'.")
    study:       Optional[dict] = Field(None, description="The study session, if one runs (Phase 13).")


class ChangeRequest(BaseModel):
    """A code block from V's answer to put into the open file (chat panel "Apply to file", Phase 10.3)."""
    code: str      = Field(..., min_length=1, max_length=400_000, description="The code block.")
    file: OpenFile = Field(..., description="The file as it is now; selection / selection_line = what the answer read.")


class ChangeResponse(BaseModel):
    content: str = Field(..., description="The whole file with the change — shown as a diff, written only on Apply.")
    how:     str = Field(..., description="selection / whole / blocks / insert")
    summary: str = Field(..., description="What the change does, e.g. 'replaces add() and adds 1 import'.")


class MemoryFact(BaseModel):
    id:      int   = Field(..., description="Fact id — use it to delete the fact.")
    key:     str   = Field(..., description="What the fact is about; a newer fact with the same key replaces it.")
    text:    str   = Field(..., description="The fact as V sees it.")
    created: float = Field(..., description="When it was saved (unix time).")


class MemoryListResponse(BaseModel):
    enabled:  bool             = Field(..., description="Whether memory is on.")
    facts:    list[MemoryFact] = Field(default_factory=list, description="Active facts, newest first.")
    messages: int              = Field(0, description="Messages saved across all chats.")
    sessions: int              = Field(0, description="Chats with saved messages.")


class MemoryDeleteResponse(BaseModel):
    id:      int  = Field(..., description="The fact id that was asked to be deleted.")
    deleted: bool = Field(..., description="True if it existed and is now gone.")


class ProjectIndexRequest(BaseModel):
    root: str = Field(..., min_length=1, description="The workspace folder to index (the user's own code, stays local).")


class ProjectStatus(BaseModel):
    root:     str
    status:   str             = Field(..., description="idle / indexing / paused / ready / error / off")
    files:    int             = 0
    pieces:   int             = 0
    embedded: int             = 0
    updated:  Optional[float] = None
    error:    Optional[str]   = None


class ApproveRequest(BaseModel):
    """ "Good answer" in the panel — saved as a training example (Phase 13)."""
    question:   str           = Field(..., min_length=1)
    answer:     str           = Field(..., min_length=1)
    mode:       Optional[str] = None
    language:   Optional[str] = None
    session_id: Optional[str] = None


class TopicApproval(BaseModel):
    approved: bool = Field(..., description="True = her notes on this topic may go into the next training run.")


class HealthResponse(BaseModel):
    status: str = "ok"
    model:  str