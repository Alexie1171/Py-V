"""
schemas.py — PY-V (inference/api/)
Pydantic request and response models for the FastAPI layer.
No business logic here — types only.
"""

from typing import Optional

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


class HealthResponse(BaseModel):
    status: str = "ok"
    model:  str