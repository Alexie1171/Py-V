"""
schemas.py — PY-V (inference/api/)
Pydantic request and response models for the FastAPI layer.
No business logic here — types only.
"""

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


class HealthResponse(BaseModel):
    status: str = "ok"
    model:  str