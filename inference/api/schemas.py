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


class HealthResponse(BaseModel):
    status: str = "ok"
    model:  str