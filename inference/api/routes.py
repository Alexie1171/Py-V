"""
routes.py — PY-V (inference/api/)
FastAPI route definitions. Each endpoint delegates immediately to the
inference engine — no model logic lives here.
"""

from fastapi import APIRouter, HTTPException

from inference.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
)
from inference.engine.generator import generate_code
from model.training.config_loader import CFG

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """Liveness check — confirms the server and model name are reachable."""
    return HealthResponse(status="ok", model=CFG.model.name)


@router.post("/generate", response_model=GenerateResponse, tags=["inference"])
def generate(request: GenerateRequest):
    """
    Stateless code generation from a natural language instruction.
    No session context or RAG — use /chat for the full assistant experience.
    """
    from inference.api.main import get_model

    model, tokenizer = get_model()

    try:
        code = generate_code(
            model       = model,
            tokenizer   = tokenizer,
            instruction = request.instruction,
            max_tokens  = request.max_tokens,
            temperature = request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return GenerateResponse(
        instruction = request.instruction,
        code        = code,
        tokens_used = len(code.split()),
    )


@router.post("/chat", response_model=ChatResponse, tags=["inference"])
def chat(request: ChatRequest):
    """
    Context-aware chat with RAG retrieval and session history.
    Pass a stable session_id to maintain conversation state across turns.
    """
    from inference.api.main import get_chat_engine

    engine = get_chat_engine()

    try:
        result = engine.chat(
            session_id = request.session_id,
            user_input = request.message,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

    return ChatResponse(
        session_id  = request.session_id,
        response    = result["response"],
        mode        = result["mode"],
        confidence  = result["confidence"],
        rag_chunks  = result["rag_chunks"],
    )