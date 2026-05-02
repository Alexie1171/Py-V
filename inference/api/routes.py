"""
routes.py — PY-V (inference/api/)
FastAPI route definitions. Each endpoint delegates immediately to the
inference engine — no model logic lives here.
"""

from fastapi import APIRouter, HTTPException

from inference.api.schemas import GenerateRequest, GenerateResponse, HealthResponse
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
    Generate Python code from a natural language instruction.

    The model and tokenizer are loaded once at startup (see main.py)
    and injected via app state.
    """
    from inference.api.main import get_model  # late import avoids circular dep

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
        tokens_used = len(code.split()),   # rough token estimate
    )