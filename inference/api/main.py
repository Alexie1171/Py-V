"""
main.py — PY-V (inference/api/)
FastAPI application entry point.
Loads the model once at startup and stores it in app state so routes
don't reload it on every request.

Run with:
    uvicorn inference.api.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from inference.api.routes import router
from inference.engine.model_loader import load_lora_model
from model.training.config_loader import CFG

# ─── Model State ──────────────────────────────────────────────────────────────

_model     = None
_tokenizer = None


def get_model():
    """Return the globally loaded model and tokenizer."""
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Did startup complete?")
    return _model, _tokenizer


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global _model, _tokenizer

    _model, _tokenizer = load_lora_model()
    print("Server ready.")

    yield  # app runs here

    _model     = None
    _tokenizer = None
    print("Model released.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "PY-V Inference API",
    description = "Local Python code generation using Phi-2 + LoRA.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.include_router(router, prefix="/api/v1")


# ─── Dev Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference.api.main:app", host="0.0.0.0", port=8000, reload=False)