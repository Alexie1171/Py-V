"""
main.py — PY-V (inference/api/)
FastAPI application entry point.
Loads the model and chat engine once at startup and stores them in app
state so routes don't reload on every request.

Run with:
    uvicorn inference.api.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from inference.api.routes import router
from inference.engine.model_loader import load_lora_model
from model.training.config_loader import CFG

# ─── App State ────────────────────────────────────────────────────────────────

_model       = None
_tokenizer   = None
_chat_engine = None


def get_model():
    """Return the globally loaded base model and tokenizer."""
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Did startup complete?")
    return _model, _tokenizer


def get_chat_engine():
    """Return the globally loaded ChatEngine instance."""
    if _chat_engine is None:
        raise RuntimeError("ChatEngine not loaded. Did startup complete?")
    return _chat_engine


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and chat engine on startup, release on shutdown."""
    global _model, _tokenizer, _chat_engine

    # Load base model for the stateless /generate endpoint
    _model, _tokenizer = load_lora_model()

    # Load the full chat engine (includes retriever if RAG is enabled)
    from inference.engine.chat import ChatEngine
    _chat_engine = ChatEngine()

    print("Server ready.")

    yield  # app runs here

    _model       = None
    _tokenizer   = None
    _chat_engine = None
    print("Resources released.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "PY-V Inference API",
    description = "Local Python code generation using Phi-2 + LoRA + RAG.",
    version     = "1.1.0",
    lifespan    = lifespan,
)

app.include_router(router, prefix="/api/v1")


# ─── Dev Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference.api.main:app", host="0.0.0.0", port=8000, reload=False)