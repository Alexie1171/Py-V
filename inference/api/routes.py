"""
routes.py — PY-V (inference/api/)
FastAPI route definitions. Each endpoint delegates immediately to the
inference engine — no model logic lives here.
"""

import json
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from inference.api.schemas import (
    ApproveRequest,
    ChangeRequest,
    ChangeResponse,
    GenerateRequest,
    GenerateResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MemoryDeleteResponse,
    MemoryFact,
    MemoryListResponse,
    ProjectIndexRequest,
    ProjectStatus,
    TopicApproval,
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
            open_file  = request.file.model_dump() if request.file else None,
            project    = request.project,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

    return ChatResponse(
        session_id  = request.session_id,
        response    = result["response"],
        mode        = result["mode"],
        confidence  = result["confidence"],
        rag_chunks  = result["rag_chunks"],
        memories    = result.get("memories", 0),
        language    = result.get("language"),
        load        = result.get("load"),
        note        = result.get("note"),
        file        = result.get("file"),
        notes       = result.get("notes", 0),
        project     = result.get("project", 0),
        project_files = result.get("project_files") or [],
        sources     = result.get("sources"),
        asks        = result.get("asks"),
        study       = result.get("study"),
    )


@router.post("/chat/stream", tags=["inference"])
async def chat_stream(request: ChatRequest):
    """
    /chat for the chat panel, as server-sent events: `start` (mode, confidence,
    load, language, file) → `status` (while she looks something up online) →
    `piece` (text as V writes it) → `done` (same fields as /chat — its
    `response` is the cleaned answer and replaces the pieces), or `error`.
    When the client disconnects (Stop button, panel closed) V stops writing.
    """
    from inference.api.main import get_chat_engine

    stop   = threading.Event()
    events = get_chat_engine().chat_stream(request.session_id, request.message, stop,
                                           open_file=request.file.model_dump() if request.file else None,
                                           project=request.project)

    async def sse():
        try:
            while True:
                item = await run_in_threadpool(next, events, None)
                if item is None:
                    break
                kind, data = item
                yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'detail': f'Chat failed: {e}'})}\n\n"
        finally:
            stop.set()   # the client left early: the brain stops after its current word

    return StreamingResponse(sse(), media_type="text/event-stream")


@router.post("/file/change", response_model=ChangeResponse, tags=["files"])
def file_change(request: ChangeRequest):
    """
    Where a code block from V's answer goes in the open file (Phase 10.3): the
    whole file with the change, for the panel's diff. Nothing is written here —
    the panel writes the file only when the user clicks Apply. No brain needed.
    """
    from inference.engine.file_update import propose

    f = request.file
    try:
        change = propose(f.content, request.code, f.language_id, f.selection, f.selection_line, f.cursor_line)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ChangeResponse(content=change.content, how=change.how, summary=change.summary)


@router.get("/memory", response_model=MemoryListResponse, tags=["memory"])
def list_memory():
    """What V remembers: active facts (newest first) and how many messages are saved."""
    from inference.api.main import get_chat_engine

    memory = get_chat_engine().memory
    if memory is None:
        return MemoryListResponse(enabled=False)
    stats = memory.stats()
    return MemoryListResponse(
        enabled  = True,
        facts    = [MemoryFact(id=f.id, key=f.key, text=f.text, created=f.created) for f in memory.list_facts()],
        messages = stats["messages"],
        sessions = stats["sessions"],
    )


@router.delete("/memory/{fact_id}", response_model=MemoryDeleteResponse, tags=["memory"])
def delete_memory(fact_id: int):
    """Make V forget one fact for good."""
    from inference.api.main import get_chat_engine

    memory = get_chat_engine().memory
    if memory is None:
        raise HTTPException(status_code=409, detail="Memory is disabled (config memory.enabled)")
    deleted = memory.forget(fact_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No fact with id {fact_id}")
    return MemoryDeleteResponse(id=fact_id, deleted=True)


# ─── Project search (Phase 10.5) ──────────────────────────────────────────────

@router.post("/project/index", response_model=ProjectStatus, tags=["project"])
def project_index(request: ProjectIndexRequest):
    """Index the workspace folder's code (in the background, incremental). The panel calls this when V is
    ready and after files are saved. The index stays on this computer (config project.index_dir)."""
    from inference.api.main import get_chat_engine

    return ProjectStatus(**_project_fields(get_chat_engine().index_project(request.root)))


@router.get("/project/status", response_model=ProjectStatus, tags=["project"])
def project_status(root: str):
    from inference.api.main import get_chat_engine

    engine = get_chat_engine()
    index  = engine.project_index(root)
    if index is None:
        return ProjectStatus(root=root, status="off")
    return ProjectStatus(**_project_fields({"root": str(index.root), **index.state}))


def _project_fields(state: dict) -> dict:
    return {k: state.get(k) for k in ("root", "status", "files", "pieces", "embedded", "updated", "error")
            if state.get(k) is not None}


# ─── Learning (Phase 13) ──────────────────────────────────────────────────────

def _learning():
    from inference.api.main import get_chat_engine

    learning = get_chat_engine().learning
    if learning is None:
        raise HTTPException(status_code=409, detail="Learning is switched off (config learning.enabled)")
    return learning


@router.get("/learning", tags=["learning"])
def learning_overview():
    """What V has studied (topics: time, notes, covered, next, approved), how many answers are saved
    for training, and the study session if one runs."""
    return _learning().overview()


@router.post("/learning/approve", tags=["learning"])
def approve_answer(request: ApproveRequest):
    """"Good answer": the question + answer become a training example for the next Kaggle retrain."""
    answer_id = _learning().approve(request.question, request.answer, request.mode, request.language,
                                    request.session_id)
    return {"id": answer_id}


@router.delete("/learning/approve/{answer_id}", tags=["learning"])
def unapprove_answer(answer_id: str):
    if not _learning().unapprove(answer_id):
        raise HTTPException(status_code=404, detail=f"No saved answer {answer_id}")
    return {"id": answer_id, "deleted": True}


@router.post("/learning/topics/{topic_id}/approve", tags=["learning"])
def approve_topic(topic_id: int, request: TopicApproval):
    """Owner's approval: her notes on this topic may go into the next training run (learning/export.py)."""
    if not _learning().store.approve_topic(topic_id, request.approved):
        raise HTTPException(status_code=404, detail=f"No topic {topic_id}")
    return {"id": topic_id, "approved": request.approved}


@router.delete("/learning/topics/{topic_id}", tags=["learning"])
def forget_topic(topic_id: int):
    """Forget a studied topic and its notes for good."""
    if not _learning().store.forget_topic(topic_id):
        raise HTTPException(status_code=404, detail=f"No topic {topic_id}")
    return {"id": topic_id, "deleted": True}


@router.get("/learning/study", tags=["learning"])
def study_status():
    """The study session: topic, minutes left, notes so far — {"running": false} when none."""
    return _learning().study_status() or {"running": False}


@router.post("/learning/study/stop", tags=["learning"])
def stop_study():
    return {"reply": _learning().stop_study()}
