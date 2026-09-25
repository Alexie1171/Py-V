from inference.engine.controller import Controller
from inference.engine.context_manager import ContextManager
from inference.engine.prompt_builder import build_prompt
from inference.engine.model_loader import load_lora_model
from inference.engine.generator import generate_from_prompt
from model.training.config_loader import CFG

import logging

logger = logging.getLogger(__name__)

# Modes that benefit from RAG context — must match prompt_templates.py slots
_RAG_MODES = set(CFG.rag.active_modes)


def _load_retriever():
    """
    Lazily import and instantiate the Retriever.
    Wrapped in a function so that if the index doesn't exist yet
    (e.g. during a fresh setup before indexer.py has been run),
    the chat engine still starts — it just disables RAG gracefully.
    """
    try:
        from retrieval.retriever import Retriever
        return Retriever(index_path=str(CFG.rag.index_path))
    except Exception as e:
        logger.warning(f"RAG retriever could not be loaded: {e}")
        logger.warning("Continuing without RAG context.")
        return None


def _load_memory():
    """V's long-term memory (Phase 11). If the database can't be opened,
    chat carries on without memory — same as RAG."""
    try:
        from memory.manager import MemoryManager
        return MemoryManager()
    except Exception as e:
        logger.warning(f"Memory could not be loaded: {e}")
        logger.warning("Continuing without memory.")
        return None


class ChatEngine:

    def __init__(self, model=None, tokenizer=None):
        """model/tokenizer: pass the already-loaded brain (the API server does)
        so it is never loaded twice; loaded here when not given."""
        self.controller      = Controller()
        self.memory          = _load_memory() if CFG.memory.enabled else None
        self.context_manager = ContextManager(self.memory)
        if model is None:
            model, tokenizer = load_lora_model()
        self.model, self.tokenizer = model, tokenizer

        # RAG retriever — None if disabled in config or index missing
        self.retriever = _load_retriever() if CFG.rag.enabled else None

        if self.retriever:
            logger.info(f"RAG enabled — index: {CFG.rag.index_path}, top_k: {CFG.rag.top_k}")
        else:
            logger.info("RAG disabled.")
        logger.info(f"Memory {'enabled — ' + str(CFG.memory.db_path) if self.memory else 'disabled'}.")

    def _retrieve(self, mode: str, user_input: str) -> list:
        """
        Run retrieval if RAG is active and mode supports it.
        Returns a list of result dicts (may be empty).
        """
        if not self.retriever:
            return []
        if mode not in _RAG_MODES:
            return []

        try:
            results = self.retriever.search(user_input, k=CFG.rag.top_k)
            logger.debug(f"RAG retrieved {len(results)} chunks for mode='{mode}'")
            return results
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            return []

    def _recall(self, mode: str, user_input: str) -> dict:
        """Memory search before the prompt is built — {} when memory is off or fails."""
        if not self.memory:
            return {}
        try:
            return self.memory.recall(user_input, mode)
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return {}

    def chat(self, session_id: str, user_input: str):

        context = self.context_manager.load(session_id)

        intent = self.controller.detect_mode(user_input)

        # Retrieve relevant context and memories before building the prompt
        retrieved_chunks = self._retrieve(intent.mode, user_input)
        memories         = self._recall(intent.mode, user_input)

        prompt = build_prompt(
            mode             = intent.mode,
            user_input       = user_input,
            context          = context.to_dict(),
            retrieved_chunks = retrieved_chunks,
            memories         = memories,
        )

        response = generate_from_prompt(
            model       = self.model,
            tokenizer   = self.tokenizer,
            prompt      = prompt,
            mode        = intent.mode,
            temperature = 0.2,
        )

        # Saves the turn to memory (and tags facts from the user's message)
        self.context_manager.append_history(context, user_input, response, intent.mode)
        self.context_manager.update(context, mode=intent.mode)

        return {
            "response":      response,
            "mode":          intent.mode,
            "confidence":    intent.confidence,
            "flags":         intent.flags,
            "rag_chunks":    len(retrieved_chunks),   # useful for debugging
            "memories":      len(memories.get("facts", [])) + len(memories.get("code", [])),
        }
