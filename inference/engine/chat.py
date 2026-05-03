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


class ChatEngine:

    def __init__(self):
        self.controller      = Controller()
        self.context_manager = ContextManager()
        self.model, self.tokenizer = load_lora_model()

        # RAG retriever — None if disabled in config or index missing
        self.retriever = _load_retriever() if CFG.rag.enabled else None

        if self.retriever:
            logger.info(f"RAG enabled — index: {CFG.rag.index_path}, top_k: {CFG.rag.top_k}")
        else:
            logger.info("RAG disabled.")

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

    def chat(self, session_id: str, user_input: str):

        context = self.context_manager.load(session_id)

        intent = self.controller.detect_mode(user_input)

        # Retrieve relevant context before building the prompt
        retrieved_chunks = self._retrieve(intent.mode, user_input)

        prompt = build_prompt(
            mode             = intent.mode,
            user_input       = user_input,
            context          = context.to_dict(),
            retrieved_chunks = retrieved_chunks,
        )

        response = generate_from_prompt(
            model       = self.model,
            tokenizer   = self.tokenizer,
            prompt      = prompt,
            mode        = intent.mode,
            temperature = 0.2,
        )

        self.context_manager.append_history(context, user_input, response)
        self.context_manager.update(context, mode=intent.mode)

        return {
            "response":      response,
            "mode":          intent.mode,
            "confidence":    intent.confidence,
            "flags":         intent.flags,
            "rag_chunks":    len(retrieved_chunks),   # useful for debugging
        }