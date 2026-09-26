from inference.engine.controller import Controller, IntentResult
from inference.engine.context_manager import ContextManager
from inference.engine.intent_classifier import classify_with_brain
from inference.engine.prompt_builder import build_prompt, build_chat_prompt, uses_chat_format, format_machine
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


def _load_machine():
    """What V knows about the computer (inference/engine/machine.py). If it
    can't be read, chat carries on with the normal settings."""
    try:
        from inference.engine.machine import Machine
        return Machine()
    except Exception as e:
        logger.warning(f"Machine check could not start: {e}")
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
        self.machine         = _load_machine() if CFG.machine.enabled else None
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

    def _recall(self, mode: str, user_input: str, work=None) -> dict:
        """Memory search before the prompt is built — {} when memory is off or fails.
        work = machine limits (fewer items when the machine is busy)."""
        if not self.memory:
            return {}
        try:
            if work is None:
                return self.memory.recall(user_input, mode)
            return self.memory.recall(user_input, mode, work.memory_facts, work.memory_code)
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return {}

    def _check_machine(self):
        """How busy the computer is right now (machine.Snapshot), or None."""
        if not self.machine:
            return None
        try:
            return self.machine.check()
        except Exception as e:
            logger.warning(f"Machine check failed: {e}")
            return None

    def _detect_mode(self, user_input: str) -> IntentResult:
        """Word rules; a message they flag "unclear" is decided by the brain
        (config intent.brain_for_unclear) — if it names no mode or fails, the rules' guess stands."""
        intent = self.controller.detect_mode(user_input)
        if "unclear" not in intent.flags or not CFG.intent.brain_for_unclear:
            return intent
        try:
            picked = classify_with_brain(self.model, self.tokenizer, user_input)
        except Exception as e:
            logger.warning(f"Brain mode check failed: {e}")
            return intent
        if picked is None:
            return intent
        return IntentResult(mode=picked, confidence=intent.confidence, flags=intent.flags + [f"brain_{picked}"])

    def chat(self, session_id: str, user_input: str):

        context = self.context_manager.load(session_id)

        intent = self._detect_mode(user_input)

        # How busy the computer is decides how much V takes on for this answer
        snap = self._check_machine()
        work = self.machine.work(snap) if snap else None

        # Retrieve relevant context and memories before building the prompt
        retrieved_chunks = self._retrieve(intent.mode, user_input)
        memories         = self._recall(intent.mode, user_input, work)

        # Chat mode on a chat brain: a real conversation (persona, what she
        # knows about the computer, the last turns, the message as written);
        # every other mode: V's templates
        formatted = intent.mode == "chat" and uses_chat_format(self.model, self.tokenizer)
        if formatted:
            prompt = build_chat_prompt(
                user_input, context.to_dict(), memories, self.tokenizer,
                history_turns = work.history_turns if work else None,
                machine       = format_machine(self.machine.specs, snap) if snap else "",
            )
        else:
            prompt = build_prompt(
                mode             = intent.mode,
                user_input       = user_input,
                context          = context.to_dict(),
                retrieved_chunks = retrieved_chunks,
                memories         = memories,
            )

        response = generate_from_prompt(
            model     = self.model,
            tokenizer = self.tokenizer,
            prompt    = prompt,
            mode       = intent.mode,
            formatted  = formatted,
            max_tokens = work.prose_max_tokens if work and intent.mode in ("chat", "explain") else None,
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
            "load":          snap.level if snap else None,          # free / busy / tight
            "note":          self.machine.heads_up(snap) if snap else None,   # V's casual heads-up, if any
        }
