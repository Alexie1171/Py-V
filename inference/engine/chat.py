"""
chat.py — PY-V (inference/engine/)
The chat engine: every message to V goes through here (API /chat and
/chat/stream — the chat panel — and the terminal test_chat.py).

  _special()   replies that need no brain: study requests ("learn about X for
               30 minutes"), "stop studying", a "no" to "Want me to look that up?"
  _prepare()   mode, machine check, the open file (10.2), project search
               (10.5), language, memory + study notes, the prompt; a yes to a
               lookup (or "look up X") makes it a lookup turn (Phase 13)
  _lookup()    the web part of a lookup turn: search, read, prompt with what she found
  _finish()    save the turn; after an answer that shows she doesn't know,
               she asks "Want me to look that up online?"

The brain goes to the user first: a study note being written is stopped when
a message arrives, and background work (study sessions, the project index)
waits while she answers.
"""

import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field

from inference.engine import file_context
from inference.engine.controller import CODE, Controller, IntentResult
from inference.engine.context_manager import ContextManager
from inference.engine.context_schema import SessionContext
from inference.engine.file_context import OpenFile
from inference.engine.intent_classifier import classify_with_brain
from inference.engine.language_detector import detect_language, file_language, names_python
from inference.engine.prompt_builder import (build_prompt, build_chat_prompt, build_study_prompt, uses_chat_format,
                                             format_machine, format_lookup, format_studies)
from inference.engine.model_loader import load_lora_model
from inference.engine.generator import BRAIN_LOCK, generate_from_prompt, stream_from_prompt
from model.training.config_loader import CFG

logger = logging.getLogger(__name__)

# Modes that benefit from RAG context — must match prompt_templates.py slots
_RAG_MODES = set(CFG.rag.active_modes)
_CODE_MODES = {"generate", "debug", "refactor"}

# When cleanup leaves nothing (in chat / explain usually: she started writing
# code, which is cut there), she says so instead of an empty answer
# ("give an example on type script" came out empty, 2026-09-26)
_EMPTY_WORDS = ("Hmm, I couldn't put that into words. If you wanted code, ask me to write it "
                "(like \"write an example of ...\") and I'll show it.")
_EMPTY_CODE  = "Sorry, I came up empty on that one. Could you describe it a bit more?"

# Phase 13 — looking things up, after asking
_OFFER       = "Want me to look that up online?"
_ASKS        = ["Yes, look it up", "No thanks"]
_LOOKUP_NO   = ["Okay, I won't look it up.", "No problem, I'll leave it.", "Alright, no lookup then."]
_LOOKUP_NONE = "I looked, but couldn't find anything useful online for that (or I'm offline right now)."
_ABOUT_STUDY = re.compile(r"\b(stud(y|ied|ying)|learn(ed|t|ing)?|your notes)\b", re.IGNORECASE)
_IT_UP       = re.compile(r"\b(it|that|this) up\b", re.IGNORECASE)
USER_QUIET_SECONDS = 45    # background study waits this long after the user's last message


def _load_retriever():
    """
    Lazily import and instantiate the Retriever.
    Wrapped in a function so that if the index doesn't exist yet
    (e.g. during a fresh setup before indexer.py has been run),
    the chat engine still starts — it just disables RAG gracefully.
    """
    try:
        from retrieval.retriever import Retriever
        return Retriever(index_path=str(CFG.rag.index_path), device=CFG.rag.device, min_score=CFG.rag.min_score)
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


def _load_learning():
    """What V learns (Phase 13: study sessions, web lookups, approved answers).
    If it can't start, chat carries on without it."""
    try:
        from learning.manager import LearningManager
        return LearningManager()
    except Exception as e:
        logger.warning(f"Learning could not be loaded: {e}")
        return None


class ChatEngine:

    def __init__(self, model=None, tokenizer=None):
        """model/tokenizer: pass the already-loaded brain (the API server does)
        so it is never loaded twice; loaded here when not given."""
        self.controller      = Controller()
        self.memory          = _load_memory() if CFG.memory.enabled else None
        self.context_manager = ContextManager(self.memory)
        self.machine         = _load_machine() if CFG.machine.enabled else None
        self.learning        = _load_learning() if CFG.learning.enabled else None
        if model is None:
            model, tokenizer = load_lora_model()
        self.model, self.tokenizer = model, tokenizer

        # RAG retriever — None if disabled in config or index missing
        self.retriever = _load_retriever() if CFG.rag.enabled else None

        # Project search (10.5): one project's index at a time
        self.projects       = {}
        self._projects_lock = threading.Lock()

        # The user comes first: background work waits while she answers
        self._answering = threading.Event()
        self._last_user = 0.0
        self._level     = ("free", 0.0)      # machine level, checked at most every 15 s for background work

        if self.retriever:
            logger.info(f"RAG enabled — index: {CFG.rag.index_path}, top_k: {CFG.rag.top_k}")
        else:
            logger.info("RAG disabled.")
        logger.info(f"Memory {'enabled — ' + str(CFG.memory.db_path) if self.memory else 'disabled'}.")

    # ─── helpers ──────────────────────────────────────────────────────────────

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

    def _machine_level(self) -> str:
        """free / busy / tight for background work — read at most every 15 s."""
        level, at = self._level
        if self.machine and time.time() - at > 15:
            snap  = self._check_machine()
            level = snap.level if snap else "free"
            self._level = (level, time.time())
        return level

    def _study_should_wait(self) -> bool:
        """A study session waits while the user is chatting (and a while after) or the laptop is busy."""
        return (self._answering.is_set() or time.time() - self._last_user < USER_QUIET_SECONDS
                or self._machine_level() != "free")

    def _index_should_wait(self) -> bool:
        """The project index waits while the brain writes (it shares the CPU) or the laptop is tight."""
        return self._answering.is_set() or BRAIN_LOCK.locked() or self._machine_level() == "tight"

    def _before_brain(self):
        """The user's answer is next: a study note being written stops (done again later)."""
        self._answering.set()
        self._last_user = time.time()
        if self.learning:
            self.learning.yield_brain()

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

    # ─── project search (10.5) ────────────────────────────────────────────────

    def project_index(self, root: str):
        """The search index of the user's project folder (retrieval/project.py), created and brought
        up to date in the background on first use; None when off or not a folder."""
        if not CFG.project.enabled or not root:
            return None
        key = os.path.normcase(os.path.abspath(root))
        with self._projects_lock:
            index = self.projects.get(key)
            if index is None:
                if not os.path.isdir(key):
                    return None
                from retrieval.project import ProjectIndex
                index = ProjectIndex(root, CFG.project.index_dir,
                                     embed          = self._embed if CFG.memory.semantic_search else None,
                                     max_file_kb    = CFG.project.max_file_kb,
                                     max_files      = CFG.project.max_files,
                                     piece_lines    = CFG.files.piece_lines,
                                     min_similarity = CFG.project.min_similarity)
                self.projects = {key: index}          # one project in RAM at a time
                index.update_in_background(self._index_should_wait)
        return index

    def index_project(self, root: str) -> dict:
        """Bring a project's index up to date (the panel calls this when it opens and after saves)."""
        index = self.project_index(root)
        if index is None:
            return {"root": root, "status": "off" if not CFG.project.enabled else "not a folder"}
        index.update_in_background(self._index_should_wait)
        return {"root": str(index.root), **index.state}

    @staticmethod
    def _embed(texts):
        try:
            from retrieval.embedder import cpu_embedder
            return cpu_embedder().encode(texts)
        except Exception as e:
            logger.warning(f"Embedder unavailable ({e}) - keyword search only")
            return None

    # ─── study sessions (Phase 13) ────────────────────────────────────────────

    def _study_writer(self):
        """write() for a study session on the server: a note with the plain brain, stopped when the user needs it."""
        def write(topic, subtopic, title, excerpt, covered, stop):
            prompt = build_study_prompt(topic, subtopic, title, excerpt, covered, self.tokenizer)
            text   = generate_from_prompt(self.model, self.tokenizer, prompt, mode="study",
                                          max_tokens=CFG.learning.note_tokens, temperature=0.3,
                                          formatted=True, adapter=False, stop=stop)
            return None if stop.is_set() else (text or None)
        return write

    # ─── one message ──────────────────────────────────────────────────────────

    def chat(self, session_id: str, user_input: str, open_file: dict = None, project: str = None):
        """open_file: the file open in the chat panel's editor (file_context.OpenFile
        fields as a dict) — its code goes in when the message is about it.
        project: the panel's workspace folder — searched when a question is about the project."""
        special = self._special(session_id, user_input)
        if special:
            return special
        turn = self._prepare(session_id, user_input, open_file, project)
        self._before_brain()
        try:
            if turn.lookup is not None and not self._lookup(turn):
                return self._finish(turn, _LOOKUP_NONE)
            response = generate_from_prompt(self.model, self.tokenizer, turn.prompt, **turn.generation)
        finally:
            self._answering.clear()
        return self._finish(turn, response)

    def chat_stream(self, session_id: str, user_input: str, stop: threading.Event = None, open_file: dict = None,
                    project: str = None):
        """
        chat(), streamed for the chat panel: yields ("start", {mode, confidence,
        load, language, file}) once the mode is known, ("status", {"text"})
        while she looks something up, ("piece", {"text"}) as V writes, then
        ("done", the chat() result) with the cleaned answer, which replaces the
        streamed text. stop: set it (Stop button, the user left) and she stops
        after her current word; what she wrote so far is kept.
        """
        special = self._special(session_id, user_input)
        if special:
            yield "start", {"mode": "chat", "confidence": 1.0, "load": None, "language": None, "file": None}
            yield "done", special
            return
        turn = self._prepare(session_id, user_input, open_file, project)
        yield "start", {"mode": turn.intent.mode, "confidence": turn.intent.confidence,
                        "load": turn.snap.level if turn.snap else None,
                        "language": turn.language["name"] if turn.language else None,
                        "file": turn.file_label}
        self._before_brain()
        response = ""
        try:
            if turn.lookup is not None:
                yield "status", {"text": f"Looking it up online: {turn.lookup['query']}"}
                if not self._lookup(turn):
                    response = _LOOKUP_NONE
            if not response:
                for kind, text in stream_from_prompt(self.model, self.tokenizer, turn.prompt, stop=stop,
                                                     **turn.generation):
                    if kind == "piece":
                        yield "piece", {"text": text}
                    else:
                        response = text
        finally:
            self._answering.clear()
        yield "done", self._finish(turn, response)

    def _special(self, session_id: str, user_input: str):
        """Replies without the brain (Phase 13): start / stop a study session, a "no" to a lookup offer.
        None = a normal message."""
        if not self.learning:
            return None
        from learning.lookup import answer_kind
        from learning.study import is_stop_request, parse_study_request

        context = None
        if is_stop_request(user_input):
            text = self.learning.stop_study()
        elif (request := parse_study_request(user_input)) is not None:
            topic, minutes = request
            text = self.learning.start_study(topic, minutes, self._study_writer(), self._study_should_wait)
        else:
            context = self.context_manager.load(session_id)
            if not (context.pending_lookup and answer_kind(user_input) == "no" and len(user_input.split()) <= 6):
                return None
            text = random.choice(_LOOKUP_NO)
        context = context or self.context_manager.load(session_id)
        self._last_user = time.time()
        self.context_manager.append_history(context, user_input, text, "chat")
        self.context_manager.update(context, mode="chat", pending_lookup=None, pending_question=None)
        return {"response": text, "mode": "chat", "confidence": 1.0, "flags": ["no_brain"], "rag_chunks": 0,
                "memories": 0, "language": None, "load": None, "note": None, "file": None,
                "study": self.learning.study_status()}

    def _prepare(self, session_id: str, user_input: str, open_file: dict = None, project: str = None) -> "_Turn":
        """Everything before the brain writes: mode, machine check, memory, open file, project, prompt."""
        context = self.context_manager.load(session_id)
        file    = OpenFile.from_dict(open_file)

        # A yes to "Want me to look that up online?", or "look up X" (Phase 13) — the ask is the permission
        lookup = self._lookup_request(context, user_input)
        if lookup:
            intent = IntentResult(mode="chat", confidence=0.9, flags=["lookup"])
        else:
            intent = self._detect_mode(user_input)
            if file and file.selection.strip() and "fallback" in intent.flags:
                # Selected code and a message no rule places ("hmm?", "this") — same as pasted code
                intent = IntentResult(mode="explain", confidence=0.3, flags=["unclear", "selection_without_request"])
            elif project and "fallback" in intent.flags and file_context.about_project(user_input, file):
                # "where is the config loaded?" — a question about their code, not small talk
                intent = IntentResult(mode="explain", confidence=0.3, flags=["unclear", "project_question"])

        # How busy the computer is decides how much V takes on for this answer
        snap = self._check_machine()
        work = self.machine.work(snap) if snap else None

        # The open file (chat panel): its code goes in when the message is
        # about it; long files in pieces, fewer when the computer is busy
        attached = file_context.attach(
            file, user_input, intent.mode,
            own_code      = bool(CODE.search(user_input)),
            budget        = work.file_chars if work else CFG.files.max_prompt_chars,
            piece_lines   = CFG.files.piece_lines,
            answer_tokens = CFG.model.max_tokens,
            tag           = _file_tag(file),
        )

        # Another language than Python named → her answer in that language
        # (own templates, adapter off); chat mode talks about any language.
        # None named: the open file's language (its code is in the prompt, or a
        # code request with a .ts file open), unless the message says Python.
        language = None
        if intent.mode != "chat":
            language = detect_language(user_input)
            if language is None and not names_python(user_input):
                language = file_context.language_of(file, attached is not None, intent.mode)
        prompt_input = user_input + ("\n\n" + attached.block if attached else "")

        # Project search (10.5): related code from other files of the project
        project_files = []
        if project and intent.mode != "chat" and file_context.about_project(user_input, file):
            budget = work.project_chars if work else CFG.project.max_prompt_chars
            index  = self.project_index(project) if budget > 0 else None
            if index is not None:
                try:
                    from retrieval.project import format_hits
                    hits = index.search(user_input, CFG.project.top_k, skip_path=file.name if attached else None)
                    block, project_files = format_hits(hits, budget)
                    if block:
                        prompt_input += "\n\n" + block
                except Exception as e:
                    logger.warning(f"Project search failed: {e}")

        # Retrieve relevant context and memories before building the prompt
        # (by the message as typed — the file's code would drown it)
        question         = lookup["question"] if lookup else user_input
        retrieved_chunks = self._retrieve(intent.mode, user_input) if not (language or lookup) else []
        memories         = dict(self._recall(intent.mode, question, work))
        if self.learning and not lookup:
            memories["notes"] = self.learning.recall_notes(question)

        # Chat mode on a chat brain: a real conversation (persona, what she
        # knows about the computer, the last turns, the message as written);
        # every other mode: V's templates. A lookup turn's prompt is built in
        # _lookup(), once she has found something.
        formatted = intent.mode == "chat" and uses_chat_format(self.model, self.tokenizer)
        chat_args = {
            "history_turns": work.history_turns if work else None,
            "machine":       format_machine(self.machine.specs, snap) if snap else "",
            "editor":        file_context.describe(file),
        }
        extra = ""
        if self.learning and intent.mode == "chat" and not lookup and _ABOUT_STUDY.search(user_input):
            extra = format_studies(self.learning.topics()) or "You haven't studied any topics on your own yet."
        prompt = ""
        if lookup:
            pass
        elif formatted:
            prompt = build_chat_prompt(user_input, context.to_dict(), memories, self.tokenizer, extra=extra,
                                       **chat_args)
        else:
            prompt = build_prompt(
                mode             = intent.mode,
                user_input       = prompt_input,
                context          = context.to_dict(),
                retrieved_chunks = retrieved_chunks,
                memories         = memories,
                language         = language,
                file_note        = file_context.note(file) if not attached and intent.mode in _CODE_MODES else "",
            )

        generation = {
            "mode":       intent.mode,
            "formatted":  formatted,
            "max_tokens": work.prose_max_tokens if work and intent.mode in ("chat", "explain") else None,
            "adapter":    language is None,
        }
        return _Turn(user_input, context, intent, snap, retrieved_chunks, memories, prompt, generation, language,
                     attached.label if attached else None, lookup=lookup, project_files=project_files,
                     chat_args=chat_args)

    def _lookup_request(self, context: SessionContext, user_input: str):
        """{"query", "question"} when this message is a yes to her lookup offer or asks her to look
        something up; None otherwise. Only the search words ever leave the laptop."""
        if not (self.learning and CFG.learning.web):
            return None
        from learning.lookup import answer_kind, asked_to_look_up
        if context.pending_lookup and answer_kind(user_input) == "yes":
            return {"query": context.pending_lookup, "question": context.pending_question or user_input}
        previous = next((t.content for t in reversed(context.history) if t.role == "user"), "")
        query = asked_to_look_up(user_input, previous)
        if not query:
            return None
        return {"query": query, "question": previous if (_IT_UP.search(user_input) and previous) else user_input}

    def _lookup(self, turn: "_Turn") -> bool:
        """The web part of a lookup turn: what she found goes into the chat prompt. False = nothing found."""
        found = self.learning.lookup(turn.lookup["query"], turn.lookup["question"])
        if not found["found"]:
            return False
        turn.sources = found["sources"]
        block = format_lookup(found["found"], found["sources"])
        if turn.generation["formatted"]:
            turn.prompt = build_chat_prompt(turn.lookup["question"], turn.context.to_dict(), turn.memories,
                                            self.tokenizer, extra=block, **turn.chat_args)
        else:
            turn.prompt = build_prompt("chat", turn.lookup["question"], turn.context.to_dict(),
                                       memories=turn.memories, file_note=block + "\n")
        return True

    def _finish(self, turn: "_Turn", response: str) -> dict:
        """After the brain wrote: save the turn to memory (and tag facts from the
        user's message), offer a lookup when she doesn't know, return the result."""
        if not response.strip():
            response = _EMPTY_WORDS if turn.intent.mode in ("chat", "explain") else _EMPTY_CODE

        # She doesn't know → ask first (Phase 13); she searches only on a yes
        pending, pending_question, asks = None, None, None
        if turn.lookup is None and self.learning and CFG.learning.web:
            from learning.lookup import build_query, offer_lookup
            if offer_lookup(turn.user_input, response, turn.intent.mode) and build_query(turn.user_input):
                response        += "\n\n" + _OFFER
                pending          = build_query(turn.user_input)
                pending_question = turn.user_input
                asks             = _ASKS
        if turn.sources and self.memory and response not in (_LOOKUP_NONE,):
            self._remember_found(turn, response)

        self.context_manager.append_history(turn.context, turn.user_input, response, turn.intent.mode)
        self.context_manager.update(turn.context, mode=turn.intent.mode, pending_lookup=pending,
                                    pending_question=pending_question)

        return {
            "response":      response,
            "mode":          turn.intent.mode,
            "confidence":    turn.intent.confidence,
            "flags":         turn.intent.flags,
            "rag_chunks":    len(turn.retrieved_chunks),   # useful for debugging
            "memories":      len(turn.memories.get("facts", [])) + len(turn.memories.get("code", [])),
            "notes":         len(turn.memories.get("notes", [])),                    # study notes used (Phase 13)
            "language":      turn.language["name"] if turn.language else None,     # None = Python
            "load":          turn.snap.level if turn.snap else None,               # free / busy / tight
            "note":          self.machine.heads_up(turn.snap) if turn.snap else None,   # V's casual heads-up, if any
            "file":          turn.file_label,                                        # what of the open file she read
            "project":       len(turn.project_files),                                # project pieces used (10.5)
            "project_files": turn.project_files,
            "sources":       turn.sources or None,                                   # a lookup's pages (Phase 13)
            "asks":          asks,                                                   # quick replies for the panel
            "study":         self.learning.study_status() if self.learning else None,
        }

    def _remember_found(self, turn: "_Turn", response: str):
        """A lookup's answer, short, with its source — into memory as a fact (newest per question wins)."""
        try:
            sentences = re.split(r"(?<=[.!?])\s+", response.strip())
            summary   = " ".join(sentences[:2])[:240]
            source    = turn.sources[0]
            key       = "web:" + re.sub(r"[^a-z0-9]+", "-", turn.lookup["query"].lower()).strip("-")[:60]
            self.memory.remember_fact(key, f"From the web ({source['title'][:60]}, {time.strftime('%Y-%m-%d')}, "
                                           f"{source['url']}): {summary}")
        except Exception as e:
            logger.warning(f"Could not save the lookup to memory: {e}")


def _file_tag(file) -> str:
    """Code-block tag for the open file's code: python for Python files, else its language's."""
    if file is None:
        return ""
    if file.language_id == "python":
        return "python"
    lang = file_language(file.language_id)
    return lang["tag"] if lang else ""


@dataclass
class _Turn:
    """One message on its way through ChatEngine: what _prepare() worked out."""
    user_input:       str
    context:          SessionContext
    intent:           IntentResult
    snap:             object          # machine.Snapshot or None
    retrieved_chunks: list
    memories:         dict
    prompt:           str
    generation:       dict            # mode / formatted / max_tokens / adapter for the generator
    language:         dict = None     # language_detector result (None = Python)
    file_label:       str  = None     # what of the open file went in ("app.py, lines 10-24"); None = nothing
    lookup:           dict = None     # {"query", "question"} for a lookup turn (Phase 13)
    project_files:    list = field(default_factory=list)   # project pieces that went in ("src/a.py:10-40")
    chat_args:        dict = field(default_factory=dict)   # history_turns / machine / editor for build_chat_prompt
    sources:          list = field(default_factory=list)   # a lookup's pages
