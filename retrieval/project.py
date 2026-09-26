"""
project.py — PY-V (retrieval/)
Search over the user's own project files (chat panel, Phase 10.5 — "RAG over
your code"). The panel sends the workspace folder; its code files are cut in
pieces (inference/engine/file_context.split_pieces: functions, classes,
blocks) and indexed in a local SQLite file under config project.index_dir —
the user's code never leaves the laptop and the folder is gitignored.

  keyword search  SQLite FTS5 over each piece's text and path
  meaning search  the shared CPU embedder (retrieval/embedder.cpu_embedder)

A piece is found only when it is relevant: at least half of the question's
meaningful words (as in memory), or a meaning match ≥ project.min_similarity,
or a code name from the question defined in it.

Which files: `git ls-files` when the folder is a git repository (respects
.gitignore), else a walk that skips the usual build / dependency folders.
Only code and config files up to project.max_file_kb; no minified or binary
files. Updating is incremental (size + modified time) and gentle: it pauses
while V is writing an answer (generator.BRAIN_LOCK) or the computer is busy.
"""

import hashlib
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from memory.store import query_words

logger = logging.getLogger(__name__)

CODE_EXTENSIONS = set("""
py pyi pyx js jsx mjs cjs ts tsx java kt kts scala groovy go rs c h cc cpp cxx hpp hh cs fs fsx vb swift m mm
rb php pl pm lua r jl dart ex exs erl hs ml mli clj cljs elm zig nim v sv vhd vhdl sol sh bash zsh ps1 bat cmd
sql html htm css scss sass less vue svelte astro md rst yaml yml toml ini cfg json gradle cmake mk tf hcl proto
graphql gql asm s
""".split())
SPECIAL_NAMES = {"dockerfile", "makefile", "rakefile", "gemfile", "jenkinsfile", "procfile"}
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env", ".env", "dist", "build",
             "out", ".next", ".nuxt", "target", "bin", "obj", ".idea", ".vscode", ".gradle", ".mypy_cache",
             ".pytest_cache", ".tox", "coverage", ".cache", "site-packages", "vendor", ".terraform"}
MAX_JSON_KB   = 50       # big JSON files are data, not code
MAX_LINE_AVG  = 300      # minified
EMBED_BATCH   = 16
EMBED_CHARS   = 1200

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, mtime REAL);
CREATE TABLE IF NOT EXISTS pieces (
    id INTEGER PRIMARY KEY, path TEXT NOT NULL, start_line INTEGER, end_line INTEGER,
    name TEXT, text TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS pieces_by_path ON pieces(path);
CREATE TABLE IF NOT EXISTS vectors (piece_id INTEGER PRIMARY KEY, vector BLOB NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS pieces_fts USING fts5(text, path, content='pieces', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS pieces_ai AFTER INSERT ON pieces BEGIN
    INSERT INTO pieces_fts(rowid, text, path) VALUES (new.id, new.text, new.path); END;
CREATE TRIGGER IF NOT EXISTS pieces_ad AFTER DELETE ON pieces BEGIN
    INSERT INTO pieces_fts(pieces_fts, rowid, text, path) VALUES ('delete', old.id, old.text, old.path); END;
"""

_DEF_NAME = re.compile(r"^\s*(?:export\s+|public\s+|private\s+|static\s+|async\s+|pub\s+)*"
                       r"(?:def|class|function|fn|func|fun|struct|interface|enum|impl|trait|type)\s+([A-Za-z_$][\w$]*)",
                       re.MULTILINE)
_CODE_NAME = re.compile(r"`([^`\n]+)`|\b([A-Za-z_]\w*)(?=\()|\b(\w*[a-z][A-Z]\w*|\w+_\w+)\b")


@dataclass
class ProjectHit:
    path:       str
    start_line: int
    end_line:   int
    name:       str
    text:       str
    score:      float

    @property
    def label(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


class ProjectIndex:

    def __init__(self, root: str, index_dir: Path, embed: Optional[Callable] = None,
                 max_file_kb: int = 200, max_files: int = 5000, piece_lines: int = 40,
                 min_similarity: float = 0.55):
        self.root           = Path(root).resolve()
        self.embed          = embed
        self.max_file_bytes = max_file_kb * 1024
        self.max_files      = max_files
        self.piece_lines    = piece_lines
        self.min_similarity = min_similarity
        key = hashlib.sha1(str(self.root).lower().encode("utf-8")).hexdigest()[:16]
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = index_dir / f"{self.root.name}-{key}.db"
        self._lock   = threading.Lock()
        self._db     = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()
        self.state   = {"status": "idle", "files": 0, "pieces": 0, "embedded": 0, "updated": None, "error": None}
        self._thread = None
        self._again  = False
        self.refresh_counts()

    # ─── which files ──────────────────────────────────────────────────────────

    def list_files(self) -> List[str]:
        """Project files to index, relative paths with forward slashes."""
        paths = self._git_files()
        if paths is None:
            paths = self._walk()
        wanted = []
        for rel in paths:
            name = rel.rsplit("/", 1)[-1].lower()
            ext  = name.rsplit(".", 1)[-1] if "." in name else ""
            if any(part in SKIP_DIRS for part in rel.split("/")[:-1]):
                continue
            if ext in CODE_EXTENSIONS or name in SPECIAL_NAMES:
                wanted.append(rel)
            if len(wanted) >= self.max_files:
                break
        return wanted

    def _git_files(self) -> Optional[List[str]]:
        if not (self.root / ".git").exists():
            return None
        try:
            out = subprocess.run(["git", "-C", str(self.root), "ls-files", "--cached", "--others", "--exclude-standard"],
                                 capture_output=True, text=True, encoding="utf-8", timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]

    def _walk(self) -> List[str]:
        found = []
        for base, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for f in files:
                found.append(str((Path(base) / f).relative_to(self.root)).replace("\\", "/"))
                if len(found) >= self.max_files * 4:
                    return found
        return found

    def _read(self, rel: str) -> Optional[str]:
        path = self.root / rel
        try:
            size = path.stat().st_size
            if size == 0 or size > self.max_file_bytes:
                return None
            if rel.lower().endswith(".json") and size > MAX_JSON_KB * 1024:
                return None
            data = path.read_bytes()
        except OSError:
            return None
        if b"\0" in data[:4096]:
            return None
        text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
        lines = text.split("\n")
        if len(text) / max(1, len(lines)) > MAX_LINE_AVG:
            return None
        return text

    # ─── updating ─────────────────────────────────────────────────────────────

    def update_in_background(self, pause: Callable[[], bool] = None):
        """Start (or queue) an incremental update. pause() → True = wait a moment (V is answering, laptop busy)."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                self._again = True
                return
            self._thread = threading.Thread(target=self._run, args=(pause,), daemon=True, name="project-index")
            self._thread.start()

    def _run(self, pause):
        while True:
            try:
                self.update(pause)
            except Exception as e:
                logger.warning(f"Project index failed: {e}")
                self.state.update(status="error", error=str(e))
            with self._lock:
                if not self._again:
                    return
                self._again = False

    def update(self, pause: Callable[[], bool] = None) -> dict:
        """Bring the index up to date with the files on disk. Returns the state."""
        self.state.update(status="indexing", error=None)
        files = self.list_files()
        with self._lock:
            known = {r["path"]: (r["size"], r["mtime"]) for r in self._db.execute("SELECT * FROM files")}
        current = set(files)
        for gone in set(known) - current:
            self._forget_file(gone)
        for rel in files:
            path = self.root / rel
            try:
                st = path.stat()
            except OSError:
                continue
            if known.get(rel) == (st.st_size, st.st_mtime):
                continue
            text = self._read(rel)
            self._forget_file(rel)
            with self._lock:
                self._db.execute("INSERT OR REPLACE INTO files (path, size, mtime) VALUES (?, ?, ?)",
                                 (rel, st.st_size, st.st_mtime))
                if text:
                    for start, end, name, piece in self._pieces(text):
                        self._db.execute("INSERT INTO pieces (path, start_line, end_line, name, text) VALUES (?, ?, ?, ?, ?)",
                                         (rel, start, end, name, piece))
                self._db.commit()
        self.refresh_counts()
        self._embed_missing(pause)
        self.state.update(status="ready", updated=time.time())
        self.refresh_counts()
        return dict(self.state)

    def _pieces(self, text: str):
        from inference.engine.file_context import split_pieces
        lines = text.split("\n")
        for a, b in split_pieces(lines, self.piece_lines):
            piece = "\n".join(lines[a:b]).strip("\n")
            if len(piece.strip()) < 20:
                continue
            match = _DEF_NAME.search(piece)
            yield a + 1, b, match.group(1) if match else "", piece

    def _forget_file(self, rel: str):
        with self._lock:
            ids = [r["id"] for r in self._db.execute("SELECT id FROM pieces WHERE path = ?", (rel,))]
            if ids:
                marks = ",".join("?" * len(ids))
                self._db.execute(f"DELETE FROM vectors WHERE piece_id IN ({marks})", ids)
                self._db.execute(f"DELETE FROM pieces WHERE id IN ({marks})", ids)
            self._db.execute("DELETE FROM files WHERE path = ?", (rel,))
            self._db.commit()

    def _embed_missing(self, pause):
        if self.embed is None:
            return
        while True:
            if pause:
                waited = 0
                while pause() and waited < 600:
                    self.state["status"] = "paused"
                    time.sleep(2)
                    waited += 2
                self.state["status"] = "indexing"
            with self._lock:
                rows = self._db.execute(
                    "SELECT id, path, text FROM pieces WHERE id NOT IN (SELECT piece_id FROM vectors) LIMIT ?",
                    (EMBED_BATCH,)).fetchall()
            if not rows:
                return
            vectors = self.embed([f"{r['path']}\n{r['text']}"[:EMBED_CHARS] for r in rows])
            if vectors is None:
                return
            with self._lock:
                for r, v in zip(rows, vectors):
                    self._db.execute("INSERT OR REPLACE INTO vectors (piece_id, vector) VALUES (?, ?)",
                                     (r["id"], np.asarray(v, dtype=np.float32).tobytes()))
                self._db.commit()
            self.refresh_counts()

    def refresh_counts(self):
        with self._lock:
            self.state["files"]    = self._db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            self.state["pieces"]   = self._db.execute("SELECT COUNT(*) FROM pieces").fetchone()[0]
            self.state["embedded"] = self._db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    # ─── searching ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3, skip_path: str = None) -> List[ProjectHit]:
        """The most relevant pieces for a question — only relevant ones (see the module docstring).
        skip_path: the open file (its code already goes in through file reading)."""
        wanted = {_stem(w) for w in query_words(query)} - {""}
        names  = {next(g for g in m.groups() if g).strip() for m in _CODE_NAME.finditer(query)}
        names  = {n for n in names if len(n) > 2}
        scores = {}
        with self._lock:
            if wanted:
                fts = " OR ".join(f'"{w}"*' for w in wanted)     # prefix: "load" finds loaded / loader / load_config
                try:
                    rows = self._db.execute(
                        "SELECT p.id, p.path, p.text, p.name FROM pieces_fts f JOIN pieces p ON p.id = f.rowid "
                        "WHERE pieces_fts MATCH ? ORDER BY bm25(pieces_fts) LIMIT 60", (fts,)).fetchall()
                except sqlite3.OperationalError:
                    rows = []
                for r in rows:
                    share = len(wanted & _tokens(r["text"] + " " + r["path"])) / len(wanted)
                    if share >= 0.5:
                        # the piece that defines it, and the file named after it, before the ones that use it
                        scores[r["id"]] = (scores.get(r["id"], 0) + 0.45 * share
                                           + (0.15 if wanted & _tokens(r["name"] or "") else 0)
                                           + (0.05 if wanted & _tokens(r["path"]) else 0))
            for name in names:
                for r in self._db.execute("SELECT id, text FROM pieces WHERE name = ? LIMIT 5", (name.split(".")[-1],)):
                    scores[r["id"]] = scores.get(r["id"], 0) + 0.6
        if self.embed is not None:
            for pid, sim in self._meaning(query).items():
                scores[pid] = scores.get(pid, 0) + 0.45 * (sim - self.min_similarity) / (1 - self.min_similarity) + 0.05
        if not scores:
            return []
        best = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        hits = []
        with self._lock:
            for pid, score in best:
                r = self._db.execute("SELECT * FROM pieces WHERE id = ?", (pid,)).fetchone()
                if r is None or (skip_path and _same_path(r["path"], skip_path)):
                    continue
                hits.append(ProjectHit(r["path"], r["start_line"], r["end_line"], r["name"] or "", r["text"], round(score, 4)))
                if len(hits) >= top_k:
                    break
        return hits

    def _meaning(self, query: str) -> dict:
        with self._lock:
            rows = self._db.execute("SELECT piece_id, vector FROM vectors").fetchall()
        if not rows:
            return {}
        q = self.embed([query])
        if q is None:
            return {}
        matrix = np.frombuffer(b"".join(r["vector"] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        sims   = matrix @ np.asarray(q[0], dtype=np.float32)
        return {rows[i]["piece_id"]: float(s) for i, s in enumerate(sims) if s >= self.min_similarity}


def _stem(word: str) -> str:
    """Rough word stem, so "loaded" / "loader" / "loading" meet "load" (no dictionary, any language's code)."""
    word = word.lower()
    for end in ("ing", "ed", "er", "s"):
        if word.endswith(end) and len(word) - len(end) >= 3:
            return word[:-len(end)]
    return word


def _tokens(text: str) -> set:
    """Stemmed words of code or text; code names split: load_config / loadConfig → load, config."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return {_stem(w) for w in re.findall(r"[A-Za-z0-9]+", text) if len(w) > 1}


def _same_path(indexed: str, open_name: str) -> bool:
    a, b = indexed.replace("\\", "/").lower(), open_name.replace("\\", "/").lower()
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def format_hits(hits: List[ProjectHit], budget: int) -> tuple:
    """(block for the prompt, labels) — the pieces within `budget` characters, best first."""
    parts, labels, used = [], [], 0
    for h in hits:
        tag   = _tag_for(h.path)
        block = f"From {h.path} (lines {h.start_line}-{h.end_line}):\n```{tag}\n{h.text}\n```"
        if used + len(block) > budget:
            if parts:
                break
            keep  = max(0, budget - 80)
            block = f"From {h.path} (lines {h.start_line}-...):\n```{tag}\n{h.text[:keep]}\n```"
        parts.append(block)
        labels.append(h.label)
        used += len(block)
    if not parts:
        return "", []
    return "Related code from the user's project:\n\n" + "\n\n".join(parts), labels


def _tag_for(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {"py": "python", "js": "javascript", "ts": "typescript", "tsx": "tsx", "jsx": "jsx", "rs": "rust",
            "go": "go", "java": "java", "kt": "kotlin", "cs": "csharp", "cpp": "cpp", "cc": "cpp", "h": "c",
            "c": "c", "rb": "ruby", "php": "php", "sh": "bash", "ps1": "powershell", "yml": "yaml",
            "yaml": "yaml", "md": "markdown"}.get(ext, ext)
