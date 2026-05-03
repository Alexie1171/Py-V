# retrieval/indexer.py

import os
import json
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.chunker import chunk_python_file
from model.training.config_loader import CFG

DATASET_PATH = str(CFG.paths.dataset)
INDEX_PATH   = str(CFG.rag.index_path)

# Directories to scan for Python source files
CODEBASE_DIRS = [
    "inference",
    "model",
    "data/scripts",
    "retrieval",
]

# Directories to always skip
EXCLUDE_DIRS = {
    "__pycache__",
    "sessions",
    "experiments",
    "extension",
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
}


def _should_skip(path: str) -> bool:
    parts = set(path.replace("\\", "/").split("/"))
    return bool(parts & EXCLUDE_DIRS)


def load_codebase():
    """
    Yield function/class-level chunks from all Python files across
    CODEBASE_DIRS using AST-based extraction from chunker.py.
    Each chunk is one function or class, not a whole file.
    """
    seen_paths = set()

    for base_dir in CODEBASE_DIRS:
        if not os.path.exists(base_dir):
            print(f"  [skip] {base_dir} — directory not found")
            continue

        for root, dirs, files in os.walk(base_dir):
            # Prune excluded dirs so os.walk doesn't descend into them
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                if not file.endswith(".py"):
                    continue

                path = os.path.join(root, file).replace("\\", "/")

                if path in seen_paths:
                    continue
                if _should_skip(path):
                    continue

                seen_paths.add(path)

                try:
                    with open(path, "r", encoding="utf-8") as f:
                        source = f.read()

                    if not source.strip():
                        continue

                    chunks = chunk_python_file(source, path)

                    if not chunks:
                        print(f"    [no chunks] {path}")
                        continue

                    for chunk in chunks:
                        content = (
                            f"FILE: {path}\n"
                            f"NAME: {chunk['metadata']['name']}\n"
                            f"TYPE: {chunk['metadata']['type']}\n\n"
                            f"{chunk['content']}"
                        )

                        yield {
                            "content": content,
                            "metadata": {
                                **chunk["metadata"],
                                "file":    path,
                                "type":    "codebase",
                                "content": content,
                            }
                        }

                    print(f"    indexed: {path} ({len(chunks)} chunks)")

                except Exception as e:
                    print(f"  [error] {path}: {e}")
                    continue


def load_dataset():
    """Yield instruction/output pairs from the training JSONL."""
    if not os.path.exists(DATASET_PATH):
        print(f"  [skip] Dataset not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            instruction = item.get("instruction", "").strip()
            output      = item.get("output", "").strip()

            if not instruction or not output:
                continue

            content = f"INSTRUCTION:\n{instruction}\n\nOUTPUT:\n{output}"

            yield {
                "content": content,
                "metadata": {
                    **item.get("metadata", {}),
                    "type":    "dataset",
                    "content": content,
                }
            }


def build_index():
    embedder   = Embedder()
    all_chunks = []

    dataset_count  = 0
    codebase_count = 0

    print("Loading dataset...")
    for item in load_dataset():
        all_chunks.append(item)
        dataset_count += 1
    print(f"  → {dataset_count} dataset chunks")

    print("Loading codebase (AST function-level chunks)...")
    for item in load_codebase():
        all_chunks.append(item)
        codebase_count += 1
    print(f"  → {codebase_count} codebase chunks")

    print(f"TOTAL chunks: {len(all_chunks)}")

    if not all_chunks:
        print("No data found — aborting.")
        return

    texts = [c["content"] for c in all_chunks]

    print("Generating embeddings...")
    embeddings = embedder.encode(texts)

    dim   = embeddings.shape[1]
    store = VectorStore(dim)

    print("Building FAISS index...")
    store.add(embeddings, all_chunks)

    print("Saving index...")
    store.save(INDEX_PATH)

    print(f"\nIndex built successfully → {INDEX_PATH}")
    print(f"  dataset:  {dataset_count} chunks")
    print(f"  codebase: {codebase_count} chunks  (was 27 whole-file chunks before)")
    print(f"  total:    {len(all_chunks)} chunks")


if __name__ == "__main__":
    build_index()