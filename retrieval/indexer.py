# retrieval/indexer.py

import os
import json
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore


CODEBASE_DIR = "model"
DATASET_PATH = "data/datasets/train.jsonl"
INDEX_PATH = "retrieval/index"


def load_codebase():
    for root, _, files in os.walk(CODEBASE_DIR):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)

                try:
                    with open(path, "r", encoding="utf-8") as f:
                        code = f.read()

                    yield {
                        "content": f"CODE FILE:\n{code}",
                        "metadata": {
                            "file": path,
                            "type": "codebase"
                        }
                    }
                except Exception:
                    continue


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            instruction = item.get("instruction", "")
            output = item.get("output", "")

            yield {
                "content": f"INSTRUCTION:\n{instruction}\n\nOUTPUT:\n{output}",
                "metadata": {
                    **item.get("metadata", {}),
                    "type": "dataset"
                }
            }


def build_index():
    embedder = Embedder()
    all_chunks = []

    dataset_count = 0
    codebase_count = 0

    print("Loading dataset...")
    for item in load_dataset():
        all_chunks.append(item)
        dataset_count += 1

    print("Loading codebase...")
    for item in load_codebase():
        all_chunks.append(item)
        codebase_count += 1

    print(f"Dataset chunks: {dataset_count}")
    print(f"Codebase chunks: {codebase_count}")
    print(f"TOTAL chunks: {len(all_chunks)}")

    if not all_chunks:
        print("No data found!")
        return

    texts = [c["content"] for c in all_chunks]

    print("Generating embeddings...")
    embeddings = embedder.encode(texts)

    dim = embeddings.shape[1]
    store = VectorStore(dim)

    print("Building FAISS index...")
    store.add(embeddings, all_chunks)

    print("Saving index...")
    store.save(INDEX_PATH)

    print("Hybrid index built successfully!")


if __name__ == "__main__":
    build_index()