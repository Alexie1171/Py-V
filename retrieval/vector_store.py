# retrieval/vector_store.py

import faiss
import numpy as np
import pickle
import os


class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity (with normalized vectors)
        self.metadata = []

    def add(self, embeddings: np.ndarray, metadatas: list):
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def search(self, query_embedding: np.ndarray, k=5):
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append({
                "score": float(scores[0][i]),
                "metadata": self.metadata[idx]
            })

        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f"{path}/faiss.index")
        with open(f"{path}/metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)