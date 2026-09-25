# retrieval/embedder.py

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, device: str = None):
        # device=None lets sentence-transformers pick (GPU if present);
        # memory (Phase 11) passes "cpu" — it must never use the GPU
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    def encode(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
