# retrieval/embedder.py

import threading

from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    def __init__(self, device: str = None):
        # device=None lets sentence-transformers pick (GPU if present);
        # memory (Phase 11) passes "cpu" — it must never use the GPU
        self.model = SentenceTransformer(MODEL, device=device)

    def encode(self, texts, batch_size: int = 32):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True
        )


_cpu = None
_cpu_lock = threading.Lock()


def cpu_embedder() -> Embedder:
    """
    The one CPU embedder the app shares — memory (Phase 11), project search
    (Phase 10.5), study notes (Phase 13) and RAG over training examples. One
    copy in RAM instead of one per feature (the laptop's RAM runs full).
    Loaded on first use; raises if the model can't load.
    """
    global _cpu
    with _cpu_lock:
        if _cpu is None:
            _cpu = Embedder(device="cpu")
        return _cpu
