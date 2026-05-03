# retrieval/retriever.py

import faiss
import pickle
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore


class Retriever:

    # ----------------------------
    # AST TYPE (KEEP AS REQUESTED)
    # ----------------------------
    def detect_ast_type(self, text: str):
        t = text.lower()

        if "merge_sort" in t or "quicksort" in t or "binary search" in t:
            return "algorithm"

        if "init" in t or "__init__" in t:
            return "class"

        if "batch" in t or "dataset" in t:
            return "ml"

        return "general"

    # ----------------------------
    # INIT
    # ----------------------------
    def __init__(self, index_path="retrieval/index"):
        self.embedder = Embedder()
        self.index_path = index_path

        self.store = VectorStore(dim=0)
        self._load_index()

    # ----------------------------
    # INTENT DETECTION
    # ----------------------------
    def detect_intent(self, query: str):
        q = query.lower()

        if any(k in q for k in ["sort", "quick", "merge", "heap"]):
            return "sorting"

        if any(k in q for k in ["binary search", "search"]):
            return "searching"

        if any(k in q for k in ["api", "request", "http", "fastapi"]):
            return "web"

        if any(k in q for k in ["dataset", "batch", "iterator", "dataloader"]):
            return "ml"

        return "general"

    # ----------------------------
    # QUERY EXPANSION (FIXED)
    # ----------------------------
    def expand_query(self, query: str):
        q = query.lower()

        if "quicksort" in q:
            return query + " sorting algorithm divide and conquer partition"

        if "merge sort" in q:
            return query + " sorting algorithm recursion divide conquer"

        if "binary search" in q:
            return query + " search algorithm sorted array divide and conquer"

        return query

    # ----------------------------
    # LOAD INDEX
    # ----------------------------
    def _load_index(self):
        self.store.index = faiss.read_index(f"{self.index_path}/faiss.index")

        with open(f"{self.index_path}/metadata.pkl", "rb") as f:
            self.store.metadata = pickle.load(f)

    # ----------------------------
    # UTILITY
    # ----------------------------
    def _keyword_overlap(self, query, text):
        q_tokens = set(query.lower().split())
        t_tokens = set(text.lower().split())
        return len(q_tokens & t_tokens) / (len(q_tokens) + 1e-5)

    # ----------------------------
    # RERANKER
    # ----------------------------
    def _rerank(self, query, results):
        reranked = []
        intent = self.detect_intent(query)

        for r in results:
            content = r["metadata"].get("content", "")

            faiss_score = r["score"]
            keyword_score = self._keyword_overlap(query, content)

            instruction_boost = 0.05 if "instruction" in content.lower() else 0.0
            content_type = self.detect_ast_type(content)

            type_boost = 0.0
            penalty = 0.0

            # --------------------
            # INTENT LOGIC
            # --------------------

            if intent == "sorting":
                if "def " in content:
                    type_boost = 0.15
                if "dataset" in content or "batch" in content:
                    penalty += 0.6
                if "binary search" in content:
                    penalty += 0.4

            elif intent == "searching":
                if "binary" in content or "search" in content:
                    type_boost = 0.15
                else:
                    penalty += 0.5

            elif intent == "ml":
                if "dataset" in content or "batch" in content:
                    type_boost = 0.15
                else:
                    penalty += 0.4

            final_score = (
                faiss_score
                + 0.25 * keyword_score
                + instruction_boost
                + type_boost
                - penalty
            )

            reranked.append({
                "score": final_score,
                "metadata": r["metadata"]
            })

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked

    # ----------------------------
    # SEARCH PIPELINE
    # ----------------------------
    def search(self, query: str, k=5):
        expanded_query = self.expand_query(query)

        query_emb = self.embedder.encode([expanded_query])

        results = self.store.search(query_emb, k=max(20, k * 4))

        # light filtering (OK here)
        intent = self.detect_intent(query)

        filtered = []
        for r in results:
            content = r["metadata"].get("content", "").lower()

            if intent == "sorting":
                if "dataset" in content or "batch" in content:
                    continue

            if intent == "searching":
                if "merge" in content or "sort" in content:
                    continue

            filtered.append(r)

        return self._rerank(query, filtered)[:k]