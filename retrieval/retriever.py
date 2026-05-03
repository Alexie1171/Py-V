# retrieval/retriever.py

import faiss
import pickle
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore


class Retriever:

    # ----------------------------
    # INIT
    # ----------------------------
    def __init__(self, index_path="retrieval/index"):
        self.embedder   = Embedder()
        self.index_path = index_path
        self.store      = VectorStore(dim=0)
        self._load_index()

    # ----------------------------
    # LOAD INDEX
    # ----------------------------
    def _load_index(self):
        self.store.index = faiss.read_index(f"{self.index_path}/faiss.index")
        with open(f"{self.index_path}/metadata.pkl", "rb") as f:
            self.store.metadata = pickle.load(f)

    # ----------------------------
    # CONTENT EXTRACTION
    # ----------------------------
    def _get_content(self, result: dict) -> str:
        """
        Unified content extraction.
        The indexer stores content at both top-level and metadata["content"].
        Check both so the reranker always has text to work with.
        """
        return (
            result.get("metadata", {}).get("content", "")
            or result.get("content", "")
        ).strip()

    # ----------------------------
    # INTENT DETECTION
    # ----------------------------
    def detect_intent(self, query: str) -> str:
        q = query.lower()

        if any(k in q for k in ["quicksort", "quick sort", "merge sort", "heap sort", "bubble sort", "insertion sort"]):
            return "sorting_specific"

        if any(k in q for k in ["sort", "sorted", "sorting"]):
            return "sorting"

        if any(k in q for k in ["binary search", "bisect", "search"]):
            return "searching"

        if any(k in q for k in ["api", "request", "http", "fastapi", "endpoint"]):
            return "web"

        if any(k in q for k in ["dataset", "batch", "iterator", "dataloader", "train"]):
            return "ml"

        return "general"

    # ----------------------------
    # QUERY EXPANSION
    # ----------------------------
    def expand_query(self, query: str) -> str:
        q = query.lower()

        expansions = {
            "quicksort":     " quicksort quick sort partition pivot divide conquer algorithm",
            "quick sort":    " quicksort quick sort partition pivot divide conquer algorithm",
            "merge sort":    " merge sort mergesort recursion divide conquer algorithm",
            "binary search": " binary search bisect sorted array logarithmic algorithm",
            "heap sort":     " heap sort heapify priority queue algorithm",
            "bubble sort":   " bubble sort swap adjacent comparison algorithm",
        }

        for key, expansion in expansions.items():
            if key in q:
                return query + expansion

        return query

    # ----------------------------
    # KEYWORD OVERLAP
    # ----------------------------
    def _keyword_overlap(self, query: str, text: str) -> float:
        q_tokens = set(query.lower().split())
        t_tokens = set(text.lower().split())
        return len(q_tokens & t_tokens) / (len(q_tokens) + 1e-5)

    # ----------------------------
    # AST TYPE
    # ----------------------------
    def detect_ast_type(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["quicksort", "quick_sort", "merge_sort", "binary_search", "bubble_sort"]):
            return "algorithm"
        if "__init__" in t:
            return "class"
        if any(k in t for k in ["batch", "dataset", "dataloader"]):
            return "ml"
        return "general"

    # ----------------------------
    # RERANKER
    # ----------------------------
    def _rerank(self, query: str, results: list) -> list:
        intent  = self.detect_intent(query)
        reranked = []

        for r in results:
            content      = self._get_content(r)
            faiss_score  = r["score"]
            keyword_score = self._keyword_overlap(query, content)
            type_boost   = 0.0
            penalty      = 0.0

            # Boost dataset chunks that have matching instruction keywords
            instruction_boost = 0.05 if "instruction" in content.lower() else 0.0

            if intent in ("sorting_specific", "sorting"):
                # Penalise list-chunking results heavily — "chunks" != sorting
                if "yield" in content and "chunk" in content:
                    penalty += 0.8
                if "dataset" in content or "batch" in content:
                    penalty += 0.6
                if "binary search" in content or "bisect" in content:
                    penalty += 0.4
                # Reward actual sort implementations
                if any(k in content for k in ["def quick", "def merge", "def bubble", "def heap", "def sort"]):
                    type_boost += 0.3
                if intent == "sorting_specific":
                    # Extra reward for exact algorithm match
                    algo = query.lower().replace(" ", "")
                    if algo in content.lower().replace(" ", "").replace("_", ""):
                        type_boost += 0.4

            elif intent == "searching":
                if any(k in content for k in ["binary", "bisect", "search"]):
                    type_boost += 0.2
                if "merge" in content or "sort" in content:
                    penalty += 0.3

            elif intent == "ml":
                if any(k in content for k in ["dataset", "batch", "dataloader"]):
                    type_boost += 0.2
                else:
                    penalty += 0.3

            final_score = (
                faiss_score
                + 0.3  * keyword_score
                + instruction_boost
                + type_boost
                - penalty
            )

            reranked.append({
                "score":    final_score,
                "metadata": r["metadata"],
                "content":  content,
            })

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked

    # ----------------------------
    # SEARCH PIPELINE
    # ----------------------------
    def search(self, query: str, k: int = 5) -> list:
        expanded   = self.expand_query(query)
        query_emb  = self.embedder.encode([expanded])
        candidates = self.store.search(query_emb, k=max(20, k * 4))

        intent   = self.detect_intent(query)
        filtered = []

        for r in candidates:
            content = self._get_content(r).lower()

            # Hard filter: remove list-chunking noise for sorting queries
            if intent in ("sorting", "sorting_specific"):
                if "yield" in content and "chunk" in content:
                    continue
                if "dataset" in content or "batch" in content:
                    continue

            if intent == "searching":
                if "merge_sort" in content or ("merge" in content and "sort" in content):
                    continue

            filtered.append(r)

        return self._rerank(query, filtered)[:k]