from retrieval.retriever import Retriever

r = Retriever()

query = "quicksort implementation"

results = r.search(query, k=3)

print(f"\nQuery: {query}")

for i, res in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Score: {res['score']:.4f}")
    print("Snippet:")
    print(res["metadata"].get("content", "")[:400])