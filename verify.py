import json

with open("data/datasets/train.jsonl", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

scores  = [s.get("metadata", {}).get("code_score", 0) for s in samples]
sources = {}
for s in samples:
    src = s.get("metadata", {}).get("source", "unknown")
    sources[src] = sources.get(src, 0) + 1

print(f"Total samples:  {len(samples)}")
print(f"Keys in sample: {list(samples[0].keys())}")
print(f"\n--- Quality Score Distribution ---")
print(f"Min:       {min(scores):.2f}")
print(f"Max:       {max(scores):.2f}")
print(f"Avg:       {sum(scores)/len(scores):.2f}")
print(f"Above 2.0: {sum(1 for s in scores if s >= 2.0)}")
print(f"Above 3.0: {sum(1 for s in scores if s >= 3.0)}")
print(f"\n--- Source Distribution ---")
for src, count in sorted(sources.items()):
    print(f"  {src}: {count}")

print(f"\n--- Sample 1 ---")
print("INSTRUCTION:", samples[0]["instruction"][:150])
print("OUTPUT:\n",    samples[0]["output"][:300])

print(f"\n--- Sample 2 ---")
print("INSTRUCTION:", samples[1]["instruction"][:150])
print("OUTPUT:\n",    samples[1]["output"][:300])