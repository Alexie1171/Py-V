import json

with open("data/datasets/train.jsonl") as f:
    samples = [json.loads(line) for line in f]

print(f"Total samples: {len(samples)}")
print(f"Keys in each sample: {list(samples[0].keys())}")
print(f"\n--- Sample 1 ---")
print("INSTRUCTION:", samples[0]["instruction"][:150])
print("OUTPUT:\n", samples[0]["output"][:300])
print(f"\n--- Sample 2 ---")
print("INSTRUCTION:", samples[1]["instruction"][:150])
print("OUTPUT:\n", samples[1]["output"][:300])