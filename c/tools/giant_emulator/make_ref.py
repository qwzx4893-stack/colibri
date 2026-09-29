import json, random, sys

vocab = int(sys.argv[1]) if len(sys.argv) > 1 else 64
n_prompt = int(sys.argv[2]) if len(sys.argv) > 2 else 16
n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 32
out = sys.argv[4] if len(sys.argv) > 4 else "ref.json"
rng = random.Random(1)
prompt_ids = [rng.randrange(vocab) for _ in range(n_prompt)]
full_ids = prompt_ids + [rng.randrange(vocab) for _ in range(n_new)]
json.dump({"prompt_ids": prompt_ids, "full_ids": full_ids}, open(out, "w"))
print(f"wrote {out}: prompt={n_prompt} new={n_new} vocab={vocab}")
