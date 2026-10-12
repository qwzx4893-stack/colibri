import statistics, sys
sys.path.insert(0, r"C:\Users\STS\lma_ab")
from gpu_run import serve_run
# usage: ab_cfg.py N reps PROMPT_WORDS name:K=V,K=V name2:K=V ...
N = int(sys.argv[1]); reps = int(sys.argv[2]); words = int(sys.argv[3])
cfgs = []
for a in sys.argv[4:]:
    name, _, kv = a.partition(":")
    env = dict(x.split("=", 1) for x in kv.split(",") if x)
    cfgs.append((name, env))
prompt = "Explain in detail how a hash table works, covering hashing, collisions, resizing and complexity. " * max(1, words // 16)
rows = {n: [] for n, _ in cfgs}; walls = {n: [] for n, _ in cfgs}; texts = {n: set() for n, _ in cfgs}
for r in range(reps):
    for name, env in cfgs:
        res = serve_run(env, N, prompt=prompt, warm=True)
        rows[name].append(res["tok_s"]); walls[name].append(res["wall_s"]); texts[name].add(res["text"])
        print(f"rep {r+1} {name}: {res['tok_s']:.2f} tok/s wall {res['wall_s']:.2f}s gen {res['gen']}  pf: {res.get('gpu_pf')}", flush=True)
for n in rows:
    v = rows[n]; w = walls[n]
    print(f"{n}: tok/s median {statistics.median(v):.2f} min {min(v):.2f} max {max(v):.2f} | wall median {statistics.median(w):.2f}s")
base = list(texts.values())[0]
for n in texts: print(n, "text identical to first cfg:", texts[n] == base, "distinct:", len(texts[n]))
