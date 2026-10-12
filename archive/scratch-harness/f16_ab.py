import statistics, sys
sys.path.insert(0, r"C:\Users\STS\lma_ab")
from compute_check import run

engine = r"C:\Users\STS\projects\colibri\c\olmoe.exe"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 64
res = {}
texts = {}
# interleave configs so drift in machine load hits both equally
for r in range(reps):
    for name, env in (("f32", {"LMA_DENSE_F16": "0"}), ("f16", {"LMA_DENSE_F16": "1"})):
        tok_s, text, err = run(engine, env, N, cap=cap)
        res.setdefault(name, []).append(tok_s)
        texts.setdefault(name, []).append(text)
        print(f"rep {r+1} {name}: {tok_s:.3f} tok/s", flush=True)
for name, v in res.items():
    print(f"{name}: median {statistics.median(v):.3f}  min {min(v):.3f}  max {max(v):.3f}")
allt = set(t for v in texts.values() for t in v)
print("TOKEN-EXACT across all runs and both configs:", len(allt) == 1, f"({N} tokens, cap={cap})")
