import statistics, sys
sys.path.insert(0, r"C:\Users\STS\lma_ab")
from gpu_run import serve_run

N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
extra = {}
for a in sys.argv[3:]:
    k, v = a.split("=", 1); extra[k] = v
cpu_txt = None
rows = {"cpu": [], "gpu": []}
texts = {"cpu": set(), "gpu": set()}
for r in range(reps):
    for mode, env in (("cpu", {"LMA_GPU": "0"}), ("gpu", dict({"LMA_GPU": "1"}, **extra))):
        res = serve_run(env, N, warm=True)
        rows[mode].append(res["tok_s"]); texts[mode].add(res["text"])
        print(f"rep {r+1} {mode}: {res['tok_s']:.2f} tok/s  startup {res['startup_s']:.1f}s  {res.get('gpu_ready') or ''}", flush=True)
        if mode == "gpu":
            print("   ", res["gpu_line"], flush=True)
for m, v in rows.items():
    print(f"{m}: median {statistics.median(v):.2f}  min {min(v):.2f}  max {max(v):.2f}")
print("cpu==gpu text:", texts["cpu"] == texts["gpu"], "(distinct cpu:", len(texts["cpu"]), " gpu:", len(texts["gpu"]), ")")
if texts["cpu"] != texts["gpu"]:
    a = list(texts["cpu"])[0]; b = list(texts["gpu"])[0]
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print("first diff at char", i, repr(a[max(0, i-30):i+30]), "|", repr(b[max(0, i-30):i+30])); break
