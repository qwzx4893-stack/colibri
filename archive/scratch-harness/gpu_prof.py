import sys
sys.path.insert(0, r"C:\Users\STS\lma_ab")
from gpu_run import serve_run
env = {"LMA_GPU": "1", "LMA_GPU_PROF": "1"}
for a in sys.argv[2:]:
    k, v = a.split("=", 1); env[k] = v
r = serve_run(env, int(sys.argv[1]) if len(sys.argv) > 1 else 200, warm=True)
print(f"{r['tok_s']:.2f} tok/s ({r['gen']} tokens)")
print(r["gpu_ready"]); print(r["gpu_line"]); print(r["gpu_prof"])
