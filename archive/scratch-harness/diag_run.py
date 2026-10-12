import sys
sys.path.insert(0, r"C:\Users\STS\lma_ab")
import gpu_run
# capture LMA_DIAG lines: patch serve_run's err_tail parsing by running and scanning full stderr
import subprocess, os
from gpu_run import serve_run
env = dict(a.split("=",1) for a in sys.argv[2:])
r = serve_run(env, int(sys.argv[1]), warm=True)
print(r["tok_s"], r["gpu_ready"])
print("\n".join(l for l in r["err_tail"].splitlines() if "LMA_DIAG" in l or "circuit" in l or "[GPU]" in l))
