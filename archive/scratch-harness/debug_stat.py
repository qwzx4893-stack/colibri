import sys
sys.path.insert(0, r"C:\Users\STS\lma_ab")
from ab_bench import run_one, CONFIGS

NEW = r"C:\Users\STS\projects\colibri\c\olmoe.exe"
r = run_one(NEW, "pilot", "short", 64, extra_env=CONFIGS["pilot"], cap=8)
print("wall_s", r["wall_s"])
print("stat", r["stat"])
print("text", r["text"][:200])
