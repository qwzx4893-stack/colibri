import sys
sys.path.insert(0, r"C:\Users\STS\lma_ab")
from compute_check import run
engine = r"C:\Users\STS\projects\colibri\c\olmoe.exe"
cfgs = {"default": {}, "IDOT=1 (no FUSED3)": {"IDOT": "1"}, "FUSED3=1": {"FUSED3": "1"}}
T = {}
for name, env in cfgs.items():
    tok_s, text, _ = run(engine, env, 200, cap=64)
    T[name] = text
    print(f"{name}: {tok_s:.2f} tok/s", flush=True)
names = list(T)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print(f"{names[i]} == {names[j]}: {T[names[i]] == T[names[j]]}")
