import sys
sys.path.insert(0, r"C:\Users\STS\lma_ab")
from compute_check import run

engine = r"C:\Users\STS\projects\colibri\c\olmoe.exe"
tok_s1, t1, _ = run(engine, {}, 96, cap=64)
tok_s2, t2, _ = run(engine, {"FUSED3": "1"}, 96, cap=64)
print("baseline:", repr(t1))
print("fused3  :", repr(t2))
print("identical:", t1 == t2)
if t1 != t2:
    for i, (a, b) in enumerate(zip(t1, t2)):
        if a != b:
            print(f"first diff at char {i}: {t1[max(0,i-20):i+20]!r} vs {t2[max(0,i-20):i+20]!r}")
            break
