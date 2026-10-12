"""Per-layer sensitivity map for int4 experts + greedy mixed-precision profiles (Quality Governor, real OLMoE).
usage: sens.py OUTDIR REFDIR   (REFDIR holds exact.qdump / qref.json / qref.json.mask.json from qrun.py)"""
import json, os, subprocess, sys, time
sys.path.insert(0, r"C:\Users\STS\projects\colibri")
from lmatomic import quality as Q

MODEL = r"C:\Users\STS\models\olmoe-colibri"
EXE = r"C:\Users\STS\projects\colibri\c\olmoe.exe"
out, ref = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
REF = os.path.join(ref, "exact.qdump"); QREF = os.path.join(ref, "qref.json"); MASK = QREF + ".mask.json"


def measure(name, env):
    path = os.path.join(out, name + ".qdump")
    e = dict(os.environ, SNAP=MODEL, PPL="1", QDUMP=path, LMA_GPU_CTX="2048", LMA_TEMP="0", LMA_GPU="1")
    e.update(env)
    subprocess.run([EXE, "64", "8", QREF], env=e, capture_output=True, text=True, timeout=3600)
    m = Q.compare_dumps(REF, path, MASK)
    os.remove(path)
    return m


res = {"per_layer": {}, "greedy": {}}
for l in range(16):
    t0 = time.time()
    m = measure(f"L{l}", {"LMA_HF_INT4": str(l)})
    res["per_layer"][l] = m
    print(f"layer {l:2d}: KL {m['kl_mean_nats']:.5f}  top1 {m['top1_agreement']:.3f}  ppl ratio {m['ppl_ratio']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
order = sorted(range(16), key=lambda l: res["per_layer"][l]["kl_mean_nats"])
print("least -> most sensitive:", order, flush=True)
res["order"] = order
for k in (4, 8, 12):
    layers = ",".join(str(l) for l in order[:k])
    m = measure(f"k{k}", {"LMA_HF_INT4": layers})
    res["greedy"][k] = {"layers": order[:k], "metrics": m}
    print(f"int4 on the {k} least sensitive layers {sorted(order[:k])}: KL {m['kl_mean_nats']:.5f} top1 {m['top1_agreement']:.3f} "
          f"ppl ratio {m['ppl_ratio']:.4f} -> {'ACCEPT' if Q.verdict(m, None)['accepted'] else 'REJECT'}", flush=True)
json.dump(res, open(os.path.join(out, "sensitivity.json"), "w"), indent=1)
