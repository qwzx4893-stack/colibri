"""Quality Governor run: Exact reference vs candidate High-Fidelity profiles on the real OLMoE.
usage: qrun.py OUTDIR name:K=V,K=V [name2:...]   (the Exact reference is always run first)"""
import json, os, subprocess, sys, time
sys.path.insert(0, r"C:\Users\STS\projects\colibri")
from lmatomic import quality as Q

MODEL = r"C:\Users\STS\models\olmoe-colibri"
EXE = r"C:\Users\STS\projects\colibri\c\olmoe.exe"
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
profiles = [("exact", {"LMA_GPU": "1"})]
for a in sys.argv[2:]:
    n, _, kv = a.partition(":")
    profiles.append((n, dict(dict(x.split("=", 1) for x in kv.split(",") if x), LMA_GPU="1")))


def dump(name, env):
    path = os.path.join(out, f"{name}.qdump")
    e = dict(os.environ, SNAP=MODEL, PPL="1", QDUMP=path, LMA_GPU_CTX="2048", LMA_TEMP="0")
    e.update(env)
    r = subprocess.run([EXE, "64", "8", os.path.join(out, "qref.json")], env=e, capture_output=True, text=True, timeout=3600)
    line = next((l for l in r.stdout.splitlines() if l.startswith("TF-NLL")), r.stdout[-300:] + r.stderr[-300:])
    return path, line


results = {}
suites = {}
for name, env in profiles:
    t0 = time.time()
    suites[name] = Q.run_suite(MODEL, env)
    json.dump(suites[name], open(os.path.join(out, f"suite_{name}.json"), "w"), indent=1)
    print(f"[{name}] suite done in {time.time()-t0:.0f}s: {sum(r['ok'] for r in suites[name])}/{len(suites[name])} tasks ok, "
          f"median {sorted(r['tok_s'] for r in suites[name])[len(suites[name])//2]:.1f} tok/s", flush=True)
    if name == "exact":
        n = Q.build_qref(MODEL, suites["exact"], os.path.join(out, "qref.json"))
        print(f"  qref sequence: {n} tokens", flush=True)
    p, line = dump(name, env)
    print("  ", line, flush=True)
    if name != "exact":
        lm = Q.compare_dumps(os.path.join(out, "exact.qdump"), p, os.path.join(out, "qref.json.mask.json"))
        sc = Q.compare_suites(suites["exact"], suites[name])
        v = Q.verdict(lm, sc)
        results[name] = {"env": env, "logit": lm, "suite": sc, "verdict": v,
                         "median_tok_s": sorted(r["tok_s"] for r in suites[name])[len(suites[name]) // 2]}
        print(f"  KL {lm['kl_mean_nats']:.4f}  top1 {lm['top1_agreement']:.3f}  ppl ratio {lm['ppl_ratio']:.3f}  "
              f"task retention {sc['ALL']['success_retention']}  identical {sc['ALL']['identical']}/{sc['ALL']['n']}  "
              f"-> {'ACCEPT' if v['accepted'] else 'REJECT'} {v['reasons']}", flush=True)
json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)
