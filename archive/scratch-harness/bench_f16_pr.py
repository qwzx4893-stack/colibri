"""Interleaved A/B for the half-resident dense-weight PR: COLI_DENSE_F16=0 (baseline) vs 1 (trial), upstream dev build.
Speed = olmoe PPL-mode 'Speed:' line (teacher-forced decode, prefill included, same for both arms); quality = identical
TF-NLL and identical greedy token stream (generate mode). Writes the raw log used as manifest evidence."""
import json, os, re, subprocess, sys, time, statistics
EXE = r"C:\Users\STS\projects\colibri-pr\c\olmoe.exe"
MODEL = r"C:\Users\STS\models\olmoe-colibri"
src = json.load(open(r"C:\Users\STS\lma_ab\q3\qref.json"))["full_ids"]
ref = r"C:\Users\STS\lma_ab\ref_pr.json"
json.dump({"prompt_ids": src[:24], "full_ids": src[:24 + 256]}, open(ref, "w"))
refg = r"C:\Users\STS\lma_ab\ref_pr_gen.json"
json.dump({"prompt_ids": src[:24], "full_ids": src[:24 + 128]}, open(refg, "w"))
log = open(r"C:\Users\STS\lma_ab\f16_pr_raw.txt", "w", encoding="utf-8")


def run(f16, mode, r):
    env = dict(os.environ, SNAP=MODEL, COLI_TEMP="0", COLI_DENSE_F16=str(f16))
    if mode == "ppl":
        env["PPL"] = "1"
    t0 = time.time()
    p = subprocess.run([EXE, "64", "8", r], env=env, capture_output=True, text=True, timeout=3600)
    log.write(f"=== COLI_DENSE_F16={f16} mode={mode} wall={time.time()-t0:.1f}s\n{p.stdout}\n{p.stderr[-300:]}\n"); log.flush()
    return p.stdout


res = {0: [], 1: []}; rss = {0: [], 1: []}; nll = {0: set(), 1: set()}
for rep in range(9):                              # rep 0 is the warm-up pair, not counted
    for f16 in ((0, 1) if rep % 2 == 0 else (1, 0)):
        out = run(f16, "ppl", ref)
        m = re.search(r"Speed: ([\d.]+) tok/s", out); n = re.search(r"TF-NLL: ([\d.]+)", out); r_ = re.search(r"RSS after load: ([\d.]+)", out)
        print(f"rep {rep} F16={f16}: {m[1]} tok/s  nll {n[1]}  rss_after_load {r_[1]} GB", flush=True)
        if rep > 0:
            res[f16].append(float(m[1])); rss[f16].append(float(r_[1])); nll[f16].add(n[1])
toks = {}
for f16 in (0, 1):
    out = run(f16, "gen", refg)
    toks[f16] = re.search(r"C engine : (.*)", out)[1].strip()
print("greedy token streams identical:", toks[0] == toks[1], len(toks[0].split()), "tokens")
for f16 in (0, 1):
    print(f"F16={f16}: tok/s {res[f16]} median {statistics.median(res[f16]):.3f}  rss_after_load {rss[f16]}  nll {nll[f16]}")
json.dump({"samples": res, "rss": rss, "nll": {k: sorted(v) for k, v in nll.items()}, "tokens_identical": toks[0] == toks[1]},
          open(r"C:\Users\STS\lma_ab\f16_pr_summary.json", "w"), indent=1)
