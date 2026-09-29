"""Giant Emulator on the GPU path: real olmoe.exe + real lma_olmoe_cuda.dll against a disk-backed synthetic giant
(`make_giant.py <preset> <dir> <pool_gb> --gpu [--int4]`).

Measured per run: end-to-end tok/s (prefill included, so it is conservative), VRAM expert hit rate, misses/token,
NVMe/page-cache GB read per token, peak RSS. >=3 repetitions per cell, median/min/max. Everything here is labelled
PHYSICALLY EMULATED: weights are random, the I/O volume, tiering and kernels are real. `--drop` sets EXPERT_DROP=1
(no OS page cache reuse of expert blocks: the cold-NVMe lower bound)."""
import argparse, json, os, re, statistics, subprocess, sys, time

ENGINE = os.environ.get("LMA_OLMOE_ENGINE") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "olmoe.exe")
HERE = os.path.dirname(os.path.abspath(__file__))


def make_ref(model_dir, n_prompt=8, n_new=64):
    subprocess.run([sys.executable, os.path.join(HERE, "make_ref.py"), "64", str(n_prompt), str(n_new), os.path.join(model_dir, "ref.json")],
                   check=True, capture_output=True)


def run_once(model_dir, env_extra, timeout=1800):
    env = dict(os.environ, SNAP=model_dir, PPL="1", LMA_TEMP="0", LMA_GPU="1", LMA_GPU_CTX="256")
    env.update(env_extra)
    t0 = time.time()
    p = subprocess.run([ENGINE, "4", "8", os.path.join(model_dir, "ref.json")], env=env, capture_output=True, text=True, timeout=timeout)
    out = p.stdout
    r = {"wall_s": round(time.time() - t0, 1)}
    m = re.search(r"Speed: ([\d.]+) tok/s \(([\d.]+)s for (\d+) tokens\) \| PEAK RSS: ([\d.]+) GB", out)
    if m: r.update(tok_s=float(m[1]), seconds=float(m[2]), tokens=int(m[3]), peak_rss_gb=float(m[4]))
    g = re.search(r"\[GPU_STATS\] on=(\d) slots=(\d+)/(\d+) tokens=(\d+) expert_misses=(\d+) miss_per_token=([\d.]+) fetch_syncs=(\d+) disk_gb_per_token=([\d.]+)", out)
    if g: r.update(gpu_on=int(g[1]), slots=int(g[2]), total_experts=int(g[3]), miss_per_token=float(g[6]), disk_gb_per_token=float(g[8]))
    rd = re.search(r"\[GPU\] ready in ([\d.]+)s: (\d+)/(\d+) experts resident", p.stderr)
    if rd: r["vram_resident"] = f"{rd[2]}/{rd[3]}"
    if "tok_s" not in r: r["error"] = (p.stderr[-300:] + out[-300:])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--label", default="")
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--drop", action="store_true")
    ap.add_argument("--int4", default=None, help="overlay dir: run with LMA_HF_INT4=all and LMA_EXPERT_OVERLAY")
    ap.add_argument("--slots", type=int, default=None, help="cap resident experts (LMA_GPU_SLOTS)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    make_ref(a.model)
    env = {}
    if a.drop: env["EXPERT_DROP"] = "1"
    if a.int4: env.update(LMA_HF_INT4="all", LMA_EXPERT_OVERLAY=a.int4)
    if a.slots: env["LMA_GPU_SLOTS"] = str(a.slots)
    runs = []
    for i in range(a.reps):
        r = run_once(a.model, env); runs.append(r)
        print(f"[{a.label} rep {i+1}/{a.reps}] {r}", flush=True)
    ok = [r["tok_s"] for r in runs if "tok_s" in r]
    res = {"label": a.label, "model": a.model, "env": env, "class": "PHYSICALLY EMULATED (random weights; real tiering, I/O, kernels)",
           "runs": runs}
    if ok:
        res.update(median_tok_s=statistics.median(ok), min_tok_s=min(ok), max_tok_s=max(ok), stdev=statistics.pstdev(ok))
        print(f"== {a.label}: median {res['median_tok_s']:.3f} tok/s (min {res['min_tok_s']:.3f} max {res['max_tok_s']:.3f})", flush=True)
    if a.out: json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
