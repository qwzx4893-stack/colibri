"""Giant Emulator benchmark with proper statistics: N repetitions per
(preset, locality) cell, reporting median/min/max instead of a single noisy
sample -- fixes the explicit complaint that the previous single-shot runs
could not distinguish a real effect from this machine's run-to-run noise
(already shown elsewhere to be 30-60% at comparable timescales)."""
import json, os, statistics, subprocess, sys, time

ENGINE = os.environ.get("LMA_OLMOE_ENGINE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "olmoe" + (".exe" if os.name == "nt" else ""))
LAUNCHER = None  # unused: this drives olmoe.exe directly via the CLI ref.json path, not serve mode


def run_once(model_dir, cap, stick, tokens=32):
    env = dict(os.environ)
    env["SNAP"] = model_dir
    env["EXPERT_DROP"] = "1"
    env["LMA_TEMP"] = "0"
    env["LMA_BYTES_LOG"] = "1"
    env["PILOT"] = "3"; env["WIDE"] = "2"; env["HOT"] = "8"; env["WARMUP"] = "5"
    if stick > 0:
        env["LMA_GIANT_STICK"] = str(stick)
    hot_pin = os.path.join(model_dir, "hot_pinned.bin")
    if os.path.exists(hot_pin):
        os.remove(hot_pin)
    t0 = time.time()
    proc = subprocess.run([ENGINE, str(cap), "8", "ref.json"], cwd=model_dir, env=env,
                          capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    tok_s = None
    hit = None
    for line in proc.stdout.splitlines():
        if line.startswith("Speed:"):
            # "Speed: 0.39 tok/s (48.2s for 32 tokens)"
            tok_s = float(line.split()[1])
        if line.startswith("Expert cache hit rate:"):
            hit = float(line.split()[4].rstrip("%"))
    return {"tok_s": tok_s, "hit_pct": hit, "wall_s": round(dt, 1)}


def bench(model_dir, cap, stick, reps, tokens=32, label=""):
    samples = []
    for i in range(reps):
        r = run_once(model_dir, cap, stick, tokens)
        samples.append(r)
        print(f"  [{label} rep {i+1}/{reps}] tok_s={r['tok_s']} hit={r['hit_pct']}% wall={r['wall_s']}s")
    tok_s_vals = [s["tok_s"] for s in samples if s["tok_s"] is not None]
    hit_vals = [s["hit_pct"] for s in samples if s["hit_pct"] is not None]
    if not tok_s_vals:
        return {"label": label, "samples": samples, "error": "no successful runs"}
    return {
        "label": label, "reps": len(tok_s_vals),
        "median_tok_s": statistics.median(tok_s_vals),
        "min_tok_s": min(tok_s_vals), "max_tok_s": max(tok_s_vals),
        "stdev_tok_s": statistics.stdev(tok_s_vals) if len(tok_s_vals) > 1 else 0.0,
        "median_hit_pct": statistics.median(hit_vals) if hit_vals else None,
        "samples": samples,
    }


if __name__ == "__main__":
    model_dir = sys.argv[1]   # a directory built by make_giant.py (+ make_ref.py's ref.json)
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    results = []
    for name, stick in (("hostile", 0.0), ("typical", 0.30), ("favorable", 0.55)):
        r = bench(model_dir, cap, stick, reps, label=name)
        results.append(r)
        print(f"=== {name}: median={r.get('median_tok_s')} tok/s "
              f"(min={r.get('min_tok_s')}, max={r.get('max_tok_s')}, "
              f"stdev={r.get('stdev_tok_s')}, n={r.get('reps')}) hit={r.get('median_hit_pct')}% ===")
        print()
    json.dump(results, open(os.path.join(os.path.dirname(model_dir), "giant_bench_result.json"), "w"), indent=2)
