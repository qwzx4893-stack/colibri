import json, os, re, subprocess, sys, time, urllib.request, socket

REPO = r"C:\Users\STS\projects\colibri"
LAUNCHER = os.path.join(REPO, "c", "lmatomic")
MODEL = r"C:\Users\STS\models\olmoe-colibri"

PROMPTS = {
    "short": "Explain in one sentence what a hash table is.",
    "long": ("Write a short Python function that computes the nth Fibonacci "
             "number iteratively, then explain its time complexity in two sentences, "
             "then list three real-world uses of Fibonacci numbers."),
}

def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

def run_one(engine_path, label, prompt_key, max_tokens, extra_env=None, cap=None):
    port = free_port()
    env = dict(os.environ)
    env["LMA_ENGINE"] = engine_path
    env["LMA_BYTES_LOG"] = "1"
    env["LMA_TEMP"] = "0"
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, LAUNCHER, "serve", "--model", MODEL, "--port", str(port), "--model-id", "ab"]
    if cap is not None:
        cmd += ["--cap", str(cap)]
    srv = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr_lines = []
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("engine did not come up")
        body = json.dumps({"model": "ab", "messages": [{"role": "user", "content": PROMPTS[prompt_key]}],
                            "max_tokens": max_tokens, "temperature": 0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                      {"Content-Type": "application/json"})
        t0 = time.time()
        resp = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        dt = time.time() - t0
        text = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
    finally:
        srv.terminate()
        err = ""
        try:
            _, err = srv.communicate(timeout=20)
        except Exception:
            srv.kill()
    # The gateway forwards the engine's stderr and interleaves its own [api]
    # log lines onto the same stream without atomic line writes, so a single
    # [LMA_STAT] line can get an [api] line spliced into the middle of it.
    # Strip interleaved [api] lines first, then regex the key=value pairs
    # directly instead of relying on line boundaries.
    clean = "\n".join(l for l in err.splitlines() if not l.startswith("[api]"))
    m = re.search(r"disk_bytes=(\d+)\s+gb_per_token=([\d.]+)\s+pf_issued=(\d+)\s+"
                  r"pf_used=(\d+)\s+pf_precision=([\d.]+)\s+tok_s=([\d.]+)", clean)
    stat = None
    if m:
        stat = {"disk_bytes": int(m.group(1)), "gb_per_token": float(m.group(2)),
                "pf_issued": int(m.group(3)), "pf_used": int(m.group(4)),
                "pf_precision": float(m.group(5)), "engine_tok_s": float(m.group(6))}
    return {"label": label, "prompt": prompt_key, "text": text, "usage": usage,
            "wall_s": dt, "stat": stat, "stderr": err}

CONFIGS = {
    "off": {},
    "pilot": {"PILOT": "3", "WIDE": "2", "HOT": "8", "WARMUP": "5"},
    # Same PILOT/HOT config, but forces the new binary's added mechanisms back
    # to their pre-existing equivalents (gate-only predictor, plain-LRU
    # eviction, single prefetch worker) — an ablation using ONE binary so the
    # comparison isolates exactly what the new mechanisms change, with no
    # baseline-vs-new binary skew.
    "pilot_old_equiv": {"PILOT": "3", "WIDE": "2", "HOT": "8", "WARMUP": "5",
                         "PREDICT_BETA": "0", "SCHED_COST": "0", "PILOT_WORKERS": "1"},
    # Decomposition: workers-only isolates the Asynchronous Expert Pipeline's
    # contribution; predictor-only isolates the Predictive Expert Engine +
    # Multi-Tier Scheduler's contribution, both against the same old_equiv base.
    "pilot_workers_only": {"PILOT": "3", "WIDE": "2", "HOT": "8", "WARMUP": "5",
                            "PREDICT_BETA": "0", "SCHED_COST": "0", "PILOT_WORKERS": "2"},
    "pilot_predictor_only": {"PILOT": "3", "WIDE": "2", "HOT": "8", "WARMUP": "5",
                              "PREDICT_BETA": "0.4", "SCHED_COST": "1", "PILOT_WORKERS": "1"},
}

if __name__ == "__main__":
    # olmoe_baseline_fixed.exe = pre-runtime-work olmoe.c with exactly one
    # change: the two mid-request [HOT] printfs moved from stdout to stderr
    # (a pre-existing protocol-corrupting bug, confirmed to reproduce
    # identically on the untouched binary — see debug_crash.py). Using the
    # untouched binary here would make every HOT+PILOT comparison fail for a
    # reason that has nothing to do with the runtime work being measured.
    baseline = r"C:\Users\STS\olmoe_baseline_fixed.exe"
    new = os.path.join(REPO, "c", "olmoe.exe")
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    prompt_key = sys.argv[2] if len(sys.argv) > 2 else "short"
    cfg_key = sys.argv[3] if len(sys.argv) > 3 else "off"
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else None
    extra_env = CONFIGS[cfg_key]
    results = []
    for label, path in (("baseline", baseline), ("new", new)):
        r = run_one(path, label, prompt_key, max_tokens, extra_env=extra_env, cap=cap)
        results.append(r)
        print(f"=== {label} ({prompt_key}, max_tokens={max_tokens}) ===")
        print(f"wall_s={r['wall_s']:.3f} usage={r['usage']}")
        print(f"stat: {r['stat']}")
        print(f"text: {r['text'][:300]!r}")
        print()
    a, b = results
    match = a["text"] == b["text"]
    print(f"EXACT MATCH: {match}")
    if not match:
        print("--- baseline text ---"); print(a["text"])
        print("--- new text ---"); print(b["text"])
    json.dump(results, open(r"C:\Users\STS\lma_ab\last_result.json", "w"), indent=2)
