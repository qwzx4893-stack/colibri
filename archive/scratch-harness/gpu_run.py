import json, os, re, subprocess, sys, time, urllib.request, socket

REPO = r"C:\Users\STS\projects\colibri"
MODEL = r"C:\Users\STS\models\olmoe-colibri"


def serve_run(env_extra, max_tokens, prompt="Explain in one sentence what a hash table is.", cap=64, warm=False, timeout_up=900):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    env = dict(os.environ)
    env.update({"LMA_ENGINE": os.path.join(REPO, "c", "olmoe.exe"), "LMA_TEMP": "0", "LMA_BYTES_LOG": "1"})
    env.update(env_extra)
    cmd = [sys.executable, os.path.join(REPO, "c", "lmatomic"), "serve", "--model", MODEL, "--port", str(port), "--model-id", "g"]
    if cap is not None:
        cmd += ["--cap", str(cap)]
    t_start = time.time()
    srv = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    up = None
    for _ in range(int(timeout_up * 2)):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read(); up = time.time() - t_start; break
        except Exception:
            time.sleep(0.5)
    res = {"startup_s": up}
    try:
        def ask(msgs, n):
            body = json.dumps({"model": "g", "messages": msgs, "max_tokens": n, "temperature": 0}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
            t0 = time.time(); r = json.loads(urllib.request.urlopen(req, timeout=3600).read()); dt = time.time() - t0
            return r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"], dt
        msgs = [{"role": "user", "content": prompt}]
        if warm:
            ask(msgs, 16)
        text, gen, dt = ask(msgs, max_tokens)
        res.update(text=text, gen=gen, wall_s=dt, tok_s=gen / dt)
    finally:
        srv.terminate()
        try:
            _, err = srv.communicate(timeout=30)
        except Exception:
            srv.kill(); _, err = srv.communicate()
    clean = "\n".join(l for l in err.splitlines() if not l.startswith("[api]"))
    lines = clean.splitlines()
    gl = [l for l in lines if "[LMA_GPU]" in l]
    res["gpu_line"] = gl[-1] if gl else None
    ql = [l for l in lines if "[LMA_GPU_PF]" in l]
    res["gpu_pf"] = ql[-1] if ql else None
    sl = [l for l in lines if "[LMA_SPEC]" in l]
    res["spec_line"] = sl[-1] if sl else None
    pl = [l for l in lines if "[LMA_GPU_PROF]" in l]
    res["gpu_prof"] = pl[-1] if pl else None
    res["gpu_ready"] = next((l for l in clean.splitlines() if l.startswith("[GPU]")), None)
    res["err_tail"] = clean[-1500:]
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    env = {"verify": {"LMA_GPU": "1", "LMA_GPU_VERIFY": "1"}, "gpu": {"LMA_GPU": "1"}, "cpu": {"LMA_GPU": "0"}}[mode]
    r = serve_run(env, n)
    print(json.dumps({k: v for k, v in r.items() if k != "text" and k != "err_tail"}, indent=1))
    print("PF:", r.get("gpu_pf"))
    print("TEXT:", repr(r.get("text", "")[:300]))
    if not r.get("gpu_line") and mode != "cpu":
        print("ERR TAIL:", r["err_tail"])
