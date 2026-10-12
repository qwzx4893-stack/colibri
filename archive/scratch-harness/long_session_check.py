import json, os, re, subprocess, sys, time, urllib.request, socket

def run(engine, cfg_env, max_tokens, cap=None):
    s = socket.socket(); s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]; s.close()
    env = dict(os.environ)
    env['LMA_ENGINE'] = engine
    env['LMA_BYTES_LOG'] = '1'
    env['LMA_TEMP'] = '0'
    env.update(cfg_env)
    launcher = r'C:\Users\STS\projects\colibri\c\lmatomic'
    model = r'C:\Users\STS\models\olmoe-colibri'
    cmd = [sys.executable, launcher, 'serve', '--model', model, '--port', str(port), '--model-id', 'ab']
    if cap is not None:
        cmd += ['--cap', str(cap)]
    srv = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for _ in range(240):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2).read(); break
        except Exception:
            time.sleep(0.5)
    prompt = ("Write a detailed step by step explanation of how a binary search tree "
              "insertion and deletion works, including balancing considerations, then "
              "compare it to a hash table and a skip list, then discuss when you would "
              "pick each one in a real system design interview, then write example code.")
    body = json.dumps({'model': 'ab', 'messages': [{'role': 'user', 'content': prompt}],
                        'max_tokens': max_tokens, 'temperature': 0}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', body, {'Content-Type': 'application/json'})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    dt = time.time() - t0
    srv.terminate()
    try:
        out, err = srv.communicate(timeout=20)
    except Exception:
        srv.kill(); out, err = srv.communicate()
    return resp, dt, err

if __name__ == "__main__":
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    engine = r'C:\Users\STS\projects\colibri\c\olmoe.exe'
    cfg = {"PILOT": "3", "WIDE": "2", "HOT": "8", "WARMUP": "5", "HOT_REEVAL_TOK": "50"}
    resp, dt, err = run(engine, cfg, max_tokens, cap=cap)
    gen = resp["usage"]["completion_tokens"]
    print(f"tokens={gen} wall_s={dt:.2f} tok_s={gen/dt:.3f}")
    adapt_lines = [l for l in err.splitlines() if "[HOT-ADAPT]" in l]
    print(f"HOT-ADAPT events: {len(adapt_lines)}")
    for l in adapt_lines:
        print(" ", l)
    clean = "\n".join(l for l in err.splitlines() if not l.startswith("[api]"))
    m = re.search(r"disk_bytes=(\d+)\s+gb_per_token=([\d.]+)\s+pf_issued=(\d+)\s+"
                  r"pf_used=(\d+)\s+pf_precision=([\d.]+)", clean)
    if m:
        print("LMA_STAT:", m.group(0))
    print("text tail:", resp["choices"][0]["message"]["content"][-300:])
