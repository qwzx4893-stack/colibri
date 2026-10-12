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
    body = json.dumps({'model': 'ab', 'messages': [{'role': 'user', 'content': 'Explain in one sentence what a hash table is.'}],
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
    gen = resp["usage"]["completion_tokens"]
    text = resp["choices"][0]["message"]["content"]
    return gen / dt, text, err

if __name__ == "__main__":
    engine = r'C:\Users\STS\projects\colibri\c\olmoe.exe'
    configs = {
        "baseline (scalar matmul_q)": {},
        "FUSED3=1 (AVX2 gate+up fused, opt-in)": {"FUSED3": "1"},
        "FUSED3=1 IDOT=1 (explicit, same default)": {"FUSED3": "1", "IDOT": "1"},
    }
    texts = {}
    for name, cfg in configs.items():
        tok_s, text, err = run(engine, cfg, 96, cap=64)
        texts[name] = text
        print(f"{name}: {tok_s:.3f} tok/s")
    print()
    all_texts = set(texts.values())
    print("ALL IDENTICAL OUTPUT:", len(all_texts) == 1)
    if len(all_texts) != 1:
        for k, v in texts.items():
            print(f"--- {k} ---")
            print(v[:200])
