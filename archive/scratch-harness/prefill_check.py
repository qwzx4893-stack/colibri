import json, os, re, subprocess, sys, time, urllib.request, socket

def run(engine, prompt_len_words, max_tokens=1):
    s = socket.socket(); s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]; s.close()
    env = dict(os.environ)
    env['LMA_ENGINE'] = engine
    env['LMA_TEMP'] = '0'
    launcher = r'C:\Users\STS\projects\colibri\c\lmatomic'
    model = r'C:\Users\STS\models\olmoe-colibri'
    srv = subprocess.Popen([sys.executable, launcher, 'serve', '--model', model,
                             '--port', str(port), '--model-id', 'ab'],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for _ in range(240):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2).read(); break
        except Exception:
            time.sleep(0.5)
    prompt = " ".join(["word%d" % i for i in range(prompt_len_words)])
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
    engine = r'C:\Users\STS\projects\colibri\c\olmoe.exe'
    for words in (50, 400, 1200):
        resp, dt, err = run(engine, words, max_tokens=1)
        pt = resp["usage"]["prompt_tokens"]
        m = re.search(r"PROF ([\d.]+) (\d+) (\d+) ([\d.]+) [\d.]+ ([\d.]+) ([\d.]+) ([\d.]+) (\d+)", "\n".join(l for l in err.splitlines() if l.startswith("PROF")))
        prof = m.groups() if m else None
        print(f"prompt_tokens={pt} wall_TTFT_s={dt:.3f} PROF(dt,np,gen,disk_s,attn_s,head_s,forwards)={prof}")
