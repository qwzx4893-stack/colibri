import json, os, subprocess, sys, time, urllib.request, socket

def run(preset):
    s = socket.socket(); s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]; s.close()
    env = dict(os.environ)
    env['LMA_ENGINE'] = r'C:\Users\STS\projects\colibri\c\olmoe.exe'
    env['LMA_TEMP'] = '0'
    launcher = r'C:\Users\STS\projects\colibri\c\lmatomic'
    model = r'C:\Users\STS\models\olmoe-colibri'
    cmd = [sys.executable, launcher, 'serve', '--model', model, '--port', str(port),
           '--model-id', 'ab', '--preset', preset]
    srv = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for _ in range(240):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2).read(); break
        except Exception:
            time.sleep(0.5)
    body = json.dumps({'model': 'ab', 'messages': [{'role': 'user', 'content': 'Say hi in one word.'}],
                        'max_tokens': 8, 'temperature': 0}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', body, {'Content-Type': 'application/json'})
    ok, err_text = None, None
    try:
        ok = urllib.request.urlopen(req, timeout=120).read()
    except Exception as e:
        err_text = str(e)
    srv.terminate()
    try:
        out, err = srv.communicate(timeout=20)
    except Exception:
        srv.kill(); out, err = srv.communicate()
    print(f"=== preset={preset} ===  ok={ok is not None}  err={err_text}")
    for l in err.splitlines():
        if l.startswith("== Streaming"):
            print(" ", l)

for p in ("auto", "single-user", "throughput"):
    run(p)
