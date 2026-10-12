import subprocess, sys, time, urllib.request, json, os, socket

def try_engine(engine_path, label):
    s = socket.socket(); s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]; s.close()
    env = dict(os.environ)
    env['LMA_ENGINE'] = engine_path
    env['LMA_BYTES_LOG'] = '1'
    env['LMA_TEMP'] = '0'
    env['PILOT'] = '3'
    env['WIDE'] = '2'
    env['HOT'] = '8'
    env['WARMUP'] = '5'
    launcher = r'C:\Users\STS\projects\colibri\c\lmatomic'
    model = r'C:\Users\STS\models\olmoe-colibri'
    srv = subprocess.Popen([sys.executable, launcher, 'serve', '--model', model,
                             '--port', str(port), '--model-id', 'ab'],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ok = False
    for _ in range(240):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2).read()
            ok = True
            break
        except Exception:
            time.sleep(0.5)
    result_text = None
    err_text = None
    if ok:
        body = json.dumps({'model': 'ab', 'messages': [{'role': 'user', 'content': 'Explain in one sentence what a hash table is.'}],
                            'max_tokens': 16, 'temperature': 0}).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', body, {'Content-Type': 'application/json'})
        try:
            resp = urllib.request.urlopen(req, timeout=120).read()
            result_text = resp[:400]
        except Exception as e:
            err_text = str(e)
    srv.terminate()
    try:
        out, err = srv.communicate(timeout=20)
    except Exception:
        srv.kill()
        out, err = srv.communicate()
    print(f'=== {label} === up={ok} result={result_text} req_error={err_text}')
    print('STDOUT tail:', out[-1500:] if out else None)
    print('STDERR tail:', err[-3000:] if err else None)
    print()

try_engine(r'C:\Users\STS\olmoe_baseline_pre_runtime.exe', 'baseline')
try_engine(r'C:\Users\STS\projects\colibri\c\olmoe.exe', 'new')
