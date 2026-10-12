import subprocess, sys, time, urllib.request, json, os, socket

s = socket.socket(); s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]; s.close()
env = dict(os.environ)
env['LMA_ENGINE'] = r'C:\Users\STS\projects\colibri\c\olmoe.exe'
env['LMA_BYTES_LOG'] = '1'
env['LMA_TEMP'] = '0'
env['PILOT'] = '3'
env['WIDE'] = '2'
env['HOT'] = '8'
env['WARMUP'] = '5'
launcher = r'C:\Users\STS\projects\colibri\c\lmatomic'
model = r'C:\Users\STS\models\olmoe-colibri'
srv = subprocess.Popen([sys.executable, launcher, 'serve', '--model', model,
                         '--port', str(port), '--model-id', 'ab', '--cap', '8'],
                        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
for _ in range(240):
    try:
        urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2).read()
        break
    except Exception:
        time.sleep(0.5)
body = json.dumps({'model': 'ab', 'messages': [{'role': 'user', 'content': 'Explain in one sentence what a hash table is.'}],
                    'max_tokens': 64, 'temperature': 0}).encode()
req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', body, {'Content-Type': 'application/json'})
try:
    resp = urllib.request.urlopen(req, timeout=120).read()
    print('OK', resp[:200])
except Exception as e:
    print('FAIL', e)
srv.terminate()
try:
    out, err = srv.communicate(timeout=20)
except Exception:
    srv.kill()
    out, err = srv.communicate()
print('=== FULL STDERR ===')
print(err)
