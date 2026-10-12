import json, os, subprocess, sys, time, urllib.request, socket

s = socket.socket(); s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]; s.close()
env = dict(os.environ)
env['LMA_ENGINE'] = r'C:\Users\STS\projects\colibri\c\olmoe.exe'
env['LMA_PREFIX_LOG'] = '1'
env['LMA_TEMP'] = '0'
launcher = r'C:\Users\STS\projects\colibri\c\lmatomic'
model = r'C:\Users\STS\models\olmoe-colibri'
srv = subprocess.Popen([sys.executable, launcher, 'serve', '--model', model,
                         '--port', str(port), '--model-id', 'ab'],
                        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
for _ in range(240):
    try:
        urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2).read()
        break
    except Exception:
        time.sleep(0.5)

def ask(messages, max_tokens=16):
    body = json.dumps({'model': 'ab', 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', body, {'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]

sys_prompt = "You are a terse assistant. Answer in one short sentence."
turn1 = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "What is 2+2?"}]
r1 = ask(turn1)
turn2 = turn1 + [{"role": "assistant", "content": r1}, {"role": "user", "content": "What is 3+3?"}]
r2 = ask(turn2)

srv.terminate()
try:
    out, err = srv.communicate(timeout=20)
except Exception:
    srv.kill()
    out, err = srv.communicate()

print("turn1:", r1)
print("turn2:", r2)
print()
print("=== [PREFIX] lines ===")
for l in err.splitlines():
    if l.startswith("[PREFIX]"):
        print(l)
