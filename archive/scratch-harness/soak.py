"""Long-session soak test for the LM Atomic serve path.
usage: soak.py MINUTES OUT_PREFIX [KEY=VAL ...]
Copies olmoe.exe + DLL to an isolated dir (so the repo can be rebuilt meanwhile), serves the real OLMoE model,
sends varied requests back-to-back, and samples engine RSS / handles / threads / VRAM / tok/s after each request.
Greedy decoding => the same prompt must always give the same text: any change is a correctness failure."""
import csv, json, os, shutil, socket, statistics, subprocess, sys, time, urllib.request
import psutil

REPO = r"C:\Users\STS\projects\colibri"
MODEL = r"C:\Users\STS\models\olmoe-colibri"
minutes = float(sys.argv[1]); out = sys.argv[2]
env_extra = dict(a.split("=", 1) for a in sys.argv[3:])
bin_dir = r"C:\Users\STS\lma_ab\soak_bin"
os.makedirs(bin_dir, exist_ok=True)
for f in ("olmoe.exe", "lma_olmoe_cuda.dll"):
    shutil.copy2(os.path.join(REPO, "c", f), os.path.join(bin_dir, f))

PROMPTS = [
    "Explain in one sentence what a hash table is.",
    "Write a Python function that returns the n-th Fibonacci number, with a short docstring.",
    "What is 17 * 23? Show the steps.",
    "Summarise the causes of the French Revolution in three bullet points.",
    "Translate to French: 'The weather is nice today and we are going to the park.'",
    "List five differences between TCP and UDP.",
    "Give me a JSON object describing a book with title, author and year.",
    "Explain the difference between a process and a thread. " * 6,
    "Why is the sky blue? Answer for a ten year old.",
    "Write a haiku about the sea.",
    "What does the Rust borrow checker do?",
    "Describe how quicksort works and its worst-case complexity.",
]
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
env = dict(os.environ)
env.update({"LMA_ENGINE": os.path.join(bin_dir, "olmoe.exe"), "LMA_TEMP": "0", "LMA_BYTES_LOG": "1", "LMA_GPU": "1"})
env.update(env_extra)
errf = open(out + ".stderr.log", "w", encoding="utf-8", errors="replace")
srv = subprocess.Popen([sys.executable, os.path.join(REPO, "c", "lmatomic"), "serve", "--model", MODEL, "--port", str(port),
                        "--model-id", "g", "--cap", "64"], env=env, stdout=subprocess.DEVNULL, stderr=errf)
t_up = time.time()
for _ in range(1800):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read(); break
    except Exception:
        time.sleep(0.5)
print(f"up after {time.time()-t_up:.1f}s", flush=True)


def engine_proc():
    try:
        for c in psutil.Process(srv.pid).children(recursive=True):
            if c.name().lower().startswith("olmoe"):
                return c
    except Exception:
        pass
    return None


def vram_mb():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).split()[0])
    except Exception:
        return -1


def ask(prompt, n=96):
    body = json.dumps({"model": "g", "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); r = json.loads(urllib.request.urlopen(req, timeout=3600).read()); dt = time.time() - t0
    return r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"], dt


ep = engine_proc()
rows = []; ref = {}; mismatches = 0
t_start = time.time(); i = 0
with open(out + ".csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["t_s", "req", "prompt_idx", "gen", "wall_s", "tok_s", "rss_mb", "vms_mb", "handles", "threads", "vram_mb", "same_text"])
    while time.time() - t_start < minutes * 60:
        pi = i % len(PROMPTS)
        text, gen, dt = ask(PROMPTS[pi])
        same = 1
        if pi in ref:
            same = int(ref[pi] == text)
            if not same: mismatches += 1
        else:
            ref[pi] = text
        ep = ep if (ep and ep.is_running()) else engine_proc()
        try:
            mi = ep.memory_info(); rss = mi.rss / 2**20; vms = mi.vms / 2**20; h = ep.num_handles(); th = ep.num_threads()
        except Exception:
            rss = vms = h = th = -1
        row = [round(time.time() - t_start, 1), i, pi, gen, round(dt, 3), round(gen / dt, 2), round(rss, 1), round(vms, 1), h, th, vram_mb(), same]
        w.writerow(row); fh.flush(); rows.append(row)
        if i % 10 == 0:
            print(row, flush=True)
        i += 1
srv.terminate()
try: srv.wait(timeout=30)
except Exception: srv.kill()
errf.close()


def slope_per_hour(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) * 3600 if den else 0.0


t = [r[0] for r in rows]
summ = {"requests": len(rows), "minutes": round((time.time() - t_start) / 60, 1), "text_mismatches": mismatches}
for name, idx in (("rss_mb", 6), ("vms_mb", 7), ("handles", 8), ("threads", 9), ("vram_mb", 10), ("tok_s", 5)):
    ys = [r[idx] for r in rows]
    half = len(ys) // 2
    summ[name] = {"first_quarter_median": round(statistics.median(ys[:max(1, len(ys) // 4)]), 2),
                  "last_quarter_median": round(statistics.median(ys[-max(1, len(ys) // 4):]), 2),
                  "min": min(ys), "max": max(ys), "slope_per_hour": round(slope_per_hour(t, ys), 3)}
json.dump(summ, open(out + ".summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
