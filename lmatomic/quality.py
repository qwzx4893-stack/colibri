"""Quality Governor: measures what an optimisation profile costs in quality, on the real model, before it is allowed.

Two layers, both real runs of the real engine (nothing modelled):

  1. Task suite (`SUITE`): 7 domains -- coding, reasoning, math, factual QA, instruction following, structured
     output, long context -- each with programmatic checkers. A profile is run free (greedy) through `lmatomic serve`;
     we record per-task success and how far its text agrees with the Exact reference (first-divergence position,
     word-level prefix agreement).
  2. Logit-level metrics (`compare_dumps`): the engine's teacher-forced probe (`QDUMP=<path>` with `PPL=1`) writes
     the log-softmax at every scored position; from a reference dump (Exact) and a candidate dump we compute
     top-1 agreement, KL(ref || cand), and perplexity on the reference continuation.

`verdict()` turns the metrics of a candidate against an explicit `QualityBudget` into accept/reject, so the High-Fidelity
search (`lmatomic/hifi.py` style loops) never accepts a profile on speed alone.
"""
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

CDIR = Path(__file__).resolve().parent.parent / "c"

LONG_FILLER = ("The archive contains many unrelated notes about weather, trains, gardens and old bridges. " * 14)


def _num(text):
    m = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return m


def _json_ok(text, keys):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return False
    try:
        d = json.loads(m.group(0))
    except Exception:
        return False
    return all(k in d for k in keys)


def _py_ok(text, fname, tests):
    """Extract a python function from the answer and run tiny tests in a subprocess with a timeout."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    code = m.group(1) if m else text
    if f"def {fname}" not in code:
        return False
    prog = code + "\n" + "\n".join(f"assert {t}" for t in tests) + "\nprint('OK')\n"
    try:
        r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, timeout=10)
        return r.stdout.strip().endswith("OK")
    except Exception:
        return False


SUITE = [
    # coding
    ("coding", "Write a Python function named square(x) that returns x squared. Reply with only the code.", lambda t: _py_ok(t, "square", ["square(3)==9", "square(-2)==4"])),
    ("coding", "Write a Python function named is_even(n) that returns True if n is even. Reply with only the code.", lambda t: _py_ok(t, "is_even", ["is_even(4)", "not is_even(7)"])),
    ("coding", "Write a Python function named maximum(a, b) returning the larger of a and b. Reply with only the code.", lambda t: _py_ok(t, "maximum", ["maximum(2,5)==5", "maximum(9,1)==9"])),
    # reasoning
    ("reasoning", "Tom is taller than Sam. Sam is taller than Ken. Who is the shortest? Answer with one name.", lambda t: "ken" in t.lower() and "tom" not in t.lower().split("ken")[0][-5:]),
    ("reasoning", "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Answer yes or no.", lambda t: t.strip().lower().startswith("yes")),
    ("reasoning", "I have 3 boxes with 4 apples each. I eat 2 apples. How many apples are left? Answer with a number.", lambda t: "10" in _num(t)),
    # math
    ("math", "What is 12 + 29? Answer with a number only.", lambda t: "41" in _num(t)),
    ("math", "What is 7 times 8? Answer with a number only.", lambda t: "56" in _num(t)),
    ("math", "What is 100 minus 37? Answer with a number only.", lambda t: "63" in _num(t)),
    # factual QA
    ("factual", "What is the capital of France? Answer in one word.", lambda t: "paris" in t.lower()),
    ("factual", "What is the chemical symbol for water? Answer briefly.", lambda t: "h2o" in t.lower().replace(" ", "").replace("₂", "2")),
    ("factual", "Who wrote the play Romeo and Juliet? Answer briefly.", lambda t: "shakespeare" in t.lower()),
    # instruction following
    ("instruction", "Reply with exactly the word BANANA in capital letters and nothing else.", lambda t: t.strip().strip(".").upper() == "BANANA" and "BANANA" in t),
    ("instruction", "List exactly three fruits, one per line, with no other text.", lambda t: len([l for l in t.strip().splitlines() if l.strip()]) == 3),
    ("instruction", "Write one sentence about cats that ends with the word 'purr'.", lambda t: t.strip().rstrip(".!").lower().endswith("purr")),
    # structured output
    ("structured", 'Return a JSON object with keys "name" and "age" for a person called Ana who is 30. Output only JSON.', lambda t: _json_ok(t, ["name", "age"])),
    ("structured", 'Return a JSON object with keys "city" and "country" for Rome. Output only JSON.', lambda t: _json_ok(t, ["city", "country"])),
    ("structured", 'Return a JSON object with a key "items" listing the numbers 1, 2, 3. Output only JSON.', lambda t: _json_ok(t, ["items"])),
    # long context (needle in a filler haystack)
    ("long_context", LONG_FILLER + "The secret code is 7291. " + LONG_FILLER + "\nWhat is the secret code? Answer with the number only.", lambda t: "7291" in _num(t)),
    ("long_context", LONG_FILLER + "The visitor's name is Marguerite. " + LONG_FILLER + "\nWhat is the visitor's name? Answer with one word.", lambda t: "marguerite" in t.lower()),
    ("long_context", LONG_FILLER + "The train leaves at platform 12. " + LONG_FILLER + "\nWhich platform does the train leave from? Answer with the number only.", lambda t: "12" in _num(t)),
]


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def run_suite(model_dir, engine_env=None, max_tokens=80, launcher=None, log=print, suite=None, cap=64):
    """Run the suite greedily in one serve session. Returns [{domain, prompt, text, ok, tok_s}]."""
    launcher = launcher or str(CDIR / "lmatomic")
    port = _free_port()
    env = dict(os.environ, LMA_TEMP="0")
    env.update(engine_env or {})
    srv = subprocess.Popen([sys.executable, launcher, "serve", "--model", str(model_dir), "--port", str(port), "--model-id", "q",
                            "--cap", str(cap)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = []
    try:
        for _ in range(1800):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read(); break
            except Exception:
                time.sleep(0.5)
        for dom, prompt, chk in (suite or SUITE):
            body = json.dumps({"model": "q", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
            t0 = time.time()
            r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
            dt = time.time() - t0
            text = r["choices"][0]["message"]["content"]
            try:
                ok = bool(chk(text))
            except Exception:
                ok = False
            out.append({"domain": dom, "prompt": prompt[:80], "prompt_full": prompt, "text": text, "ok": ok,
                        "tokens": r["usage"]["completion_tokens"], "tok_s": r["usage"]["completion_tokens"] / dt})
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
    return out


def _agree(a, b):
    """word-level agreement: (fraction of the reference's words matched before the first divergence, first-divergence word index)"""
    wa, wb = a.split(), b.split()
    n = 0
    while n < min(len(wa), len(wb)) and wa[n] == wb[n]:
        n += 1
    return (n / len(wa) if wa else 1.0), (n if (n < len(wa) or n < len(wb)) else None)


def compare_suites(ref, cand):
    """Per-domain and overall: task success, success retention vs the reference, exact-text identity, prefix agreement."""
    doms = {}
    for r, c in zip(ref, cand):
        d = doms.setdefault(r["domain"], {"n": 0, "ref_ok": 0, "cand_ok": 0, "identical": 0, "prefix_agree": 0.0})
        d["n"] += 1; d["ref_ok"] += r["ok"]; d["cand_ok"] += c["ok"]
        d["identical"] += int(r["text"] == c["text"]); d["prefix_agree"] += _agree(r["text"], c["text"])[0]
    tot = {"n": 0, "ref_ok": 0, "cand_ok": 0, "identical": 0, "prefix_agree": 0.0}
    for d in doms.values():
        for k in tot: tot[k] += d[k]
        d["prefix_agree"] = round(d["prefix_agree"] / d["n"], 4)
    tot["prefix_agree"] = round(tot["prefix_agree"] / max(1, tot["n"]), 4)
    doms["ALL"] = tot
    for d in doms.values():
        d["success_retention"] = round(d["cand_ok"] / d["ref_ok"], 4) if d["ref_ok"] else None
    return doms


def _engine_encode(model_dir, segments, exe=None):
    """Token ids of each text segment from the engine's own tokenizer (QENCODE probe): exactly what serve mode feeds."""
    import tempfile
    exe = exe or str(CDIR / ("olmoe.exe" if sys.platform == "win32" else "olmoe"))
    with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".seg") as f:
        f.write(b"\x01".join(t.encode("utf-8") for t in segments)); path = f.name
    try:
        r = subprocess.run([exe], env=dict(os.environ, SNAP=str(model_dir), QENCODE=path), capture_output=True, text=True, timeout=120)
    finally:
        os.unlink(path)
    lines = r.stdout.split("\n")
    return [[int(x) for x in ln.split()] for ln in lines[:len(segments)]]


def build_qref(model_dir, suite_results, out_json, max_len=1900, exe=None):
    """Teacher-forcing sequence for the logit-level probe: one chat conversation made of the (short) suite prompts and the
    Exact reference answers. Writes an engine `ref.json` (prompt_ids = first token only) and a sidecar `<out>.mask.json`
    marking the positions that belong to assistant answers (the only ones scored by compare_dumps)."""
    segs = []
    for r in suite_results:
        if r["domain"] == "long_context":
            continue
        lead = "|||IP_ADDRESS|||" + ("" if not segs else "\n")
        segs += [lead + "<|user|>\n" + r["prompt_full"] + "\n<|assistant|>\n", r["text"]]
    enc = _engine_encode(model_dir, segs, exe)
    ids, ans = [], []
    for k in range(0, len(segs), 2):
        u, a = enc[k], enc[k + 1]
        if len(ids) + len(u) + len(a) > max_len:
            break
        ids += u; ans += [0] * len(u)
        ids += a; ans += [1] * len(a)
    Path(out_json).write_text(json.dumps({"prompt_ids": ids[:1], "full_ids": ids}))
    Path(str(out_json) + ".mask.json").write_text(json.dumps(ans[1:]))     # positions 1.. are the scored ones
    return len(ids)


def dump_logits(model_dir, env, qref_json, out_path, exe=None, timeout=3600):
    """Teacher-forced pass of the engine over `qref_json` (see build_qref) with profile `env`, writing the QDUMP file.
    Returns the engine's TF-NLL summary line."""
    exe = exe or str(CDIR / ("olmoe.exe" if sys.platform == "win32" else "olmoe"))
    e = dict(os.environ, SNAP=str(model_dir), PPL="1", QDUMP=str(out_path), LMA_GPU_CTX="2048", LMA_TEMP="0")
    e.update(env or {})
    r = subprocess.run([exe, "64", "8", str(qref_json)], env=e, capture_output=True, text=True, timeout=timeout)
    return next((l for l in r.stdout.splitlines() if l.startswith("TF-NLL")), None)


def read_dump(path):
    """QDUMP file: int32 vocab, int32 n, then n records of (int32 target, float32 logprob[vocab])."""
    import numpy as np
    with open(path, "rb") as f:
        vocab, n = [int(x) for x in np.frombuffer(f.read(8), dtype=np.int32)]
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    rec = 4 + 4 * vocab
    raw = raw[: n * rec].reshape(n, rec)
    tgt = raw[:, :4].copy().view(np.int32).reshape(n)
    lp = raw[:, 4:].copy().view(np.float32).reshape(n, vocab)
    return tgt, lp


def compare_dumps(ref_path, cand_path, mask_path=None):
    """Logit-level divergence of a candidate from the Exact reference over the same teacher-forced positions
    (optionally only the assistant-answer positions given by the sidecar mask)."""
    import numpy as np
    tr, lr = read_dump(ref_path)
    tc, lc = read_dump(cand_path)
    assert (tr == tc).all(), "dumps were taken on different token sequences"
    if mask_path:
        mk = np.array(json.loads(Path(mask_path).read_text()), dtype=bool)[: len(tr)]
        tr, lr, tc, lc = tr[mk], lr[mk], tc[mk], lc[mk]
    pr = np.exp(lr.astype(np.float64))
    kl = (pr * (lr - lc)).sum(axis=1)
    idx = np.arange(len(tr))
    nll_r = -lr[idx, tr].astype(np.float64); nll_c = -lc[idx, tc].astype(np.float64)
    return {"positions": int(len(tr)),
            "top1_agreement": float((lr.argmax(1) == lc.argmax(1)).mean()),
            "kl_mean_nats": float(kl.mean()), "kl_p95_nats": float(np.percentile(kl, 95)), "kl_max_nats": float(kl.max()),
            "ppl_ref": float(math.exp(nll_r.mean())), "ppl_cand": float(math.exp(nll_c.mean())),
            "ppl_ratio": float(math.exp(nll_c.mean() - nll_r.mean()))}


@dataclass
class QualityBudget:
    """What a profile is allowed to cost. Defaults are deliberately strict: a profile has to be nearly indistinguishable."""
    min_top1_agreement: float = 0.97
    max_kl_mean_nats: float = 0.02
    max_ppl_ratio: float = 1.03
    min_success_retention: float = 0.95


def verdict(logit_metrics, suite_cmp, budget=None):
    b = budget or QualityBudget()
    reasons = []
    if logit_metrics:
        if logit_metrics["top1_agreement"] < b.min_top1_agreement: reasons.append(f"top-1 agreement {logit_metrics['top1_agreement']:.3f} < {b.min_top1_agreement}")
        if logit_metrics["kl_mean_nats"] > b.max_kl_mean_nats: reasons.append(f"mean KL {logit_metrics['kl_mean_nats']:.4f} > {b.max_kl_mean_nats}")
        if logit_metrics["ppl_ratio"] > b.max_ppl_ratio: reasons.append(f"ppl ratio {logit_metrics['ppl_ratio']:.3f} > {b.max_ppl_ratio}")
    if suite_cmp:
        r = suite_cmp["ALL"]["success_retention"]
        if r is not None and r < b.min_success_retention: reasons.append(f"task-success retention {r:.3f} < {b.min_success_retention}")
    return {"accepted": not reasons, "reasons": reasons}
