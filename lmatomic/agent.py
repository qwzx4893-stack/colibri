"""LM Atomic Model Agent: the per-(model, architecture, hardware) runtime profile.

`build_agent()` runs once when a model is added:

  1. inspect     read config.json and every safetensors header (no weights are read),
                 find the layers, the routed experts, their sizes and dtypes, the dense
                 remainder, shared experts and the router
  2. hardware    load or measure the hardware profile (lmatomic.hwprof)
  3. calibrate   drive the real engine over a small multi-domain prompt set with
                 ROUTE_TRACE on, then compute the Routing Locality Fingerprint
  4. decide      pick a strategy from the measured numbers (not from constants)
  5. persist     write agent_profile.json, bound to the model hash, the hardware
                 fingerprint and the tool versions

`load_or_build()` reuses a stored profile only if all three still match; otherwise it
rebuilds. The original model files are read-only inputs and are never modified.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from . import hwprof, routing

SCHEMA = 1
HERE = Path(__file__).resolve().parent
CDIR = HERE.parent / "c"
EXPERT_RE = re.compile(r"^(?P<prefix>.*?layers\.(?P<layer>\d+)\..*?experts\.(?P<expert>\d+)\.)(?P<rest>.+)$")
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
               "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "BOOL": 1}


def read_safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 512 << 20:
            raise ValueError(f"{path}: safetensors header of {n} bytes refused")
        return json.loads(f.read(n))


def model_hash(model_dir: Path):
    """Identity of the model as far as planning is concerned: config, shard names and
    sizes, and the tensor table. Weights are not hashed (fast); a changed tensor shape,
    dtype or shard invalidates the profile."""
    h = hashlib.sha256()
    for name in ("config.json", "generation_config.json"):
        p = model_dir / name
        if p.exists():
            h.update(p.read_bytes())
    for p in sorted(model_dir.glob("*.safetensors")):
        h.update(p.name.encode()); h.update(str(p.stat().st_size).encode())
        hdr = read_safetensors_header(p)
        for k in sorted(k for k in hdr if k != "__metadata__"):
            h.update(k.encode()); h.update(json.dumps(hdr[k].get("shape")).encode()); h.update(hdr[k]["dtype"].encode())
    return h.hexdigest()[:24]


def inspect_model(model_dir):
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tensors = {}
    for p in sorted(model_dir.glob("*.safetensors")):
        hdr = read_safetensors_header(p)
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s = v["data_offsets"]
            tensors[k] = {"shard": p.name, "dtype": v["dtype"], "shape": v["shape"], "bytes": s[1] - s[0]}
    experts = {}                                   # (layer, expert) -> bytes
    expert_dtypes = {}
    dense_bytes = 0
    for name, t in tensors.items():
        m = EXPERT_RE.match(name)
        if m:
            key = (int(m["layer"]), int(m["expert"]))
            experts[key] = experts.get(key, 0) + t["bytes"]
            expert_dtypes[t["dtype"]] = expert_dtypes.get(t["dtype"], 0) + t["bytes"]
        else:
            dense_bytes += t["bytes"]
    layers = sorted({l for l, _ in experts})
    per_layer = {l: sorted(e for ll, e in experts if ll == l) for l in layers}
    sizes = sorted(set(experts.values()))
    n_exp = max((len(v) for v in per_layer.values()), default=0)
    topk = next((cfg[k] for k in ("num_experts_per_tok", "num_experts_per_token", "n_activated_experts", "topk") if k in cfg), None)
    family = None
    try:
        sys.path.insert(0, str(CDIR))
        import family_registry
        family = family_registry.resolve_model(str(model_dir)).descriptor.id
    except Exception as e:                          # unknown family is a valid state, say so
        family = f"unresolved ({type(e).__name__})"
    total = sum(experts.values()) + dense_bytes
    return {
        "model_dir": str(model_dir),
        "family": family, "model_type": cfg.get("model_type"),
        "hidden": cfg.get("hidden_size"), "layers_total": cfg.get("num_hidden_layers"),
        "moe_layers": len(layers), "moe_layer_ids": layers,
        "experts_per_layer": n_exp, "topk": topk,
        "shared_experts": bool(any("shared_expert" in k for k in tensors)),
        "expert_bytes": {"min": sizes[0] if sizes else 0, "max": sizes[-1] if sizes else 0, "distinct_sizes": len(sizes)},
        "expert_dtypes_bytes": expert_dtypes,
        "routed_bytes": sum(experts.values()), "dense_bytes": dense_bytes, "total_bytes": total,
        "bytes_per_token_if_uncached": (sizes[-1] * (topk or 0) * len(layers)) if sizes else 0,
        "tensor_count": len(tensors),
        "shards": sorted({t["shard"] for t in tensors.values()}),
        "router_tensors": sorted(k for k in tensors if re.search(r"\.(mlp|ffn)\.(gate|router)\.weight$", k))[:4],
    }


CALIBRATION = {
    "prose": ["Explain the difference between throughput and latency using a concrete example.",
              "Write a short story about a lighthouse keeper who discovers a message in a bottle."],
    "code": ["Write a Python function that merges two sorted lists and explain its complexity.",
             "Show a C function that reverses a singly linked list in place."],
    "math": ["Solve step by step: if 3x + 7 = 25, what is x? Then compute the derivative of x^3 + 2x.",
             "How many ways can 5 people sit in a row of 5 chairs if two of them refuse to sit together?"],
    "qa": ["What causes the tides on Earth, and why are there usually two high tides a day?",
           "Summarize the main causes of the French Revolution in five bullet points."],
}


def _free_port():
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def run_calibration(model_dir, work_dir, max_tokens=96, domains=None, launcher=None, log=print):
    """One engine run per domain so each gets its own trace. Returns {domain: trace_path}."""
    work_dir = Path(work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    launcher = launcher or str(CDIR / "lmatomic")
    traces = {}
    for dom, prompts in (domains or CALIBRATION).items():
        tpath = work_dir / f"trace_{dom}.txt"
        port = _free_port()
        env = dict(os.environ, ROUTE_TRACE=str(tpath))
        env.pop("PROF", None)
        srv = subprocess.Popen([sys.executable, launcher, "serve", "--model", str(model_dir), "--port", str(port),
                                "--model-id", "cal"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for _ in range(240):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read(); break
                except Exception:
                    time.sleep(0.5)
            else:
                raise RuntimeError("engine did not come up")
            for pr in prompts:
                body = json.dumps({"model": "cal", "messages": [{"role": "user", "content": pr}],
                                   "max_tokens": max_tokens, "temperature": 0}).encode()
                req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                             {"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=900).read()
            log(f"  calibrated {dom}: {len(prompts)} prompts")
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=20)
            except subprocess.TimeoutExpired:
                srv.kill()
        traces[dom] = str(tpath)
    return traces


def decide(graph, fp_all, hw):
    """Turn measured locality and measured hardware into an execution strategy."""
    cm = (hw or {}).get("cost_model", {})
    loc = fp_all.get("locality_class", "typical")
    E = graph["experts_per_layer"]
    expert_b = graph["expert_bytes"]["max"]
    nv = cm.get("nvme_peak_gbs")
    strat = {
        "locality_class": loc,
        "cache_policy": {"high": "lru+frequency-guard, pin session hot set", "typical": "lru with second-chance guard",
                         "low": "plain lru, small cache, spend budget on prefetch depth 1"}[loc],
        "pin_hot_fraction": {"high": 0.5, "typical": 0.25, "low": 0.0}[loc],
        "prefetch": "cross-layer gate prediction, depth 1-2, byte-budgeted",
        "prefill": "expert-major, one layer-wise pass over the whole prompt",
        "decode": "cpu executes RAM-resident misses; gpu executes vram-resident experts",
    }
    if nv and expert_b:
        strat["ideal_miss_read_ms_per_expert"] = round(expert_b / (nv * 1e9) * 1e3, 3)
    if cm.get("h2d_pinned_gbs") and cm.get("cpu_expert_gbs_all_threads") and expert_b:
        # tokens per expert at which GPU (transfer + compute) beats CPU compute, int8 == 1 B/param
        params = expert_b / 1.0
        cpu_rate = cm["cpu_expert_gbs_all_threads"] * 1e9      # weight bytes/s the CPU consumes, memory-bound
        strat["cpu_gpu_crossover_note"] = ("cpu expert time is memory-bound at %.1f GB/s; a transfer at %.1f GB/s costs "
                                           "%.2fx the cpu time of one token" % (cm["cpu_expert_gbs_all_threads"], cm["h2d_pinned_gbs"],
                                                                              cpu_rate / (cm["h2d_pinned_gbs"] * 1e9)))
    return strat


def build_agent(model_dir, out_dir, hw_profile=None, calibrate=True, log=print, **kw):
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("[agent] analyzing architecture")
    graph = inspect_model(model_dir)
    log("[agent] hardware profile")
    hw = hw_profile
    if hw is None:
        hp = out_dir / "hardware_profile.json"
        if hp.exists() and hwprof.is_current(json.loads(hp.read_text(encoding="utf-8"))):
            hw = json.loads(hp.read_text(encoding="utf-8"))
        else:
            hw = hwprof.build_profile(out_dir / "probe", kw.get("disk_gb", 4.0))
            hp.write_text(json.dumps(hw, indent=2), encoding="utf-8")
    fps = {}; fp_all = {}
    if calibrate:
        log("[agent] calibrating routing")
        traces = run_calibration(model_dir, out_dir / "calibration", log=log, **{k: v for k, v in kw.items() if k in ("max_tokens", "domains")})
        merged = None
        for dom, path in traces.items():
            tr = routing.parse(path, graph["experts_per_layer"])
            fps[dom] = routing.fingerprint(tr)
        # domain-mixed fingerprint over the concatenated decode streams
        all_tr = routing.Trace(n_experts=graph["experts_per_layer"])
        for path in traces.values():
            t = routing.parse(path, graph["experts_per_layer"])
            all_tr.forwards.extend(t.forwards); all_tr.layers = t.layers; all_tr.n_layers = t.n_layers
            all_tr.topk = max(all_tr.topk, t.topk)
        fp_all = routing.fingerprint(all_tr)
        doms = list(traces)
        trs = {d: routing.parse(traces[d], graph["experts_per_layer"]) for d in doms}
        cross = {f"{a}~{b}": round(routing.jaccard_hot(trs[a], trs[b]) or 0, 3) for i, a in enumerate(doms) for b in doms[i + 1:]}
        fp_all["cross_domain_hot_jaccard"] = cross
    profile = {
        "schema": SCHEMA, "kind": "model_agent",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "model_hash": model_hash(model_dir),
        "hardware_fingerprint_id": hw["fingerprint_id"],
        "graph": graph,
        "routing_fingerprint": {"all": fp_all, "by_domain": fps},
        "strategy": decide(graph, fp_all, hw) if fp_all else None,
        "build_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "agent_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    log(f"[agent] ready ({profile['build_seconds']} s) -> {out_dir / 'agent_profile.json'}")
    return profile


def load_or_build(model_dir, out_dir, **kw):
    p = Path(out_dir) / "agent_profile.json"
    if p.exists():
        prof = json.loads(p.read_text(encoding="utf-8"))
        hwp = Path(out_dir) / "hardware_profile.json"
        ok = (prof.get("schema") == SCHEMA and prof.get("model_hash") == model_hash(Path(model_dir))
              and hwp.exists() and prof.get("hardware_fingerprint_id") == json.loads(hwp.read_text(encoding="utf-8")).get("fingerprint_id")
              and hwprof.is_current(json.loads(hwp.read_text(encoding="utf-8"))))
        if ok:
            return prof
    return build_agent(model_dir, out_dir, **kw)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="lmatomic agent")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="directory for hardware_profile.json / agent_profile.json / traces")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--disk-gb", type=float, default=4.0)
    a = ap.parse_args(argv)
    prof = load_or_build(a.model, a.out, calibrate=not a.no_calibrate, max_tokens=a.max_tokens, disk_gb=a.disk_gb)
    print(json.dumps({k: prof[k] for k in ("model_hash", "hardware_fingerprint_id", "strategy")}, indent=2))


if __name__ == "__main__":
    main()
