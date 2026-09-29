"""Digital Twin: a trace-driven, resource-timed simulator of MoE offloading.

It does NOT multiply numbers. Every simulated token walks every MoE layer, asks the
tiered cache where each routed expert lives (VRAM / RAM / NVMe), issues real
(simulated-time) reads on a single shared NVMe queue whose bandwidth comes from the
MEASURED disk grid, runs CPU or GPU expert compute from the MEASURED kernels, applies
the prefetch policy under a byte budget, and updates the caches. Output is tokens/s,
NVMe bytes per generated token, hit rates per tier, stall time and prefetch
precision/recall/wasted bytes.

Labels (used verbatim in every report built on this module):
    MEASURED   produced by running the real engine or a probe on this machine
    SIMULATED  produced by this twin, driven by MEASURED hardware numbers and a real trace
    PROJECTED  produced by this twin driven by a SYNTHETIC routing stream (any model size
               larger than what was actually run)

Routing sources
    trace replay      a recorded ROUTE_TRACE stream (SIMULATED)
    SyntheticRouting  a generator fitted to a measured Routing Locality Fingerprint, with
                      "favorable" / "typical" / "hostile" presets (PROJECTED)
"""
from __future__ import annotations

import collections
import math
import random
from dataclasses import dataclass, field

# ------------------------------------------------------------------ model / hardware


@dataclass
class ModelSpec:
    name: str
    layers: int                # MoE layers
    experts: int               # routed experts per layer
    topk: int
    expert_bytes: float        # stored bytes of one routed expert (chosen precision)
    dense_bytes: float         # resident dense bytes (attention, shared experts, embeddings)
    params_total: float = 0.0
    params_active: float = 0.0

    @property
    def routed_bytes(self):
        return self.layers * self.experts * self.expert_bytes

    @property
    def bytes_per_token_uncached(self):
        return self.layers * self.topk * self.expert_bytes


def giant(name, total_b, dense_b, layers, experts, topk, bits=4.5):
    """A hypothetical MoE of `total_b` billion parameters. Expert size follows from the
    total; quantisation `bits` per parameter includes scales (Q4 ~ 4.5)."""
    routed = total_b * 1e9 - dense_b * 1e9
    pe = routed / (layers * experts)
    active = dense_b * 1e9 + layers * topk * pe
    return ModelSpec(name, layers, experts, topk, pe * bits / 8, dense_b * 1e9 * bits / 8, total_b * 1e9, active)


GIANTS = {
    "100B/A~9B": dict(total_b=100, dense_b=2, layers=48, experts=128, topk=8),
    "300B/A~14B": dict(total_b=300, dense_b=3, layers=60, experts=256, topk=8),
    "500B/A~8B": dict(total_b=500, dense_b=2, layers=60, experts=640, topk=8),
    "500B/A~16B": dict(total_b=500, dense_b=4, layers=60, experts=320, topk=8),
    "500B/A~31B": dict(total_b=500, dense_b=6, layers=90, experts=160, topk=8),
}


def make_giant(key, bits=4.5):
    return giant(key, bits=bits, **GIANTS[key])


@dataclass
class Hardware:
    nvme_grid: dict            # {threads: [(block_mb, gbs), ...]} sorted by block
    h2d_gbs: float
    ram_gbs: float
    cpu_gbs: float             # expert weight bytes/s the CPU pool consumes (all threads)
    vram_gbs: float
    gpu_expert_ms: float       # one expert, one token, int8, warm
    gpu_expert_bytes: float    # bytes that measurement covers
    vram_cache_bytes: float
    ram_cache_bytes: float
    overlap: float = 1.0
    dense_layer_ms: float = 0.0     # non-expert time per layer per token (attention etc.)

    def nvme_gbs(self, block_bytes, qd):
        qd = max(1, qd)
        ts = sorted(self.nvme_grid)
        t = max([x for x in ts if x <= qd] or [ts[0]])
        pts = self.nvme_grid[t]
        b = block_bytes / 1e6
        if b <= pts[0][0]:
            return pts[0][1]
        for (b0, g0), (b1, g1) in zip(pts, pts[1:]):
            if b <= b1:
                w = (math.log(b) - math.log(b0)) / (math.log(b1) - math.log(b0))
                return g0 + w * (g1 - g0)
        return pts[-1][1]


def hardware_from_profile(prof, vram_cache_gb, ram_cache_gb, dense_layer_ms=0.0):
    cm = prof["cost_model"]; nv = prof["measured"]["nvme"]["points"]
    grid = collections.defaultdict(list)
    for p in nv:
        grid[p["threads"]].append((p["block_mb"], p["gbs"]))
    grid = {t: sorted(v) for t, v in grid.items()}
    g = prof["measured"]["gpu"]
    ge = g["gpu_expert_ms_by_tokens"]["1"]
    return Hardware(nvme_grid=grid, h2d_gbs=cm["h2d_pinned_gbs"], ram_gbs=cm["ram_gbs"],
                    cpu_gbs=cm["cpu_expert_gbs_all_threads"], vram_gbs=cm["vram_read_gbs"],
                    gpu_expert_ms=ge, gpu_expert_bytes=3 * 2048 * 1024,     # lma_hwprof_cuda default expert
                    vram_cache_bytes=vram_cache_gb * 1e9, ram_cache_bytes=ram_cache_gb * 1e9,
                    overlap=cm.get("copy_kernel_overlap_fraction", 1.0), dense_layer_ms=dense_layer_ms)


# ------------------------------------------------------------------ routing sources


class SyntheticRouting:
    """Routing stream generator. Per layer: with probability `stick` each expert of the
    previous token is reused, remaining slots come from a session-hot Zipf(alpha) law over
    a per-layer permutation that drifts slowly (`drift` per token), and with probability
    `xlayer` a slot is instead the fixed image of a layer-(l-1) expert (cross-layer
    correlation). hostile = uniform random, no structure."""

    PRESETS = {"favorable": dict(stick=0.55, alpha=1.3, drift=0.002, xlayer=0.6),
               "typical": dict(stick=0.30, alpha=0.9, drift=0.01, xlayer=0.35),
               "hostile": dict(stick=0.0, alpha=0.0, drift=0.0, xlayer=0.0)}

    def __init__(self, spec, preset="typical", seed=1, **override):
        self.spec = spec; self.p = dict(self.PRESETS[preset], **override); self.rng = random.Random(seed)
        E = spec.experts
        self.perm = [self.rng.sample(range(E), E) for _ in range(spec.layers)]
        self.img = [self.rng.sample(range(E), E) for _ in range(spec.layers)]
        w = [1.0 / (i + 1) ** self.p["alpha"] for i in range(E)]
        s = sum(w); self.cum = []; acc = 0
        for x in w:
            acc += x / s; self.cum.append(acc)
        self.prev = [None] * spec.layers

    def _zipf(self):
        r = self.rng.random(); lo, hi = 0, len(self.cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cum[mid] >= r:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def token(self):
        sp = self.spec; p = self.p; out = []
        for l in range(sp.layers):
            chosen = []
            prev = self.prev[l]
            if self.p["alpha"] == 0.0:
                chosen = self.rng.sample(range(sp.experts), sp.topk)
            else:
                if prev:
                    chosen = [e for e in prev if self.rng.random() < p["stick"]]
                lower = out[l - 1] if l else None
                guard = 0
                while len(chosen) < sp.topk and guard < 200:
                    guard += 1
                    if lower and self.rng.random() < p["xlayer"]:
                        e = self.img[l][self.rng.choice(lower)]
                    else:
                        e = self.perm[l][self._zipf()]
                    if e not in chosen:
                        chosen.append(e)
                while len(chosen) < sp.topk:
                    e = self.rng.randrange(sp.experts)
                    if e not in chosen:
                        chosen.append(e)
            out.append(chosen)
            if p["drift"] and self.rng.random() < p["drift"]:
                i, j = self.rng.randrange(sp.experts), self.rng.randrange(sp.experts)
                self.perm[l][i], self.perm[l][j] = self.perm[l][j], self.perm[l][i]
        self.prev = out
        return out


def trace_stream(decode_tokens, layers):
    """decode_tokens: iterable of {layer_id: [experts]} (routing.Trace.decode_tokens())."""
    for t in decode_tokens:
        yield [t[l] for l in layers]


# ------------------------------------------------------------------ caches


class LRU:
    def __init__(self, cap):
        self.cap = max(0, int(cap)); self.d = collections.OrderedDict()

    def __contains__(self, k):
        return k in self.d

    def touch(self, k):
        self.d.move_to_end(k)

    def put(self, k):
        """Insert; return the evicted key or None."""
        if self.cap == 0:
            return k
        self.d[k] = 1; self.d.move_to_end(k)
        if len(self.d) > self.cap:
            return self.d.popitem(last=False)[0]
        return None

    def discard(self, k):
        self.d.pop(k, None)


class Predictor:
    """Online cross-layer co-activation predictor: counts (layer l expert -> layer l+1
    expert) as tokens are decoded, predicts layer l+1 from layer l's actual selection.
    Costs one table lookup; needs no training data beyond the running session."""

    def __init__(self, spec, decay=0.999):
        self.co = [collections.defaultdict(collections.Counter) for _ in range(spec.layers)]
        self.freq = [collections.Counter() for _ in range(spec.layers)]
        self.decay = decay

    def observe(self, l, cur, nxt):
        for a in cur:
            self.co[l][a].update(nxt)
        self.freq[l + 1].update(nxt) if l + 1 < len(self.freq) else None

    def predict(self, l, cur, n):
        """Return [(expert, score)] for layer l+1, best first, using layer l's selection `cur`."""
        sc = collections.Counter()
        for a in cur:
            sc.update(self.co[l][a])
        return sc.most_common(n)


# ------------------------------------------------------------------ simulator


@dataclass
class Policy:
    name: str = "lru"
    prefetch: bool = False
    depth: int = 1                 # layers ahead
    delta: int = 0                 # extra candidates beyond topk
    byte_budget_frac: float = 0.4  # of the NVMe bytes the lead window can carry
    min_score_frac: float = 0.0    # candidate must carry this share of the best score
    pin_hot: float = 0.0           # fraction of VRAM tier reserved for session-hot experts
    promote_on_second_use: bool = True
    deadline: bool = False         # deadline-aware issue: skip candidates that cannot arrive in time
    warm_tokens: int = 64


def simulate_decode(spec: ModelSpec, hw: Hardware, stream, policy: Policy, tokens: int):
    """Decode `tokens` tokens from `stream` (iterator of per-token [layer][experts]).
    Returns metrics dict (all SIMULATED/PROJECTED per the caller's routing source)."""
    B = spec.expert_bytes
    vram = LRU(hw.vram_cache_bytes // B); ram = LRU(hw.ram_cache_bytes // B)
    pinned = set(); use_count = collections.Counter()
    pred = Predictor(spec) if policy.prefetch else None
    nvme_free = 0.0; t = 0.0
    inflight = {}                  # key -> ready time
    st = collections.Counter(); stall = 0.0; measured_tokens = 0; t_meas0 = None
    bw_q = lambda qd: hw.nvme_gbs(B, qd) * 1e9
    cpu_expert_s = B / (hw.cpu_gbs * 1e9)
    gpu_expert_s = (hw.gpu_expert_ms / 1e3) * (B / hw.gpu_expert_bytes)
    dense_layer_s = hw.dense_layer_ms / 1e3
    pf_issued = pf_used = pf_bytes = pf_wasted_bytes = 0
    pf_pending = {}                # key -> issue time (for wasted accounting)
    hit_v = hit_r = miss = 0
    nvme_bytes_meas = 0; layer_time_est = max(dense_layer_s + spec.topk * min(cpu_expert_s, gpu_expert_s) / 4, 1e-4)
    it = iter(stream)
    for tok in range(tokens):
        try:
            routes = next(it)
        except StopIteration:
            break
        measuring = tok >= policy.warm_tokens
        if measuring and t_meas0 is None:
            t_meas0 = t; st_start = {k: st[k] for k in ("nvme_bytes",)}
        for l, ids in enumerate(routes):
            ls = t
            # ---- issue prefetch for layers ahead, decided from what layer l actually selected
            if pred is not None and policy.prefetch and l + policy.depth < spec.layers:
                cand = pred.predict(l, ids, spec.topk + policy.delta)
                if policy.depth > 1:
                    pass                                    # deeper prediction reuses the same table one hop
                if cand:
                    top = cand[0][1]
                    window = layer_time_est * policy.depth
                    budget = window * bw_q(4) * policy.byte_budget_frac
                    spent = 0.0
                    for e, s in cand:
                        if s < policy.min_score_frac * top:
                            continue
                        key = (l + policy.depth, e)
                        if key in vram or key in ram or key in inflight:
                            continue
                        if spent + B > budget:
                            break
                        start = max(ls, nvme_free)
                        done = start + B / bw_q(4)
                        if policy.deadline and done > ls + window * 1.5 + layer_time_est:
                            continue                        # would land after its deadline: skip
                        nvme_free = done; inflight[key] = done; pf_pending[key] = ls
                        spent += B; pf_issued += 1; pf_bytes += B
                        if measuring:
                            st["nvme_bytes"] += B
            # ---- resolve this layer's experts
            ready = ls + dense_layer_s
            cpu_time = 0.0; gpu_time = 0.0; layer_miss = []
            for e in ids:
                key = (l, e)
                if key in vram:
                    vram.touch(key); gpu_time += gpu_expert_s
                    if measuring: hit_v += 1
                elif key in ram:
                    ram.touch(key); cpu_time += cpu_expert_s
                    if measuring: hit_r += 1
                    if key in pf_pending:
                        pf_used += 1; pf_pending.pop(key, None)
                elif key in inflight:
                    ready = max(ready, inflight.pop(key)); cpu_time += cpu_expert_s
                    ev = ram.put(key)
                    if key in pf_pending:
                        pf_used += 1; pf_pending.pop(key, None)
                    if measuring: hit_r += 1
                else:
                    layer_miss.append(e)
                    if measuring: miss += 1
            if layer_miss:
                n = len(layer_miss)
                qd = min(n, 8)
                start = max(ls, nvme_free)
                done = start + n * B / bw_q(qd)
                nvme_free = done; ready = max(ready, done)
                cpu_time += n * cpu_expert_s
                if measuring:
                    st["nvme_bytes"] += n * B
                for e in layer_miss:
                    ram.put((l, e))
            for e in ids:
                key = (l, e); use_count[key] += 1
                if key in ram and key not in vram and policy.promote_on_second_use and use_count[key] >= 2:
                    ev = vram.put(key)
                    if ev is not None and ev not in ram:
                        ram.put(ev)
            # expert compute: cpu pool and gpu overlap; layer ends when both are done
            end = max(ready, ls + dense_layer_s) + max(cpu_time, gpu_time)
            if measuring:
                stall += max(0.0, ready - (ls + dense_layer_s))
            t = end
            layer_time_est = 0.9 * layer_time_est + 0.1 * (t - ls)     # lead window = what layers really take
            if pred is not None and l + 1 < spec.layers:
                pred.observe(l, ids, routes[l + 1])
            # prefetched-but-never-used accounting: expire entries older than 2 tokens of layers
            if pf_pending and l % 8 == 0:
                for k in [k for k, ts in pf_pending.items() if t - ts > 4 * layer_time_est * spec.layers]:
                    pf_wasted_bytes += B; pf_pending.pop(k)
        if measuring:
            measured_tokens += 1
    wall = t - (t_meas0 or 0.0)
    n = max(1, measured_tokens)
    served = hit_v + hit_r + miss
    return {
        "policy": policy.name, "tokens": measured_tokens,
        "tok_s": measured_tokens / wall if wall > 0 else 0.0,
        "ms_per_token": wall / n * 1e3,
        "nvme_gb_per_token": st["nvme_bytes"] / n / 1e9,
        "hit_vram": hit_v / served if served else 0.0, "hit_ram": hit_r / served if served else 0.0,
        "miss": miss / served if served else 0.0,
        "gpu_stall_frac": stall / wall if wall > 0 else 0.0,
        "nvme_util": min(1.0, (st["nvme_bytes"] / (bw_q(4))) / wall) if wall > 0 else 0.0,
        "prefetch_issued": pf_issued, "prefetch_useful_frac": (pf_used / pf_issued) if pf_issued else None,
        "prefetch_wasted_gb_per_token": pf_wasted_bytes / n / 1e9,
    }
