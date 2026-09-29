"""Routing traces and the Routing Locality Fingerprint.

A trace is the engine's ROUTE_TRACE stream: one line per (moe call, token row, layer)

    <call> <row> <layer> <expert>:<gate> <expert>:<gate> ...

`call` advances once per moe() invocation, so a forward pass over all layers spans
`n_moe_layers` consecutive calls. Rows > 1 in a call are prefill positions.

Everything here is computed from measured traces. `fingerprint()` returns the numbers
LM Atomic uses to choose a caching strategy per model and per session:

  reuse_next_token       P(expert used at token t is used again at t+1), same layer
  chance_next_token      the same probability if routing were uniform (K/E)
  top_mass_experts       experts (per layer, averaged) needed to cover 95% of traffic
  entropy_norm           normalised Shannon entropy of the per-layer selection distribution
  working_set_w16/64/256 distinct experts per layer inside a sliding window of W tokens
  hot_stability          Jaccard of the top-K hot sets of the first and second half
  layer_transition_recall  recall of a next-layer prediction built from co-activation
                         counts learned on the first half, scored on the second half
  lru_hit / belady_hit   hit rate of per-layer LRU vs the offline optimum at several
                         capacities (fractions of E)
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Trace:
    n_layers: int = 0                 # distinct MoE layers seen
    layers: list = field(default_factory=list)      # sorted layer ids
    n_experts: int = 0
    topk: int = 0
    # forwards[f] = {"rows": R, "layers": {layer: [ [experts of row0], [row1], ... ]}}
    forwards: list = field(default_factory=list)

    def decode_tokens(self):
        """Per decode forward: {layer: experts}. Prefill (rows > 1) is excluded."""
        for fw in self.forwards:
            if fw["rows"] == 1:
                yield {l: fw["layers"][l][0] for l in fw["layers"]}

    def prefill_forwards(self):
        for fw in self.forwards:
            if fw["rows"] > 1:
                yield fw


def parse(path, n_experts=None):
    calls = collections.OrderedDict()             # call -> {"layer": L, "rows": {row: [experts]}}
    kmax = 0
    emax = -1
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            call, row, layer = int(p[0]), int(p[1]), int(p[2])
            ids = [int(x.split(":")[0]) for x in p[3:]]
            kmax = max(kmax, len(ids)); emax = max(emax, max(ids))
            c = calls.setdefault(call, {"layer": layer, "rows": {}})
            c["rows"][row] = ids
    tr = Trace(topk=kmax, n_experts=n_experts or (emax + 1))
    # A forward starts when the layer index does not increase.
    cur = None; last_layer = -1
    layers = set()
    for call, c in calls.items():
        layer = c["layer"]; layers.add(layer)
        if cur is None or layer <= last_layer:
            cur = {"rows": len(c["rows"]), "layers": {}}
            tr.forwards.append(cur)
        cur["layers"][layer] = [c["rows"][r] for r in sorted(c["rows"])]
        cur["rows"] = max(cur["rows"], len(c["rows"]))
        last_layer = layer
    tr.layers = sorted(layers); tr.n_layers = len(tr.layers)
    return tr


def _entropy_norm(counts):
    tot = sum(counts)
    if tot == 0:
        return 0.0
    h = -sum((c / tot) * math.log(c / tot) for c in counts if c)
    return h / math.log(len(counts)) if len(counts) > 1 else 0.0


def _mass_experts(counts, mass=0.95):
    tot = sum(counts)
    if tot == 0:
        return 0
    acc = 0
    for i, c in enumerate(sorted(counts, reverse=True), 1):
        acc += c
        if acc >= mass * tot:
            return i
    return len(counts)


def lru_hit(seq, cap):
    """seq: list of expert-id lists per token for ONE layer. Returns hit fraction."""
    cache = collections.OrderedDict(); hit = tot = 0
    for ids in seq:
        for e in ids:
            tot += 1
            if e in cache:
                hit += 1; cache.move_to_end(e)
            else:
                cache[e] = 1
                if len(cache) > cap:
                    cache.popitem(last=False)
    return hit / tot if tot else 0.0


def belady_hit(seq, cap):
    flat = [(t, e) for t, ids in enumerate(seq) for e in ids]
    nxt = {}; nexts = [0] * len(flat)
    for i in range(len(flat) - 1, -1, -1):
        e = flat[i][1]
        nexts[i] = nxt.get(e, 1 << 60); nxt[e] = i
    cache = {}; hit = 0                       # expert -> next use index
    for i, (_, e) in enumerate(flat):
        if e in cache:
            hit += 1; cache[e] = nexts[i]
        else:
            if len(cache) >= cap:
                victim = max(cache, key=cache.get)
                del cache[victim]
            cache[e] = nexts[i]
    return hit / len(flat) if flat else 0.0


def fingerprint(tr: Trace, caps=(0.0625, 0.125, 0.25, 0.5)):
    toks = list(tr.decode_tokens())
    if len(toks) < 8:
        return {"error": "fewer than 8 decode tokens in the trace", "decode_tokens": len(toks)}
    E, K = tr.n_experts, tr.topk
    per_layer = {l: [t[l] for t in toks if l in t] for l in tr.layers}
    out = {"decode_tokens": len(toks), "n_layers": tr.n_layers, "n_experts": E, "topk": K}
    reuse = []; ent = []; mass = []; ws = {16: [], 64: [], 256: []}; stab = []
    for l, seq in per_layer.items():
        counts = [0] * E
        for ids in seq:
            for e in ids:
                counts[e] += 1
        ent.append(_entropy_norm(counts)); mass.append(_mass_experts(counts))
        r = n = 0
        for a, b in zip(seq, seq[1:]):
            sa = set(a)
            r += sum(1 for e in b if e in sa); n += len(b)
        reuse.append(r / n if n else 0.0)
        for W in ws:
            if len(seq) >= W:
                vals = [len({e for ids in seq[i:i + W] for e in ids}) for i in range(0, len(seq) - W + 1, max(1, W // 4))]
                ws[W].append(sum(vals) / len(vals))
        h = len(seq) // 2
        def hot(part):
            c = collections.Counter(e for ids in part for e in ids)
            return {e for e, _ in c.most_common(K)}
        a, b = hot(seq[:h]), hot(seq[h:])
        stab.append(len(a & b) / len(a | b) if a | b else 0.0)
    avg = lambda x: sum(x) / len(x) if x else None
    out.update({
        "reuse_next_token": round(avg(reuse), 4), "chance_next_token": round(K / E, 4),
        "entropy_norm": round(avg(ent), 4), "top_mass_experts_95": round(avg(mass), 2),
        "working_set_w16": round(avg(ws[16]), 2) if ws[16] else None,
        "working_set_w64": round(avg(ws[64]), 2) if ws[64] else None,
        "working_set_w256": round(avg(ws[256]), 2) if ws[256] else None,
        "hot_stability": round(avg(stab), 4),
    })
    # cross-layer transition predictability: counts learned on the first half
    ls = tr.layers; h = len(toks) // 2; recs = []
    for a, b in zip(ls, ls[1:]):
        co = collections.defaultdict(collections.Counter)
        for t in toks[:h]:
            if a in t and b in t:
                for ea in t[a]:
                    co[ea].update(t[b])
        hit = tot = 0
        for t in toks[h:]:
            if a in t and b in t:
                score = collections.Counter()
                for ea in t[a]:
                    score.update(co[ea])
                pred = {e for e, _ in score.most_common(K)}
                hit += len(pred & set(t[b])); tot += len(t[b])
        if tot:
            recs.append(hit / tot)
    out["layer_transition_recall"] = round(avg(recs), 4) if recs else None
    curves = {}
    for f in caps:
        cap = max(1, int(round(E * f)))
        lr = [lru_hit(seq, cap) for seq in per_layer.values()]
        bl = [belady_hit(seq, cap) for seq in per_layer.values()]
        curves[str(f)] = {"cap": cap, "lru_hit": round(avg(lr), 4), "belady_hit": round(avg(bl), 4)}
    out["cache_curves"] = curves
    out["locality_class"] = classify(out)
    return out


def classify(fp):
    """High locality -> aggressive caching/pinning; low -> stream, keep caches small."""
    lift = fp["reuse_next_token"] / max(fp["chance_next_token"], 1e-9)
    c = fp["cache_curves"]
    mid = c.get("0.25") or next(iter(c.values()))
    if lift >= 3.0 and mid["lru_hit"] >= 0.6:
        return "high"
    if lift >= 1.5 or mid["lru_hit"] >= 0.4:
        return "typical"
    return "low"


def jaccard_hot(tr_a: Trace, tr_b: Trace):
    """Cross-domain overlap of hot expert sets (top-K by traffic per layer)."""
    def hot(tr):
        per = {}
        for t in tr.decode_tokens():
            for l, ids in t.items():
                per.setdefault(l, collections.Counter()).update(ids)
        return {l: {e for e, _ in c.most_common(tr.topk)} for l, c in per.items()}
    a, b = hot(tr_a), hot(tr_b)
    vals = [len(a[l] & b[l]) / len(a[l] | b[l]) for l in a if l in b and (a[l] | b[l])]
    return sum(vals) / len(vals) if vals else None
