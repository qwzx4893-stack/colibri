"""How much would verifying k tokens at once amortize expert loads?

Reads ROUTE_TRACE files ("<call> <row> <layer> <expert>:<gate> ...") from real decode runs and, for windows
of k consecutive decode tokens, measures the number of DISTINCT experts a k-token verification pass would have
to load per layer, versus the k*topk loads k separate single-token passes would need.

    ratio = mean(|union of k expert sets|) / (k * topk)        1.0 = no amortization, 1/k = perfect overlap

Speculative decoding only reduces expert traffic per accepted token when this ratio is well below 1 (and the
draft is accepted often enough to use the extra verified tokens). Usage: spec_overlap.py trace1.txt [trace2 ...]
"""
import collections, sys


def load(path):
    """-> list of decode forwards, each {layer: set(experts)}. The trace's call counter advances once per MoE layer
    call (not per token), so a forward pass is the run of calls from layer 0 up to the next layer 0. Prefill
    forwards (calls with more than one row) are dropped; only single-row decode forwards are returned."""
    calls = []                     # [(layer, nrows, experts)]
    cur = None
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            call, layer = int(p[0]), int(p[2])
            ex = set(int(x.split(":")[0]) for x in p[3:])
            if cur is None or cur[0] != call:
                cur = [call, layer, 0, set()]
                calls.append(cur)
            cur[2] += 1; cur[3] |= ex
    fwd, cur_f, ok = [], {}, True
    for _, layer, nrows, ex in calls:
        if layer == 0 and cur_f:
            if ok:
                fwd.append(cur_f)
            cur_f, ok = {}, True
        cur_f[layer] = ex
        if nrows != 1:
            ok = False
    if cur_f and ok:
        fwd.append(cur_f)
    return fwd


def main(paths):
    for k in (2, 3, 4, 6, 8):
        num = den = 0
        for p in paths:
            toks = load(p)
            for i in range(0, len(toks) - k + 1):
                for layer in toks[i]:
                    u = set()
                    topk = len(toks[i][layer])
                    for j in range(k):
                        u |= toks[i + j].get(layer, set())
                    num += len(u); den += k * topk
        print(f"k={k}: distinct experts loaded / (k*topk) = {num/den:.3f}   "
              f"(per-token expert loads x{num/den*k:.2f} of a single-token pass, i.e. saves {(1-num/den)*100:.1f}% "
              f"of expert traffic per token if all k tokens are accepted)")


if __name__ == "__main__":
    main(sys.argv[1:])
