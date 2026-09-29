"""Decisive measurement, before building anything: does a shared low-rank
basis actually capture a meaningful fraction of the variance across OLMoE's
real experts within a layer? If singular values decay fast (a few directions
dominate), Shared Expert Basis Compilation is worth building. If they're
flat (experts are ~orthogonal / independent), it is not, and no amount of
compiler engineering will make it help -- measure first, per instructions.
"""
import json, struct, numpy as np, time

import glob
import os, sys
# usage: python expert_basis_analysis.py <converted OLMoE container dir>
SHARDS = sorted(glob.glob(os.path.join(sys.argv[1], "model-*.safetensors")))
HIDDEN, INTER, NEXP = 2048, 1024, 64


def find_shard(layer):
    key = f"model.layers.{layer}.mlp.experts.0.merged_weight"
    for p in SHARDS:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        if key in header:
            return p, header
    raise KeyError(f"layer {layer} not found in any shard")


def load_expert(f, header, base_off, layer, eid):
    wname = f"model.layers.{layer}.mlp.experts.{eid}.merged_weight"
    sname = f"model.layers.{layer}.mlp.experts.{eid}.qs"
    woff = header[wname]["data_offsets"]
    soff = header[sname]["data_offsets"]
    f.seek(base_off + woff[0])
    raw = np.frombuffer(f.read(woff[1] - woff[0]), dtype=np.int8)
    f.seek(base_off + soff[0])
    scale = np.frombuffer(f.read(soff[1] - soff[0]), dtype="<f4")
    # layout: gate[inter,hidden] int8, up[inter,hidden] int8, down[hidden,inter] int8
    # scales: gate_scale[inter], up_scale[inter], down_scale[hidden] (per-row)
    ng = INTER * HIDDEN
    gate = raw[:ng].reshape(INTER, HIDDEN).astype(np.float32)
    up = raw[ng:2*ng].reshape(INTER, HIDDEN).astype(np.float32)
    down = raw[2*ng:].reshape(HIDDEN, INTER).astype(np.float32)
    gs, us, ds = scale[:INTER], scale[INTER:2*INTER], scale[2*INTER:2*INTER+HIDDEN]
    gate = gate * gs[:, None]
    up = up * us[:, None]
    down = down * ds[:, None]
    return gate, up, down


def analyze_layer(layer, n_experts=NEXP):
    p, header = find_shard(layer)
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        base_off = 8 + n
        downs = []
        t0 = time.time()
        for e in range(n_experts):
            _, _, down = load_expert(f, header, base_off, layer, e)
            downs.append(down.reshape(-1))
        print(f"  loaded+dequantized {n_experts} experts in {time.time()-t0:.1f}s")
    M = np.stack(downs, axis=0)  # [n_experts, hidden*inter]
    del downs
    norms = np.linalg.norm(M, axis=1)
    print(f"  per-expert down-matrix Frobenius norm: mean={norms.mean():.2f} std={norms.std():.2f} "
          f"(std/mean={norms.std()/norms.mean():.3f} -- low means experts are similar scale, not similarity)")
    # mean-center is wrong for a "shared basis + residual" test if the basis
    # must be a genuine subspace (not an affine shift) -- test both.
    Mc = M - M.mean(axis=0, keepdims=True)
    for label, X in (("raw (basis = linear subspace incl. a possible mean direction)", M),
                     ("mean-centered (basis = pure shared subspace)", Mc)):
        t0 = time.time()
        # economy SVD on the [n_experts, dim] matrix -- rank <= n_experts, fast
        s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
        total = (s ** 2).sum()
        cum = np.cumsum(s ** 2) / total
        print(f"  [{label}] SVD in {time.time()-t0:.1f}s. Variance captured by top-r singular directions:")
        for r in (1, 2, 4, 8, 16, 32):
            if r <= len(cum):
                print(f"    r={r:3d}: {cum[r-1]*100:.2f}%")
    # pairwise cosine similarity as a second, more intuitive signal
    Mn = M / norms[:, None]
    cos = Mn @ Mn.T
    iu = np.triu_indices(n_experts, k=1)
    print(f"  pairwise cosine similarity (off-diagonal): mean={cos[iu].mean():.4f} "
          f"max={cos[iu].max():.4f} min={cos[iu].min():.4f}")


if __name__ == "__main__":
    for layer in (0, 8, 15):
        print(f"=== layer {layer} (down_proj, {NEXP} experts, {HIDDEN}x{INTER} each) ===")
        analyze_layer(layer)
        print()
