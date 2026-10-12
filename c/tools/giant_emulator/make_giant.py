"""Physical Giant Emulator: builds a real, disk-backed synthetic OLMoE-shaped
model directory for a given giant preset, so the REAL unmodified olmoe.exe
runtime (scheduler, predictor, async pipeline, prefill planner, cache) can be
driven under real NVMe I/O pressure at 100B-500B scale without downloading or
storing a real giant checkpoint.

Design, so this is a physical test and not a disguised sleep():
  - Expert weight/scale tensors for ALL (layer, expert) pairs are declared in
    the safetensors header, but their data_offsets alias a SMALL pool of
    REAL, fully-written (non-sparse) data blocks on disk, chosen by
    (layer*n_experts+expert) % pool_blocks. Every read is a genuine pread of
    real bytes at a real offset; nothing is faked with delays.
  - The engine must be run with EXPERT_DROP=1 so a re-visited pool block is
    NOT served from the OS page cache the second time a different (layer,
    expert) id happens to alias it -- otherwise repeated aliasing would look
    faster than a real giant model's genuinely-distinct experts.
  - Dense weights (embed/lm_head/norm/attention per layer) are small, real,
    unique tensors at a synthetic internal hidden=512 (NOT meant to reflect a
    real giant model's attention width -- only the expert subsystem is under
    test here; dense/attention cost is explicitly out of scope and is small
    by construction so it cannot dominate the measurement).
  - vocab=64, so a tiny ref.json of raw token ids drives generation with no
    tokenizer needed at all (CLI ref.json path, not the HTTP serve path).
"""
import json, os, struct, sys

HIDDEN = 512
VOCAB = 64
N_HEADS = 8

GIANTS = {
    "100B_A9B": dict(total_b=100, dense_b=2, layers=48, experts=128, topk=8),
    "300B_A14B": dict(total_b=300, dense_b=3, layers=60, experts=256, topk=8),
    "500B_A8B": dict(total_b=500, dense_b=2, layers=60, experts=640, topk=8),
    "500B_A16B": dict(total_b=500, dense_b=4, layers=60, experts=320, topk=8),
    "500B_A31B": dict(total_b=500, dense_b=6, layers=90, experts=160, topk=8),
    "30B_A3B": dict(total_b=30, dense_b=1.3, layers=48, experts=128, topk=8),
    "60B_A5B": dict(total_b=60, dense_b=1.5, layers=48, experts=128, topk=8),
}


def preset_dims(name, gpu_shape=False):
    g = GIANTS[name]
    routed = (g["total_b"] - g["dense_b"]) * 1e9
    pe = routed / (g["layers"] * g["experts"])
    active_b = (g["dense_b"] * 1e9 + g["layers"] * g["topk"] * pe) / 1e9
    inter = round(pe / (3 * HIDDEN))
    if gpu_shape:                    # the CUDA kernels need inter % 32 == 0
        inter = max(32, (inter + 16) // 32 * 32)
    w_bytes = 3 * inter * HIDDEN
    s_elems = 2 * inter + HIDDEN
    return dict(layers=g["layers"], experts=g["experts"], topk=g["topk"], inter=inter,
                w_bytes=w_bytes, s_elems=s_elems, active_b=active_b, total_b=g["total_b"])


def build(preset_name, out_dir, pool_gb=2.0, seed=0, gpu_shape=False):
    import random
    global N_HEADS
    if gpu_shape:
        N_HEADS = 4                  # head_dim 128, what the CUDA path implements
    d = preset_dims(preset_name, gpu_shape)
    os.makedirs(out_dir, exist_ok=True)
    expert_bytes_total = d["w_bytes"] + d["s_elems"] * 4
    pool_blocks = max(4, int(pool_gb * 1e9 / expert_bytes_total))
    n_layers, n_experts = d["layers"], d["experts"]
    total_expert_tensors = n_layers * n_experts
    print(f"[{preset_name}] inter={d['inter']} expert_MB={expert_bytes_total/1e6:.2f} "
          f"pool_blocks={pool_blocks} (of {total_expert_tensors} distinct ids), "
          f"pool_GB={pool_blocks*expert_bytes_total/1e9:.2f} active~{d['active_b']:.1f}B")

    rng = random.Random(seed)
    header = {}
    data_chunks = []
    offset = 0

    def add_tensor(name, dtype, shape, nbytes, chunk):
        nonlocal offset
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        data_chunks.append(chunk)
        offset += nbytes

    # --- dense weights: real, unique, small ---
    def randbytes(n):
        return rng.randbytes(n)

    def randfloats(n):
        return struct.pack(f"<{n}f", *[rng.uniform(-1, 1) for _ in range(n)])

    def randhalves(n):   # FP16 dense tensors (what the checkpoint format stores and the GPU path keeps resident)
        return struct.pack(f"<{n}e", *[rng.uniform(-1, 1) * 0.05 for _ in range(n)])

    add_tensor("model.embed_tokens.weight", "F32", [VOCAB, HIDDEN], VOCAB*HIDDEN*4, randfloats(VOCAB*HIDDEN))
    if gpu_shape:
        add_tensor("lm_head.weight", "F16", [VOCAB, HIDDEN], VOCAB*HIDDEN*2, randhalves(VOCAB*HIDDEN))
    else:
        add_tensor("lm_head.weight", "F32", [VOCAB, HIDDEN], VOCAB*HIDDEN*4, randfloats(VOCAB*HIDDEN))
    add_tensor("model.norm.weight", "F32", [HIDDEN], HIDDEN*4, randfloats(HIDDEN))
    for i in range(n_layers):
        add_tensor(f"model.layers.{i}.input_layernorm.weight", "F32", [HIDDEN], HIDDEN*4, randfloats(HIDDEN))
        add_tensor(f"model.layers.{i}.post_attention_layernorm.weight", "F32", [HIDDEN], HIDDEN*4, randfloats(HIDDEN))
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if gpu_shape:
                add_tensor(f"model.layers.{i}.self_attn.{proj}.weight", "F16", [HIDDEN, HIDDEN],
                           HIDDEN*HIDDEN*2, randhalves(HIDDEN*HIDDEN))
            else:
                add_tensor(f"model.layers.{i}.self_attn.{proj}.weight", "F32", [HIDDEN, HIDDEN],
                           HIDDEN*HIDDEN*4, randfloats(HIDDEN*HIDDEN))
        add_tensor(f"model.layers.{i}.self_attn.q_norm.weight", "F32", [HIDDEN], HIDDEN*4, randfloats(HIDDEN))
        add_tensor(f"model.layers.{i}.self_attn.k_norm.weight", "F32", [HIDDEN], HIDDEN*4, randfloats(HIDDEN))
        add_tensor(f"model.layers.{i}.mlp.gate.weight", "F32", [n_experts, HIDDEN],
                   n_experts*HIDDEN*4, randfloats(n_experts*HIDDEN))

    # --- real pool of distinct expert blocks ---
    pool_w_offset = []
    pool_s_offset = []
    for b in range(pool_blocks):
        w_off = offset
        data_chunks.append(randbytes(d["w_bytes"]))
        offset += d["w_bytes"]
        pool_w_offset.append(w_off)
        s_off = offset
        data_chunks.append(randfloats(d["s_elems"]))
        offset += d["s_elems"] * 4
        pool_s_offset.append(s_off)

    # --- every (layer, expert) id aliases a real pool block ---
    for l in range(n_layers):
        for e in range(n_experts):
            block = (l * n_experts + e) % pool_blocks
            w_off = pool_w_offset[block]
            header[f"model.layers.{l}.mlp.experts.{e}.merged_weight"] = {
                "dtype": "I8", "shape": [d["w_bytes"]], "data_offsets": [w_off, w_off + d["w_bytes"]]}
            s_off = pool_s_offset[block]
            header[f"model.layers.{l}.mlp.experts.{e}.qs"] = {
                "dtype": "F32", "shape": [d["s_elems"]], "data_offsets": [s_off, s_off + d["s_elems"]*4]}

    header_json = json.dumps(header).encode("utf-8")
    shard_path = os.path.join(out_dir, "model-00000.safetensors")
    with open(shard_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for chunk in data_chunks:
            f.write(chunk)
    shard_size = os.path.getsize(shard_path)
    print(f"[{preset_name}] shard written: {shard_path} ({shard_size/1e9:.3f} GB, "
          f"{total_expert_tensors} expert ids over {pool_blocks} real blocks)")

    head_dim = HIDDEN // N_HEADS
    cfg = {
        "hidden_size": HIDDEN, "num_hidden_layers": n_layers, "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_HEADS, "num_experts": n_experts, "num_experts_per_tok": d["topk"],
        "intermediate_size": d["inter"], "vocab_size": VOCAB, "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5, "norm_topk_prob": False,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f)
    (_ := head_dim)

    return dict(preset=preset_name, dir=out_dir, shard_size_gb=shard_size/1e9,
                expert_mb=expert_bytes_total/1e6, active_b=d["active_b"], pool_blocks=pool_blocks,
                total_expert_ids=total_expert_tensors, n_layers=n_layers, n_experts=n_experts, vocab=VOCAB)


def build_overlay(preset_name, out_dir, pool_gb=2.0, seed=1, gpu_shape=True):
    """int4 overlay for the emulated model: `.w4` (U8) / `.s4` (F16) tensors for every (layer, expert), aliasing a pool of
    REAL written blocks exactly like the int8 shard does, so every read is a genuine pread of int4-sized data."""
    import random
    d = preset_dims(preset_name, gpu_shape)
    os.makedirs(out_dir, exist_ok=True)
    ng = d["inter"] * HIDDEN
    w4b, s4n = 3 * ng // 2, 3 * ng // 32
    blk = w4b + 2 * s4n
    pool_blocks = max(4, int(pool_gb * 1e9 / blk))
    rng = random.Random(seed)
    header, chunks, offset = {}, [], 0
    pool = []
    import numpy as np
    nrng = np.random.default_rng(seed)
    for b in range(pool_blocks):
        pool.append(offset)
        # random nibbles, but FINITE FP16 scales (random bit patterns would be NaN/Inf and trip the runtime's breaker)
        chunks.append(rng.randbytes(w4b) + nrng.uniform(0.0005, 0.02, s4n).astype(np.float16).tobytes()); offset += blk
    for l in range(d["layers"]):
        for e in range(d["experts"]):
            o = pool[(l * d["experts"] + e) % pool_blocks]
            header[f"model.layers.{l}.mlp.experts.{e}.w4"] = {"dtype": "U8", "shape": [w4b], "data_offsets": [o, o + w4b]}
            header[f"model.layers.{l}.mlp.experts.{e}.s4"] = {"dtype": "F16", "shape": [s4n], "data_offsets": [o + w4b, o + blk]}
    hj = json.dumps(header).encode("utf-8")
    path = os.path.join(out_dir, "overlay-00000.safetensors")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj))); f.write(hj)
        for c in chunks: f.write(c)
    print(f"[{preset_name}] int4 overlay: {blk/1e6:.2f} MB/expert, {pool_blocks} real blocks, {os.path.getsize(path)/1e9:.2f} GB on disk")
    return dict(dir=out_dir, expert_mb_int4=blk / 1e6)


if __name__ == "__main__":
    preset = sys.argv[1]
    out_dir = sys.argv[2]
    pool_gb = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    gpu = "--gpu" in sys.argv
    info = build(preset, out_dir, pool_gb=pool_gb, gpu_shape=gpu)
    if "--int4" in sys.argv:
        info["overlay"] = build_overlay(preset, out_dir + "_int4", pool_gb=pool_gb)
    print(json.dumps(info, indent=2))
