# Engine architecture: what is shared, what is per-model, where the seams are

LM Atomic is one C engine per model family over shared single headers. This page
records the boundaries as they stand after the shared-router work, so a change
lands in the right place and a new MoE family reuses what already exists.

The rule from the README still holds: an engine owns its architecture and nothing
else. The defects that keep recurring here are mechanisms that landed in one
engine and never reached its siblings, so anything two engines need lives in a
header they both include.

## Layers

```
c/<family>.c        architecture: layer graph, attention variant, score function,
                    weight normalisation, tensor names, chat-template facts
  |
  |  shared, engine-agnostic headers
  v
moe_router.h        top-K expert selection + non-finite policy      (NEW: one copy)
route_trace.h       routing telemetry, .lma_usage history, EMAP/HITS feed
expert_store.h      lease/eviction contract for streamed experts (pluggable backend)
expert_ffn.h        routed-expert FFN kernel (planar int4) + layer runner
pin_pool.h kv_prefix.h kv_persist.h kv_fp8.h kv_tq.h     state reuse between turns
quant.h fp8_format.h idot.h gsgemv.h qgemv.h             container decoders + CPU kernels
st.h                safetensors index, positioned reads, mirror replicas
tok.h json.h        tokenizer, JSON
sample.h grammar.h  sampling, grammar-forced drafts
serve_codec.h serve_poll.h serve_budget.h decode_batch.h   engine <-> gateway wire protocol
compat.h omp_tune.h platform shims, thread sizing
  |
  |  optional tiers (compiled in, chosen at runtime)
  v
backend_cuda.* backend_metal.* backend_vulkan.*   GPU residency for dense + expert tensors
```

Outside `c/`: `openai_server.py` (HTTP gateway, one engine subprocess, framed over
stdin/stdout), `lmatomic` (launcher, planner), `web/`, `desktop/`.

## MoE routing: one implementation

`moe_router.h` is the single place that ranks experts.

| function | use |
|---|---|
| `lma_router_topk(score, bias, keep, E, K, idx, val, layer, degrade)` | pick K distinct experts by `score+bias`, lowest id wins ties, optional keep mask |
| `lma_router_best(...)` | one pick, for incremental callers (prefetch predictions that stop at a mass threshold) |
| `lma_router_fill(idx, val, K, E, layer)` | give every unfilled slot an in-range, distinct expert |

Two policies, chosen by the caller and never hidden inside the ranking:

* `degrade = 1` - every slot is a valid expert id even when the scores are NaN
  (the slot falls back to the lowest expert not yet taken, one warning). For
  engines whose downstream code indexes tables with the result: GLM-5.2, GLM-5.3,
  Kimi K3, Inkling, OLMoE, Qwen3.6/3.8, DeepSeek V4.1.
* `degrade = 0` - unfilled slots are `-1`. DeepSeek V4 uses it and treats a short
  result as an error (a poisoned router must not silently pick a wrong expert);
  prefetch predictions use it because `-1` there means "no prediction".

What stays in the engine: the score function (sigmoid, softmax, sqrt-softplus),
the top-p tail cut, the weight normalisation and route scale, and what a "shared"
expert means. Those differ between architectures for real, so they are not
abstracted. `tests/test_moe_router.c` pins the ranking against the per-engine loops it
replaced and the NaN behaviour; `tests/test_router_nan.c` pins the GLM-5.2 wiring.

## Adding a MoE family

1. New `c/<family>.c` with the architecture. Include `moe_router.h`,
   `route_trace.h`, `st.h`, `quant.h`, `expert_ffn.h`.
2. Compute router scores, then `lma_router_topk(...)`; derive weights from the
   returned ids. Call `rt_route(layer, token, idx, weights, K)` so telemetry and the
   dashboard work without extra code.
3. Register a `family_registry.py` entry, add the engine rule to `c/Makefile`
   (`tests/test_makefile_deps.py` checks its header prerequisites), and add a tiny
   oracle fixture so CI compares tokens against the reference.

## Seams that exist today for tiering, prefetch and scheduling

These are the boundaries later work should build on rather than route around. None of
them is new; the refactor only made them easier to see.

* **Expert residency**: `LmaExpertStoreOps` (`expert_store.h`) is a lease contract
  (`lookup`/`release`/`prefetch`) with a name-registered backend
  (`expert_store_registry.c`, `LMA_EXPERT_STORE=<name>`). DeepSeek V4 implements it;
  the single-file engines keep their slot arrays behind the same idea.
* **Placement policy**: `tier.h` (LFRU promotion, hysteresis) and `hybrid_split.h`
  (closed-form GPU-upload vs host-compute split from measured bandwidths) are pure
  functions with no I/O, tested without a GPU.
* **Routing information**: `route_trace.h` owns per-expert counters and the
  `.lma_usage` history; `telemetry.h` turns them into EMAP/HITS frames. A
  prefetcher or scheduler reads these rather than the engine's private arrays.
* **Backends**: GPU code sits behind `backend_*.h` function tables; on Windows the
  CUDA/HIP backend is a runtime-loaded DLL (`backend_loader.c`), so the host toolchain
  and the GPU toolchain never mix.
* **Prefetch**: routing is predicted one layer ahead (`la_predict` in `glm52.c`,
  `lma_router_best` for the mass-threshold variants). The predictor and the loader
  are separate functions.
* **Serving**: the engine speaks the framed protocol in `docs/serve_protocol.md`; the
  gateway owns admission, KV-slot assignment and HTTP. Multi-request scheduling
  changes belong on the gateway/`decode_batch.h` side of that line.

## `glm52.c` module boundaries

`glm52.c` is a single translation unit by design (tests include it directly), so
its parts are headers included in place. They are not standalone; they use the
engine's types and statics, and moving code into them changes nothing in the
compiled result.

| header | responsibility |
|---|---|
| `glm52_cluster.h` | expert-worker protocol, coordinator client, worker loop (POSIX) |
| `glm52_uring_load.h` | Linux io_uring expert-load batches |
| `ssd_probe.h` | storage-bandwidth probe (Metal builds and tests) |
| `glm52_adapters.h` | Segment/Edge ABI adapters for external consumers |

## Not unified, on purpose

* **Expert-load bookkeeping and the batch union** (build the distinct expert list of
  a prefill block, load each once): every engine does this against a different slot
  layout (slab arenas, per-tensor heaps, mmap views, io_uring batches, GPU tiers).
  Forcing one implementation would change I/O patterns and numerics for no defect
  it fixes today. The seams above are where a shared version should start.
* **Tensor names**: each family reads its Hugging Face checkpoint names directly
  (`model.layers.N.mlp.experts.E...`). That mapping is architecture; a lookup layer
  would add indirection without removing a copy.
* **DeepSeek V4's unit build** (`Makefile.deepseek-v4`) already splits one source
  into `-DLMA_V4_UNIT_*` translation units, so it was left as is.
