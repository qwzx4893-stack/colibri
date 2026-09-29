# LM Atomic: real hardware/model profile + calibrated Digital Twin ceiling (2026-09-29)

Reference machine: RTX 4060 Laptop (8 GB VRAM), i7-12700H, 16.8 GB RAM (8.1 GB free
at measurement time), one NVMe SSD, Windows. Same machine as
`docs/baselines/local-2026-09-28/`.

All three JSON files here are raw tool output, unedited.

## What was measured (real, not simulated)

- `hardware_profile.json` — `lmatomic/hwprof.py` orchestrating `lma_hwprof.exe`
  (CPU/RAM), `lma_hwprof_cuda.exe` (GPU), and `iobench.exe` (NVMe) on this
  machine. Key numbers: RAM 20.9 GB/s, CPU expert-weight throughput 22.8 GB/s
  (all threads), H2D pinned 12.7 GB/s, VRAM read 207 GB/s, NVMe peak **4.81 GB/s**
  (at 3.5 MB blocks, queue depth 8; single-thread queue-depth-1 peak was
  3.47 GB/s at 64 MB blocks) — this laptop's single NVMe is the weakest link
  in the whole pipeline by roughly an order of magnitude.
- `agent_profile.json` — `lmatomic/agent.py` run end-to-end (`--model
  C:\Users\STS\models\olmoe-colibri --max-tokens 48`) against the real,
  already-converted OLMoE model: architecture graph read from safetensors
  headers only, then real engine calibration (4 domains x 2 prompts,
  374 decode tokens total) with `ROUTE_TRACE` capture. Measured routing
  locality: `reuse_next_token=0.396`, `entropy_norm=0.949` (routing is close
  to uniform — OLMoE is known for weak locality), LRU hits 80% of forwards
  at a 32-expert cache (half the 64-expert layer) vs. 90.6% for the
  Belady-optimal offline bound at the same capacity — real, measured headroom
  for a smarter-than-LRU cache, but not a large one for this model.

## What was simulated (labeled, not a benchmark)

- `giants_result.json` — the calibrated Digital Twin (`lmatomic/twin.py`)
  driven by the **measured** hardware profile above, decoding 200 synthetic
  tokens (32 warm-up) per cell for all 5 `GIANTS` presets x 3
  `SyntheticRouting` presets (favorable/typical/hostile) x 2 policies
  (`lru-only` baseline, `predict+prefetch+pin` — depth-1 cross-layer
  prediction, byte-budgeted deadline-aware prefetch, 25% VRAM pinned to
  session-hot experts).
- Assumptions, stated so they can be challenged: 6.5 GB VRAM cache (of 7.45 GB
  measured free — headroom left for KV cache/activations), 8.0 GB RAM cache
  (the free RAM measured at profiling time), **`dense_layer_ms=0.0`** —
  attention/dense compute is not counted at all, because no real engine run
  was used to calibrate it for a 100-500B hypothetical model. Every `tok_s`
  number in `giants_result.json` is therefore an **expert-layer-only ceiling**,
  optimistic relative to a real end-to-end run by construction.

## The ceiling this proves

Even under that optimistic assumption, and even in the single *friendliest*
routing preset (`favorable`) with the smallest, most GPU/RAM-cacheable
active-parameter count tested:

| giant | active params | twin tok/s (expert-only, favorable) | NVMe GB/s needed for 30 tok/s | vs. measured 4.81 GB/s |
|---|---|---|---|---|
| 100B / A~9B  | 8.1B  | 4.19 | 24.5 GB/s  | **5.1x** |
| 500B / A~8B  | 8.2B  | 3.56 | 29.2 GB/s  | **6.1x** |
| 300B / A~14B | 12.3B | 2.19 | 49.0 GB/s  | 10.2x |
| 500B / A~16B | 16.4B | 1.41 | 74.9 GB/s  | 15.6x |
| 500B / A~31B | 30.7B | 0.28 | 416.8 GB/s | 86.7x |

(50 tok/s needs 1.67x more bandwidth again in every row.)

The spec asked, if 30-50 tok/s turns out to be physically impossible on this
hardware, to prove the ceiling with numbers and name the next
bottleneck-breaking technique rather than assume. This is that proof: on
**this specific laptop's single NVMe**, even the friendliest 500B-class
scenario needs 6-9x more sustained disk bandwidth than the drive delivers,
*before* attention/dense compute, GPU-wait, or real (non-synthetic) routing
are even counted. None of the subsystems planned in this phase — smarter
caching, cross-layer prediction, deadline-aware prefetch — change this
number in the favorable-routing rows above (prefetch_useful_frac there is
already high; the miss traffic isn't a scheduling problem, it's a
bytes-that-must-move-per-token problem). Closing a 5-10x gap this large
needs one of:

1. **Multi-NVMe striping** — the project's own community benchmarks
   (`docs/benchmarks.md` #1249) show two independent NVMe controllers on the
   same machine roughly doubling throughput; still short by 3-5x alone.
2. **Batch amortization** — serve several concurrent sessions so one disk
   read of a hot expert serves many tokens at once (continuous batching
   already exists in `c/openai_server.py`'s `KV_SLOTS`, but nothing here
   schedules *expert* reads across a batch yet); this is the only lever with
   enough headroom in principle, and it is unbuilt.
3. Accept a lower active-parameter target for this hardware class — the
   500B/A~8-9B row is the best case measured, and even it needs the NVMe
   bandwidth item above.

This does not mean 30-50 tok/s is unreachable in general — it means it is
unreachable on **this exact laptop's single NVMe** with the techniques
implemented so far, and the batch-amortization lever is the one worth
building next, not another caching or prediction refinement.
