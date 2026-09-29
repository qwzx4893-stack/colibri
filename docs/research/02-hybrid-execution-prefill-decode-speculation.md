# 02 - Hybrid CPU/GPU execution, prefill vs decode, speculation (research for LM Atomic)

Scope: what to copy into a 4060-8GB / 16GB-RAM / i7-12700H / NVMe engine. Source facts were fetched 2026-09; numbers marked (S) come from an automated page summary of the paper and should be re-checked against the PDF before being quoted. Numbers marked (calc) are my arithmetic from stated assumptions, not measurements. First real task: measure the assumed values in section 2 on the target box.

## 1. Sources

| Topic | Source (authors/venue/year) | URL | Reference-impl license | Key numbers |
|---|---|---|---|---|
| A | Fiddler, Kamahori et al., arXiv 2024 (ICLR'25) | https://arxiv.org/abs/2402.07033 | Apache-2.0 (repo efeslab/fiddler; paper CC-BY) | Rule: GPU-resident -> GPU; else CPU if `cpu_lat(s) <= gpu_lat(s)+xfer`. GPU lat ~constant in s, CPU lat ~linear in s, activation xfer <1%. Mixtral-8x7B fp16 expert >300 MB; weight xfer 2-5x longer than compute. 1.26x decode, 1.30x long prefill, 11.6x beam search. |
| A | HybriMoE, Zhong et al., DAC'25 | https://arxiv.org/abs/2504.05897 | Apache-2.0 (PKU-SEC-Lab/HybriMoE, per GitHub API) | Intra-layer scheduler = fill a timeline: GPU takes cached high-load experts first, CPU takes uncached low-load first, PCIe fetches high-load uncached; minimise max(CPU,GPU) finish; costs profiled at warmup. Score-aware cache beats LRU by 6-8% at 25% capacity. 1.33x prefill / 1.70x decode vs kTransformers (A6000, 10-core Xeon). |
| A | KTransformers, Chen et al., SOSP'25 | https://dl.acm.org/doi/10.1145/3731569.3764843 | Apache-2.0 | AMX/AVX512/AVX2 int4/int8 kernels, async CPU-GPU scheduling, expert deferral; 4.6-19.7x prefill, 1.25-4.1x decode vs prior hybrids. Hot experts on GPU, cold on CPU. |
| A | DALI, workload-aware offloading on local PCs, arXiv 2602.03495 (S) | https://arxiv.org/abs/2602.03495 | not verified | Expert-to-device assignment as 0-1 ILP with greedy fallback; separate prefill/decode policies. Closest match to our target (consumer PC); read in full. |
| A | MoE-Lightning, Cao et al., ASPLOS'25 | https://arxiv.org/abs/2411.11217 | not verified | HRM = hierarchical roofline (CPU/GPU/PCIe, turning + balance points) to pick batch/placement; CGOPipe overlaps CPU attention, I/O, GPU FFN. Up to 10.3x throughput (batch workload, not latency). |
| A | MoE-Lens, arXiv 2504.09345 / HPDC'26 | https://arxiv.org/abs/2504.09345 | not verified | Performance model predicts throughput with avg 94% accuracy; 4.6x avg over prior. |
| A | ATSInfer, Liu et al., arXiv 2607.10183 (S) | https://arxiv.org/abs/2607.10183 | not verified | Tensor-granularity placement + load-aware dynamic transfer on consumer devices. |
| B | DuoServe-MoE, arXiv 2509.07379 | https://arxiv.org/abs/2509.07379 | paper CC BY-NC-ND; code not confirmed | Prefill: activation effectively dense (all experts of a layer), two CUDA streams overlap expert fetch with non-MoE compute. Decode: 7-layer MLP predictor (popularity, inter-layer affinity, history), trained on 2.5% data. TTFT 1.78-5.34x, e2e 1.42-7.55x, peak GPU mem 15% of model (A5000/A6000, 4-bit Mixtral). |
| B | SeqMoE, arXiv 2609.12978 (S) | https://arxiv.org/abs/2609.12978 | not verified | Chunked prefill under offloading multiplies expert transfers (reported 2-3x latency vs full-batch prefill). |
| B | Layered Prefill, arXiv 2510.08055 | https://arxiv.org/abs/2510.08055 | not verified | Layer-wise (not token-chunk) prefill cuts expert-weight loads 12% (ShareGPT) to 39% (arXiv long prompts) vs chunked. |
| B | Caching/prefetch analysis, arXiv 2511.05814 (S) | https://arxiv.org/abs/2511.05814 | n/a | Decode routing more predictable -> prefetch helps; prefill irregular; LFU wins under power-law reuse; prefetch accuracy falls with layer distance. |
| B/D | SSD-LLaMA, arXiv 2609.18110 (S) | https://arxiv.org/abs/2609.18110 | paper CC BY-NC-ND (built on llama.cpp) | RTX 5090 + 16 GB RAM + 9 GB/s NVMe: 1T MoE int4 at 1.03 tok/s; O_DIRECT batched reads reach 77.7% of peak seq BW; 80% VRAM hit; experts 12.75-23.6 MiB. Does not speculate on routing. |
| B | Edge0 (prerouter), arXiv 2609.18063 (S) | https://arxiv.org/abs/2609.18063 | CC BY 4.0 | Trained per-layer prerouter predicts next layer's routing one token ahead; 35B int4 in 3 GiB active memory. Needs training - not lossless-by-construction. |
| C | SP-MoE, arXiv 2510.10302 | https://arxiv.org/abs/2510.10302 | built on PyTorch/HF; no license found | Draft attention outputs fed into target gates -> ~88% expert-prediction accuracy; cutoff-layer L so prefetch finishes within drafting time (`max{(L-1)t_comp + k_L t_io, N_exp t_io} <= L_all t_comp`). TPOT 1.07-3.5x; unique experts grow sub-linearly with draft tokens. |
| C | MoE-SpeQ, arXiv 2511.14102 | https://arxiv.org/abs/2511.14102 | paper only | INT4 copy of the model as draft predicts FP16 top-k with 90.9% accuracy (44.1% exact + 46.8% soft); cache hit >96% vs LRU 29-77%; governor `k* = argmax accepted(k)/T_cycle(k)`; up to 2.3-3.3x. A100 + PCIe4 x16. |
| C | SpecMoEOff, arXiv 2508.21706 | https://arxiv.org/abs/2508.21706 | not stated | Speculation raises tokens per expert-fetch so PCIe transfer is amortised; 2.5x over MoE-Lightning. |
| C | SpecOffload, Zhuge et al., arXiv 2505.10259 | https://arxiv.org/abs/2505.10259 | MIT (MobiSense/SpecOffload-public) | Draft resident on GPU while target weights stream; 2.54x throughput, 4.5x GPU utilisation (batch throughput setting). |
| C | Cascade / Utility-Driven SD for MoE, arXiv 2506.20675, MLSys'26 | https://arxiv.org/abs/2506.20675 | paper CC BY 4.0 | Naive SD on MoE inflates verify cost 2-3x, slowdowns up to 1.5x; utility = token gain / verify-cost ratio; disable when <1, pick K by max utility. |
| C | Limits of Speculation in MoE, arXiv 2609.22156 | https://arxiv.org/abs/2609.22156 | paper only | `T_verify = b0(m) + a(m)*U + c*L_ctx`, U = layer-avg unique experts over m=K+1 positions. Qwen3-Coder-30B: U 12.1 (K=1) -> 30.2 (K=10); per-expert cost jumps 0.08 -> 0.27 ms at K>=4; oracle speedup caps ~2.34x, oracle picks ~2.8 tokens. |
| C | EcoSpec, arXiv 2607.12696 | https://arxiv.org/abs/2607.12696 | paper CC BY-NC-SA | Draft-token score `P(t)/(dCost(t|B)+eps)` where cost counts only experts not already in buffer; expert footprint -13%; 1.15-1.62x on 120-671B models (H200). |
| C | Expert coactivation SD, arXiv 2609.22471 | https://arxiv.org/abs/2609.22471 | CC BY-NC-ND | Requires retraining the MoE: 9 -> 5 unique experts for 4 verified tokens. Not applicable to frozen weights. |
| C | EAGLE-2/3, Li et al., EMNLP'24 / NeurIPS'25 | https://arxiv.org/abs/2503.01840 | SafeAILab/EAGLE: license not machine-detected, check LICENSE | Lossless (rejection-sampling verify); EAGLE-2 3.05-4.26x on dense models; dense-model gains are an upper bound for MoE. |
| C | Medusa (Cai et al., ICML'24, arXiv 2401.10774), DeepSeek-V3 MTP (arXiv 2412.19437) - from memory | https://arxiv.org/abs/2401.10774 | Medusa Apache-2.0 | Extra heads on the target; MTP head ships with V3/GLM-class checkpoints, needs no extra training. |
| D | Fiddler latency model; HybriMoE warmup-profile timeline sim; MoE-Lightning HRM; MoE-Lens (94%); SP-MoE cutoff model; MoE-SpeQ amortisation roofline; Limits-of-Speculation linear regression above | - | - | All are additive/max-of-resources models with parameters fitted from short profiling runs. Only MoE-Lens reports a model-vs-measurement figure (94% average). No paper models NVMe queue depth; SSD-LLaMA (77.7% of peak sequential) is the only data point. |

## 2. Quantitative analysis for our box

Assumptions to be replaced by measurement: PCIe 4.0 x8 (laptop 4060) theoretical 15.75 GB/s, pinned H2D ~12.5 GB/s (calc); DDR5 dual channel effective ~45 GB/s; CPU int8 compute effective ~0.8-1.2 TOPS (6 P-cores + 8 E-cores, AVX2 + AVX-VNNI, no AVX-512, laptop power limit); 4060 VRAM 272 GB/s; NVMe 3-6 GB/s O_DIRECT (repo benchmarks show 5.9 GB/s on a PCIe4 980 PRO).

Expert size = 3*d_model*d_ff params (SwiGLU). From model configs (re-verify against config.json):

| Model | d / d_ff / experts / top-k / MoE layers | params/expert | int4 (+scales ~+12%) | int8 | H2D @12.5 GB/s int4 / int8 | Read/token top-k int4 |
|---|---|---|---|---|---|---|
| OLMoE-1B-7B | 2048/1024/64/8/16 | 6.3M | 3.5 MB | 6.3 MB | 0.28 / 0.50 ms | ~0.45 GB |
| Qwen3-30B-A3B | 2048/768/128/8/48 | 4.7M | 2.6 MB | 4.7 MB | 0.21 / 0.38 ms | ~1.0 GB |
| GLM-4.5 (355B) | 5120/1536/160/8/89 | 23.6M | 13 MB | 23.6 MB | 1.06 / 1.9 ms | ~9.4 GB |
| DeepSeek-V3 | 7168/2048/256/8/58 | 44.0M | 24.7 MB | 44 MB | 2.0 / 3.5 ms | ~11.5 GB |

The int4 experts of a 500B-class model are 12-25 MB: large enough that one NVMe read (or one H2D) is efficient; small ones (Qwen3, OLMoE) need coalescing/batched reads.

**CPU vs GPU crossover (the key result).** For an expert resident in RAM and s tokens routed to it in one layer:
- CPU time = max(bytes/BW_ram, s*2P/R_cpu). GEMV is memory-bound until s ~ R_cpu*bytes/(2P*BW_ram): ~10 (int8) to ~20 (int4, with unpack cost) tokens.
- Miss-and-fetch time = bytes/BW_h2d + GPU time (GPU time is ~0.1-0.2 ms, memory-bound, ~flat in s up to hundreds of tokens).
- Setting `s*2P/R_cpu = bytes/BW_h2d`: `s* = R_cpu * (bytes/P) / (2*BW_h2d)`. Model size cancels: int8 (1 B/param) s* ~ 0.8e12/(2*12.5e9) = **~32 tokens**; int4 (0.5 B/param) **~16 tokens** (calc; 1.2 TOPS gives 48 / 24). So: **decode (s=1) is always CPU for RAM-resident experts** (CPU 3-4x faster than a transfer: 0.5-1 ms vs 2-3.5 ms for DSV3); **prefill flips to GPU when tokens-per-expert > ~16-32**, i.e. prompt length N > s* * E/k = 16-32 * 32 (DSV3, 256/8) ~ 500-1000 tokens, or ~250-500 for Qwen3 (128/8). Below that, prefill is a mix: hot/high-load experts to GPU, low-load ones to CPU (HybriMoE rule). Consistent with this repo's own report that a 5090 expert tier gave ~0% when an AVX-512 CPU matched it.
- If the expert is only on NVMe, both paths pay the read; then the compare is CPU compute from the RAM staging buffer vs H2D + GPU compute, so the same s* applies, but the read cost is sunk and only occupancy of RAM vs VRAM staging differs.

**Prefill is stream-bound, not compute-bound.** A full pass touches (nearly) all experts: 500B int4 = ~250 GB; at 5 GB/s = ~50 s per pass (only ~16 GB RAM + ~6 GB VRAM of it cacheable), regardless of prompt length. GPU compute at 20 TFLOPS achieved balances a DSV3 layer's 5.6 GB int4 stream (1.1 s at 5 GB/s) only at N ~ 25k tokens (calc). Therefore: never chunk over tokens; one layer-wise pass over the whole prompt maximises tokens per byte streamed; chunking by C multiplies traffic by C (Layered Prefill, SeqMoE).

**Decode budget.** GLM-4.5 int4: ~9.4 GB touched/token; with ~55% hit (pin + LRU, as repo logs on 20 GB caches) ~4 GB misses -> ~0.7-0.8 s/token at 5 GB/s (~1.3 tok/s); DSV3 similar; Qwen3-30B (fits mostly in RAM+VRAM) is CPU/compute-bound at tens of tok/s. This matches SSD-LLaMA's 1.03 tok/s on a 1T model at 9 GB/s. The decode ceiling is `miss_bytes / BW_nvme`; every % of hit rate is worth more than kernel tuning.

**Speculation break-even (calc from the Limits data).** Fit U(m) ~ 8*m^0.55 unique experts/layer for m=K+1 verified tokens (top-8 of 128; matches 12.1 at m=2, ~30 at m=11). In the I/O-bound regime verify cost ratio = U(m)/8 = m^0.55; tokens per cycle with per-token acceptance a = (1-a^(K+1))/(1-a). Speculation helps iff `(1-a^(K+1))/(1-a) > (K+1)^0.55 * (1+d)`, d = draft overhead fraction (5-10%).

| K | break-even a (d=0) | gain at a=0.7 | gain at a=0.8 | gain at a=0.5 |
|---|---|---|---|---|
| 1 | 0.46 | 1.16x | 1.23x | 1.03x |
| 3 | 0.59 | 1.18x | 1.38x | 0.88x (loss) |
| 5 | 0.65 | 1.10x | 1.38x | loss |
| 7 | 0.70 | 1.0x | 1.32x | loss |

Cache hits on already-resident experts lower marginal cost (EcoSpec), so real optimum K is a bit larger; but the takeaway is K = 2-3, gain ~1.2-1.4x, and a real chance of loss when acceptance is under 0.6. Lossless SD on MoE is worth far less than the 3-4x seen on dense models.

## 3. Decisions for LM Atomic

### 3.1 Placement cost model (per expert e, layer l, s_e tokens)

```
T_cpu(e)  = tier_read(e) + max( B_e / BW_ram , s_e * F_e / R_cpu ) / cpu_free     # F_e = 2P_e
T_gpu(e)  = tier_read(e) + [ B_e / BW_h2d  if not in VRAM ] + max( B_e / BW_vram , s_e * F_e / R_gpu )
tier_read = 0 if in RAM/VRAM; else B_e / BW_nvme_eff(queue depth)   (shared, queued)
choose device = argmin over feasible; ties -> device whose queue drains earlier
```
Layer step = solve minimax over all missed experts: GPU takes VRAM-resident (high s_e first), CPU takes low-s_e misses, PCIe fetches high-s_e misses (HybriMoE timeline fill; DALI ILP is the exact version; greedy is enough at <=256 experts/layer). Inputs: s_e (from the router - known exactly once the router runs), B_e (tier format), measured BW_* (EWMA, not datasheet), `cpu_free` (thread pool availability, penalise when the CPU is also doing attention/draft), PCIe queue length. Decision hysteresis: only move an expert to VRAM if reuse probability * saved time exceeds eviction cost. Lossless caveat: CPU and GPU kernels accumulate in different orders; for the Exact mode pin the kernel per expert per run (repo issue #1136 shows run-to-run drift when the kernel follows cache state) and A/B with fixed kernels.

### 3.2 Prefill Planner
- Objective: tokens per streamed byte; TTFT floor = bytes_touched / min(BW_nvme, BW_h2d) minus what's resident.
- One layer-wise pass over the whole prompt; token chunking only when activations/KV force it, and then chunk size >= s* * E/k so each chunk stays above the crossover.
- Do not predict experts for prefill: all experts are hit; instead schedule by layer order. Read layer l+1 (and l+2) into RAM while layer l computes; 3-buffer pipeline (NVMe -> pinned RAM -> VRAM) on two CUDA streams (DuoServe).
- Per layer: experts with s_e > s* -> GPU (transfer once, compute batched); s_e < s* -> CPU from staging buffer in parallel; attention/dense on GPU. Reuse the residency left by the previous decode as a free head start; end prefill by leaving the last layers' hot experts resident.
- Record per-expert routing counts during prefill (needed by the decode pin/cache policy - "prefill warms decode": SeqMoE, ADEPT idea).

### 3.3 Decode Planner
- Objective: minimise misses; s_e is 1-4, so **CPU computes any RAM-resident miss**; GPU handles VRAM-resident experts, attention and the router; PCIe/NVMe only *promote* experts.
- Cache: pinned hot set (from prefill counts + history) + score-aware LRU/LFU (HybriMoE, LFU under power-law) split VRAM/RAM; NVMe reads issued as batched O_DIRECT at queue depth >4 (SSD-LLaMA: 77.7% of peak with batching vs 43% without).
- Prefetch: only lossless-by-construction predictors (they can only waste bandwidth, never change output): cross-layer gate reuse (apply layer l+1's gate to layer l's hidden state), history/affinity tables; cap prefetch depth by the SP-MoE-style cutoff so it finishes inside the compute window and cannot evict needed experts. Do not ship trained prerouters (Edge0) in Exact mode.

### 3.4 Lossless speculative design (token + expert co-speculation)
1. Draft with the checkpoint's own MTP head or an EAGLE-3 head (self-speculation; resident on GPU, tens of MB) - rejection-sampling verify keeps the target distribution exact. Skip separate draft models and INT4 shadow models (VRAM/RAM budget; MoE-SpeQ needs the INT4 copy at 8-25% of model size).
2. Expert pre-move: while drafting token i, feed each draft hidden state into the target's gates for layers >= L_cut (SP-MoE, ~88% accuracy). Build the union set U(K+1) for the verify pass, dedupe against cache, and enqueue NVMe reads sorted by layer order and by miss cost; verify pass then runs layer-by-layer with the exact target router - a wrong guess costs only bandwidth. Order: experts common to many draft tokens first.
3. Window control (Cascade + MoE-SpeQ governor): maintain EWMA of per-token acceptance a-hat and the measured marginal cost dU per extra draft token (new-expert misses * t_miss). Choose K = argmax `E[tokens](K,a-hat) / (T_draft + T_base + U_miss(K+1)*t_miss)`; set K=0 when utility < 1; probe K+1 periodically (test/set phases). Expect K in {0,2,3}; K=0 for low-acceptance text (a < ~0.55), K=3-5 for code/templated output at a >= 0.8.
4. Draft selection: prefer draft branches whose experts are already resident (EcoSpec score) - only if it does not change the accepted distribution: it must only change which candidate tokens are *proposed*, never the acceptance rule.
5. Verify kernel: batched verification numerics differ slightly from 1-token decode; Exact mode must state whether greedy output is bit-identical to K=0 and test it.

### 3.5 Skip (for now)
- Trained predictors/prerouters, low-rank expert surrogates, expert skipping/substitution (HOBBIT mixed precision, dynamic top-k reduction): lossy; keep only as measured optional modes.
- Coactivation-trained MoEs (needs retraining) and beam-search/batch-throughput designs (MoE-Lightning, SpecOffload, MoE-Gen): wrong objective (single-user latency).
- Token-chunked prefill; separate INT4 shadow drafters; NDP/CXL/multi-GPU papers.
- AMX-specific kernels (kTransformers): our CPU has AVX2+VNNI only; copy its scheduler ideas, not its kernels.

### 3.6 Digital twin
Twin = per-resource service model (NVMe with queue-depth curve, PCIe H2D with pinned/pageable, RAM BW, CPU R by thread count, VRAM BW) + the timeline-fill scheduler above driven by recorded routing traces (cache replay simulator; the repo already has docs\experiments\cnre-offline-simulator.md). Parameters from a 60 s calibration run; validation target <10% error on tok/s and TTFT (MoE-Lens claims 94% accuracy; no paper validates NVMe-bound decode, so treat that as an open item). Use it to pick cache split, K policy and pin sets offline.

### Open verification items
Read DALI, MoE-Lightning, SeqMoE and 2511.05814 in full (summaries used); confirm EAGLE repo license; measure H2D, RAM BW, VNNI GEMV GFLOPS, NVMe queue-depth curve on the target laptop before fixing s*.
