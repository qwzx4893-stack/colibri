# GPU runtime, Quality Governor and int4 storage: measured results (2026-09-29)

Machine: Windows 11, RTX 4060 Laptop 8 GB (VRAM ~175 GB/s measured d2d traffic ceiling, ~8.5 GB/s pinned H2D),
16.8 GB RAM (~19 GB/s multi-thread memcpy traffic), one NVMe. Model: OLMoE-1B-7B-Instruct (real weights,
`C:\Users\STS\models\olmoe-colibri`). All A/B numbers are interleaved in one session, 3 reps unless stated;
machine noise between runs can reach 2x, so cross-run comparisons are not used.

Class labels: **MEASURED REAL MODEL** = real weights through the real engine. **PHYSICALLY EMULATED** = random weights,
real disk-backed tiering/I-O/kernels (used only for I/O-volume and tiering questions, never for weight-structure
properties). **PROJECTED** = arithmetic from measured ceilings, labelled where used.

## 1. Exact Mode on the GPU (MEASURED REAL MODEL)

| test | CPU | GPU | note |
|---|---|---|---|
| 140-token prompt, 128 new tokens, end-to-end tok/s (median of 3) | 3.63 | 44.70 (default) / 47.00 (all experts in VRAM) | text identical in all three |
| 512 new tokens, end-to-end | 4.42 | 62.62 | text identical |
| time to first token, 140-token prompt | 14.96 s | 0.99 s | GPU batched prefill |
| logits vs CPU reference (verify mode) | | max abs diff 5e-5 (decode), 1e-5 (prefill); argmax agreement 47/47 | |

Decode is bandwidth-bound on the GPU too: [LMA_DIAG] reports ~100 GB/s achieved against a 174 GB/s measured ceiling
(58%), per-token weight traffic 1.56 GB. Exact Mode exceeds the 30 tok/s daily-use target, so it stays Exact.

## 2. Soak (MEASURED REAL MODEL)

208 back-to-back requests (~20k tokens, 12 rotating prompts, greedy): 0 text mismatches; engine RSS 1355.3 -> 1355.4 MB;
handles 389 -> 387; threads 15 -> 12; VRAM used 7710 -> 7710 MB (+-40 MB noise). The wall-clock of the run was 150 min
because the laptop went to sleep during the last request (one 8577 s request), which also makes the tok/s slope in
`soak30.summary.json` meaningless; the resource trends are unaffected. This is a short soak, not the multi-hour one.

## 3. Speculative decoding (MEASURED REAL MODEL): rejected as a default

Prompt-lookup n-gram drafts + batched exact verification (tokens are identical to plain decoding in every run).
Median tok/s, plain vs speculative (k=6): chat 71.9 vs 54.9, code 62.6 vs 53.5, json 68.2 vs 63.2 (all worse), summarise-with-context
52.7 vs 58.0 (n-gram 3), copy 62.5 vs 69.0 (n-gram 3). A verification of 7 rows costs ~60 ms against 12.5 ms for a plain
step: OLMoE routes 8 of 64 experts, so seven tokens touch almost every expert and the verify pass reads ~4.8x the bytes of one
decode step. It only pays when acceptance is ~6 tokens per verify (pure copy). Default: off (`LMA_SPEC=0`); the engine's own
breaker switches it off mid-session when its ms/token exceeds plain decoding.

## 4. Quality Governor and High-Fidelity candidates (MEASURED REAL MODEL)

Reference = Exact. Teacher-forced probe over 1790 tokens of 18 chat turns; suite = 21 tasks in 7 domains (Exact solves 16/21).
Default budget: top-1 agreement >= 0.97, mean KL <= 0.02 nats, ppl ratio <= 1.03, task retention >= 0.95.

| profile | tok/s (suite median) | mean KL | top-1 | ppl ratio | tasks | verdict |
|---|---|---|---|---|---|---|
| Exact | 62-68 | 0 | 1.000 | 1.000 | 16/21 | reference |
| top-6 of 8 experts | 71.0 | 0.088 | 0.921 | 1.040 | 16/21 | rejected |
| top-4 of 8 experts | 91.3 | 0.354 | 0.834 | 1.303 | 16/21 | rejected |
| skip experts < 0.35 x best | 75.0 | 0.091 | 0.923 | 1.051 | 14/21 | rejected |
| int4 (group-32) all layers, plain scale | 58.6 | 0.091 | 0.924 | 1.113 | 15/21 | rejected |
| int4 all layers, MSE clip search | 58.7 | 0.079 | 0.919 | 1.068 | 15/21 | rejected |
| int4 on the 4 least-sensitive layers | | 0.0139 | 0.972 | 1.009 | | **accepted** |
| int4 on 8 least-sensitive layers | | 0.0244 | 0.954 | 1.023 | | rejected (KL) |
| int4 on 12 least-sensitive layers | | 0.0354 | 0.948 | 1.026 | | rejected |

Per-layer int4 sensitivity (KL of Exact when only that layer is int4): most sensitive layers 14 (0.0177) and 13 (0.0097), least
sensitive 15 (0.0013); the others 0.003-0.008. On OLMoE int4 does not increase speed (dequantisation makes the kernels slower
than int8 while the model already fits VRAM): its value is capacity, see section 6.

## 5. Atomic Compiler (int4 overlay, MEASURED REAL MODEL)

`lmatomic/compiler.py` + engine `QCOMPILE`: streaming (one expert in memory), shard-by-shard, atomic renames, sha256 manifest,
resume. Real OLMoE: 16 layers -> 4 shards, 3.62 GB (int8 experts: 6.5 GB, ratio 0.557), 41 s. Re-run = 4 shards verified and
skipped; deleting one shard rebuilds only that shard. The overlay route and the on-the-fly quantiser give bit-identical logits
(KL 0.0 over 1790 positions). The source model directory is never written.

## 6. Giant-model tiering on the GPU path (PHYSICALLY EMULATED, random weights)

`c/tools/giant_emulator/make_giant.py 30B_A3B <dir> 0.3 --gpu --int4` builds a disk-backed model with the real 30B-A3B
geometry (48 layers, 128 experts, top-8, 4.7 MB int8 / 2.6 MB int4 per expert; 6144 experts) whose expert tensors alias a
small pool of real blocks. Weights are random, so the routing is near-uniform: this is the **worst case for locality**
and says nothing about weight-structure-dependent properties. It does measure real tiering, real I/O volume and real
kernels. 64 teacher-forced tokens (prefill of 8 included), 3 reps, VRAM holds what fits next to the dense weights.

| config | resident in VRAM | misses/token | GB moved/token | median tok/s (min-max) |
|---|---|---|---|---|
| int8 experts, cold reads | 1515/6144 | 323 | 1.61 | 1.23 (1.13-1.23) |
| int8 experts, page-cache warm | 1515/6144 | 323 | 1.61 | 1.70 (1.69-1.72) |
| int4 overlay, cold reads | 2709/6144 | 211 | 0.66 | 4.39 (4.19-4.41) |
| int4 overlay, page-cache warm | 2709/6144 | 211 | 0.66 | 4.42 (4.40-4.43) |
| int4 overlay + async staging ring | 2709/6144 | 211 | 0.66 | 4.13 (4.12-4.20): no gain |

int4 storage is a 3.6x speed-up for this model class (2.4x fewer bytes per miss and 1.8x more experts resident). The first
int4 attempt tripped the circuit breaker (non-finite logits): the emulator had written random bit patterns as the FP16
scales; the breaker fell back to the CPU and the run finished, which is the behaviour it is there for.

Where the time goes (int4, miss path timers over 63 decode tokens, 14.1 s): reading the experts 9.5 s (151 ms/token,
3.7 GB/s), waiting on the router sync, which includes the host-to-device copies, 4.6 s (73 ms/token). An asynchronous
staging ring did not help because each layer needs its own experts before it can run; only predicting the next layer's
experts (accuracy depends on real routing, so it cannot be validated on this emulator) could overlap them.
Microbenchmark on the same machine (page-cache-hot shard, 2.6 MB blocks): pageable mapped view to GPU 5.7 GB/s, page-locked
view 8.2 GB/s (the PCIe ceiling), per-chunk register + copy + unregister 3.6 GB/s. Uploading straight from a mapped view
measured 6.05 tok/s in one run on this model, but one earlier run of that configuration hung, so it is experimental.

Physical bound: with 211 misses/token at 0.66 GB/token and ~8.5 GB/s PCIe, 77 ms/token (13 tok/s) is the ceiling at this
hit rate; 30 tok/s needs the miss traffic under ~0.25 GB/token, i.e. a hit rate that only real routing locality can give.
For a model whose experts do not fit RAM+VRAM the NVMe (2-4 GB/s here) bounds it far below that.
