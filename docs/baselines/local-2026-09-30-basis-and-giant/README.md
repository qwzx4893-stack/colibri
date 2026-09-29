# Shared-basis measurement, compute-bound finding, and statistically valid giant-emulator numbers (2026-09-30)

Same machine as the earlier baselines: RTX 4060 Laptop 8 GB, i7-12700H, 16.8 GB RAM, one NVMe, Windows.

## 1. Shared Expert Basis: REJECTED AFTER MEASUREMENT (real OLMoE weights)

`c/tools/expert_basis_analysis.py <olmoe container dir>` dequantizes the real int8 `down_proj`
of all 64 experts in a layer and asks whether a low-rank basis shared across experts captures
their variance (the premise of MoBE / MoE-SVD-style compression).

| layer | pairwise cosine (mean / max) | variance in top-8 of 64 directions | top-16 | top-32 |
|---|---|---|---|---|
| 0  | -0.0000 / 0.0026 | 13.8% | 26.6% | 51.7% |
| 8  | -0.0000 / 0.0029 | 13.4% | 26.4% | 51.9% |
| 15 |  0.0000 / 0.0029 | 15.4% | 28.8% | 54.1% |

Independent random vectors give exactly r/64 (12.5% / 25% / 50%). The measured curve sits on that
null line in every layer: the experts are statistically indistinguishable from orthogonal vectors,
so there is no shared subspace for a basis to store once. `Expert = basis + delta` therefore
degenerates to `0 + full expert`, and the delta / basis-residency ideas built on it have nothing to
save on this model. Caveat, stated plainly: this is a result about OLMoE-1B-7B (small, trained for
expert diversity). Whether a larger, more redundant MoE behaves differently cannot be tested here
without its real weights; the synthetic giant emulator uses random experts, so testing redundancy
on it would be tautological.

## 2. OLMoE is compute-bound once cached, not bandwidth-bound

With `--cap 64` (every expert resident, ~0.05 GB/token read, prediction precision 100%) the engine
sustains only ~3.7-3.9 tok/s. Removing disk I/O entirely moves the number from ~1.5-2 (cache
pressure) to ~3.8 and no further: the remaining cost is CPU matmul. Storage-reduction techniques
(shared basis, deltas, sub-byte storage for bandwidth) attack a bottleneck this model does not have
when it fits in RAM; only faster compute (kernels, GPU) can move it toward 30 tok/s. That is the
regime the Automatic 30 tok/s policy now measures and reports (`agent.measure_steady_state_tok_s`:
3 sessions, warm-up discarded, median; result 3.81 tok/s, min 3.67, max 4.10 -> `needs_high_fidelity`).

## 3. Confirmed bug: FUSED3=1 is not bit-exact

`fused_simd.h` and olmoe.c document `FUSED3=1` as bit-identical to the stock `matmul_q` path. Measured:
baseline is bit-reproducible run to run (two launches, identical text), but `FUSED3=1` diverges at
character 257 of a 96-token greedy generation. The speed gain was only ~4% (3.67 -> 3.81 tok/s). It is
off by default, so no verified result in this repository used it; treat the documentation claim as false
until the kernel is root-caused.

## 4. Giant emulator, statistically valid (100B/A~9B, cap=10, 3 reps, EXPERT_DROP=1)

The earlier single-sample rows suggested tok/s fell as locality rose. Repeated runs show that was noise:

| locality (`LMA_GIANT_STICK`) | median tok/s | min-max | stdev | hit rate |
|---|---|---|---|---|
| hostile 0.00  | 0.40 | 0.39-0.42 | 0.015 | 12.4% |
| typical 0.30  | 0.41 | 0.41-0.42 | 0.006 | 16.9% |
| favorable 0.55| 0.41 | 0.41-0.42 | 0.006 | 19.9% |

Hit rate responds monotonically to injected locality; throughput is flat because at 10 slots for 128
experts per layer even the favorable regime leaves ~80% of expert reads on disk. The cache, not the
scheduler, is the limit at this RAM size. Raw data: `giant_100B_A9B_cap10_3reps.json`.

## 5. Exact-Mode win: FP16-resident dense weights (`LMA_DENSE_F16`, default on)

Correction to section 2: FUSED3 gaining only ~4% showed the cached engine is *memory-bandwidth*-bound,
not ALU-bound. The checkpoint stores attention and lm_head as FP16 (743 MB) but the engine expanded them
to FP32 in RAM, so every decode step streamed ~1.5 GB of dense weights instead of ~0.75 GB. FP16->FP32 is
exact, so `matmul_h` keeps them FP16-resident and converts in registers, reproducing the exact reduction
structure GCC emits for `matmul` (one 8-lane accumulator, separate mul/add, lanes summed sequentially
from 0.0f — read from the generated assembly, not assumed).

Measured (256 tokens greedy, all experts cached, 3 interleaved reps each):

| config | median tok/s | min-max |
|---|---|---|
| FP32-resident (`LMA_DENSE_F16=0`) | 5.21 | 5.17-5.30 |
| FP16-resident (default) | 6.80 | 6.69-6.86 |

**+30.6%, token-exact across all six runs.** 1486 Python tests and every native C gate pass.
Remaining per-token traffic is ~0.75 GB dense + ~0.8 GB int8 experts, i.e. a ~13 tok/s ceiling at this
machine's ~20 GB/s RAM bandwidth. Reaching 30 tok/s in Exact Mode on this model therefore needs the
GPU (207 GB/s VRAM, and the whole 7.2 GB model nearly fits in 7.45 GB free VRAM), which olmoe.c does
not have today; that is the next real lever, not weight compression.
