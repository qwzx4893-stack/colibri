# Local baseline: i7-12700H / RTX 4060 Laptop, 2026-09-28

A fixed reference for this machine, taken after the router/CUDA fixes on branch
`foundation-hardening`. Compare later changes against these numbers with the same
commands. Follows [benchmarking.md](../../benchmarking.md): cache state, I/O and
hardware are stated beside every number.

## Host

| | |
|---|---|
| CPU | 12th Gen Intel Core i7-12700H, 14 physical cores / 20 threads (engine sizes OpenMP to 14) |
| RAM | 16.8 GB total, 5.5-7.2 GB available during runs |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB (sm_89), driver 616.92, CUDA 13.4 (portable toolkit, nvcc 13.4.59 + VS Build Tools 2022 17.14) |
| Storage | WD PC SN740 512 GB NVMe, model and cache on the OS volume (buffered I/O, `DIRECT=0`) |
| OS / build | Windows 11 26200, MinGW-w64 GCC 16.1 (`-O3 -march=x86-64-v3`), static, default `make` flags |
| Commit | branch `foundation-hardening` (built from the tree of this commit) |

## A. Real model: OLMoE-1B-7B-0924-Instruct (the one real MoE this build can run)

The model already on the machine (LM Studio: `Fabliq-8B-Agent-FromBase-Reasoning`, GGUF
Q5_K_M, architecture `lfm2moe`) has no Colibri engine, so it could not be used. OLMoE
is the smallest MoE family Colibri supports, and `tools/convert_gguf_to_olmoe.py` is its
supported GGUF path.

| | |
|---|---|
| Source | `allenai/OLMoE-1B-7B-0924-Instruct-GGUF`, `olmoe-1b-7b-0924-instruct-q4_k_m.gguf`, 4.21 GB, SHA-256 `8C310F14...C32F3F` |
| Container | `convert_gguf_to_olmoe.py` (dense F16, routed experts re-quantized to int8 row-wise), 7.4 GB in 5 shards, 1m55s |
| Architecture | 16 layers, 64 routed experts, top-8, ~1.3B active of ~6.9B |
| Engine | `olmoe.exe`, CPU only (no GPU tier exists for this family) |
| Serving | `coli serve` (HTTP gateway + one engine process), `KV_SLOTS=1`, greedy, `max_tokens=64`, concurrency 1 |
| Expert cache | auto: 32 of 64 slots/layer (6.6 GB budget), 6 MB per expert, no pin set |

Workload: `workload.jsonl` (three chat prompts of 26 / 188 / 790 prompt tokens).
Command: `python c/tools/benchmark_http_serving.py --base-url http://127.0.0.1:8123/v1
--model olmoe --workload workload.jsonl --concurrency 1 --repeats N --max-tokens 64 --temperature 0`.
"Cold" is the first pass after starting the server (expert LRU empty, OS page cache in
whatever state the machine left it); "warm" is two further passes. Raw reports:
`cold_run.json`, `warm_run.json`; summary: `olmoe_serving_summary.json`.

| prompt | TTFT cold / warm (s) | decode tok/s cold / warm | wall (s) |
|---|---|---|---|
| 26 tokens | 5.9 / 4.6 | 4.3 / 4.4 | 20.8 / 19.3 |
| 188 tokens | 22.1 / 23.3 | 4.3 / 4.4 | 37.1 / 38.1 |
| 790 tokens | 90.9 / 93.6 | 3.9 / 4.0 | 107.2 / 109.6 |

Prefill therefore runs at about 6-8 prompt tokens/s, no faster than decode.

Per-turn engine profile (gateway `/profile`, 64 generated tokens, warm):

| prompt tokens | expert disk (s) | expert matmul (s) | attention (s) | lm_head (s) |
|---|---|---|---|---|
| 26 | 8.6 | 5.5 | 3.9 | 1.3 |
| 188 | 16.5 | 14.5 | 4.7 | 1.3 |
| 790 | 47.8 | 50.3 | 8.6 | 1.3 |

Resources during one 64-token request (`resource_samples.json`, 10 samples over 20 s):
engine RSS 5.0 GB, disk read 529 MB/s average / 875 MB/s peak, total CPU 28% average /
38% peak (20 logical CPUs), GPU unused. The run is bound by expert reads plus int8 expert
matmul, not by CPU saturation.

`coli doctor` reports 4% projected expert residency at its own RAM budget and recommends
`coli tune`; that step was not applied, so these are default-configuration numbers.

## B. GPU: synthetic GLM-5.2-shaped fixture (CPU vs CUDA)

There is no local model for the GLM family, so the GPU tier is measured on the repository's
own fixture: `tools/make_glm_bench_model.py` (313M parameters, 8 layers, 32 experts,
top-8, random weights; real MLA/MoE/streaming shapes). Generated with the pinned
torch 2.13.0+cpu / transformers 5.16.1. Not a language model: it measures placement and
speed, not quality.

`python c/tools/benchmark_cuda_fixture.py --model <fixture> --engine ./colibri.exe --gpu 0
--runs 3 --threads 14 --pin-gb 1 --cuda-expert-gb 2` on the CUDA_DLL host (`make
colibri.exe CUDA_DLL=1 && make cuda-dll`), replay of 8 fixed tokens, warm page cache.
Raw: `glm_bench_cpu_vs_cuda.json`.

| mode | median tok/s | routed experts run on |
|---|---|---|
| cpu_stream (LRU cache 4/layer) | 40.9 | CPU, streamed |
| dense_cuda | 48.3 | CPU (dense tensors on GPU) |
| cpu_pin (1 GB hot set) | 78.0 | CPU, pinned |
| cuda_pin (2 GB VRAM experts) | 124.1 | GPU (`routed GPU critical` 11 ms, routed CPU 0) |
| cuda_pin_dense | 167.7 | GPU, dense on GPU |

GPU footprint in the fastest mode: 553 resident tensors, 0.15 GB (engine accounting);
`nvidia-smi` showed 814 MB peak used vs 581 MB idle desktop, utilization 11% mean / 48% peak
over 60 short runs (each run is ~50 ms of decode, so utilization is not a meaningful
figure for this fixture).

Correctness of the GPU paths is checked separately against transformers, not with this
fixture (random weights make argmax ties): the tiny GLM oracle (`make_glm_oracle.py`) is
32/32 teacher-forced and 20/20 greedy on CPU, with dense tensors on the GPU, with routed
experts pinned in VRAM (`routed CPU 0 rows`), and with `COLI_CUDA_PIPE=2`.
`make cuda-test` (nvcc + MSVC, real GPU) passes: q8/q4/q2/f32/e8 matmul, ragged attention
(rms 0), fp8 warp kernels (rel. RMS 3.5e-7, 0 mismatches), absorb determinism, upload
rollback, fmt trap.

## Reproduce

```sh
# CPU engines and unit tests
make -C c check          # or: make -C c test-c   (about 165 gates on Windows since the test discovery fix)
# OLMoE baseline
python c/tools/convert_gguf_to_olmoe.py --input olmoe-1b-7b-0924-instruct-q4_k_m.gguf --output olmoe-colibri
python c/coli serve --model olmoe-colibri --port 8123 --model-id olmoe
python c/tools/benchmark_http_serving.py ...   # as above
# GPU (from an x64 Native Tools prompt with CUDA on PATH)
make -C c colibri.exe CUDA_DLL=1 && make -C c cuda-dll && make -C c cuda-test
python c/tools/make_glm_bench_model.py --output glm_bench_medium --device cpu
python c/tools/benchmark_cuda_fixture.py --model glm_bench_medium --engine ./colibri.exe --gpu 0
```

## Limits of this baseline

* One machine, one model per family, three requests per cell; not a ranking of engines.
* The OS page cache was not flushed between "cold" and "warm" (no administrator rights to
  do it), so "cold" only means an empty expert LRU. Cold and warm are equal because the
  7.4 GB model largely sits in the OS cache.
* No real GLM/Kimi/Qwen/DeepSeek checkpoint fits this machine, so GPU behaviour is
  measured on the synthetic fixture only.
* Quality was not measured (no accuracy benchmark was run on the converted OLMoE).
