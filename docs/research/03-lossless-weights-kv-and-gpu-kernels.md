# 03 - Lossless weights, KV cache, GPU kernels (LM Atomic research)

Status: web research was cut short by a session limit. Tags: [V] = read on the source page/abstract this session; [M] = from memory, verify before relying (mostly licenses); [E] = my estimate, must be benchmarked on the 4060 box. Items not covered (unfinished): SSD-LLaMA details (arXiv 2609.18110), Zstd/FSE CPU speeds, GDeflate ratio on weights, CPU Huffman decode GB/s on the i7-12700H, Machete/fused-MoE-Triton specifics.

## 1. Sources

### A. Lossless weight codecs
| Source | Venue/yr, URL | License | Key numbers |
|---|---|---|---|
| DFloat11 (Zhang et al.) | NeurIPS'25 / arXiv 2504.11651, https://arxiv.org/abs/2504.11651 ; code https://github.com/LeanModels/DFloat11 | code [not confirmed]; paper CC BY | BF16 -> 67.6-67.9% (10.82-10.87 bit) on Llama-3.1 8B/70B/405B [V]. Exponent entropy ~2.6 bit, ~40 of 256 exponents used. Huffman + 256-entry hierarchical LUTs in SMEM, 2-phase kernel (count, then write with gap array). Up to 20.97x faster than nvCOMP ANS; nvCOMP ratio 79% vs 68%. Batch-1 decode ~40% slower than native BF16 on A100 (decompress per block). BF16 only. Tested incl. RTX 4090. |
| ZipNN (Hershcovitch et al., IBM) | arXiv 2411.05239, https://github.com/zipnn/zipnn | [M] MIT, verify | Byte-group split + Huffman on exponent, 256 KB chunks. BF16 ~66% (1.51x vs zstd 1.27x), FP32 ~83%. CPU only: comp 13 GB/s, decomp up to 80 GB/s with 16 workers [V] (server CPU). Supports fp8 e4m3/e5m2. |
| ECF8 (Yang et al.) | ICLR'26, arXiv 2510.02676, https://github.com/zeyuyang8/ecf8 | not stated on page [V] | Lossless FP8: exponent entropy 2-3 bit. DeepSeek-R1-0528 623.2 -> 530.3 GB (-14.8%), Qwen3-235B-A22B-FP8 -14.4%, Llama-3.3-70B-FP8 -13.4%, Qwen3-8B-FP8 -9.8%. Theory floor ~FP4.67. Decoder is Huffman-style GPU kernel (H100). |
| Lossless compression of low-precision weights/KV (Heilper, Singer; Intel) | arXiv 2508.19263 | n/a | FP8 -> 55-70% claimed (exponent ratio 0.20-0.30, mantissa >0.80) - conflicts with ECF8's 85-90% on real checkpoints; trust ECF8. FP4 (MXFP4/NVFP4) 4-bit values NOT compressible (uniform); FP8 scale factors give ~5% total for NVFP4. KV exponent ratios: FP8 0.25-0.45, BF16 <0.20 [V]. |
| Approaching Shannon Bound with Lossless LLM Weight Compression (Tan, Chen, Alonso, Wong, He) | ISCA'26, arXiv 2606.15789 | no code linked [V] | tANS/ANS at GEMM-tile granularity (A100 32x128 tiles). Claims bf16 10-12 bit, int8 4-5 bit, "int4 0.6-1.0 bit" entropy (surprising; likely model/format specific, possibly sparse SmoothQuant-style - DO NOT assume for GPTQ/AWQ/Q4_K, measure), within 0.01-0.1 bit of Shannon. 6-7x over DFloat11. |
| ZipServ (ASPLOS'26) | arXiv 2603.17435 | paper CC BY; code https://github.com/scitix/ZipServ-vLLM [license not checked] | TCA-TBE fixed-length triple-bitmap format, fused decompress+GEMM into tensor-core regs; ~71% size; RTX4090/L40S(Ada)/5090; 2.21x over cuBLAS on small-N decode, 1.10x DFloat11 standalone, 2.14x DietGPU. Ratio is worse than Huffman but SIMT-friendly. |
| NeuZip | NeurIPS'24, arXiv 2410.20650 | [M] verify | ANS on exponent only; lossless mode bit-identical. |
| DietGPU (Meta) | https://github.com/facebookresearch/dietgpu | MIT [V]; ARCHIVED Jun 2026 | GPU rANS 250-410 GB/s, float (BF16 exponent split) 250-600 GB/s on A100; BF16 ~0.67, FP16 ~0.85 [V]. |
| nvCOMP | https://developer.nvidia.com/nvcomp | proprietary NVIDIA EULA, binaries [V] | ANS/GDeflate/Zstd/LZ4/Bitcomp; no byte-plane split -> 79% on BF16. Do not depend. |
| SplitZip | arXiv 2605.01708 | n/a | KV BF16 exponent entropy 2.89-3.59 bit; fixed-length codes + escape stream (GPU-friendly). |
| Expert delta: D2-MoE (Gu et al.) | ICML'25, arXiv 2502.17298, https://github.com/lliai/D2MoE | [M] verify | Fisher-weighted shared base + per-expert SVD delta + pruning; up to ~80% param reduction, LOSSY. Related: 2510.02345 (expert clustering), 2608.07814 (LorExperts/BTExperts). No exactly-lossless expert-delta scheme found. |

### B. KV cache
| Source | URL | License | Numbers / exactness |
|---|---|---|---|
| KIVI (ICML'24) | https://arxiv.org/abs/2402.02750 | [M] MIT | 2-bit, K per-channel / V per-token, fp residual window; 2.6x less peak mem, 2.35-3.47x throughput. LOSSY. |
| KVQuant (NeurIPS'24) | https://arxiv.org/abs/2401.18079 | [M] verify | pre-RoPE per-channel K, non-uniform dtype, dense-and-sparse; <0.1 ppl at 3-bit; ~1.7x kernel speedup. LOSSY. |
| KVTuner (ICML'25) | https://arxiv.org/abs/2502.04420 | [M] verify | Offline layer-wise K/V precision search; 3.25-bit nearly lossless Llama-3.1-8B, 4-bit Qwen2.5-7B; +21% vs KIVI-KV8. K matters more than V. LOSSY. |
| XQuant (Tomar et al.) | https://arxiv.org/abs/2508.10395 | n/a | Cache layer input X, rematerialize K/V: 10x vs FP16 at +0.01 ppl, 12.5x at +0.1. LOSSY + extra compute; helps MHA, less for GQA/MLA. |
| PagedAttention/vLLM (SOSP'23), SGLang RadixAttention https://arxiv.org/abs/2312.07104 | | [M] Apache-2.0 | Exact prefix reuse: hash of token blocks chained with parent; radix tree + LRU. EXACT only if kernel numerics are identical. |
| Batch-invariant ops (Thinking Machines, 2025) | https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ , https://github.com/thinking-machines-lab/batch_invariant_ops | [M] MIT | Output depends on batch/chunk shape via reduction order; fixed-strategy RMSNorm/matmul/attention gives bit-identical results. |
| LMCache (arXiv 2510.09665) | https://github.com/LMCache/LMCache | [M] Apache-2.0 | KV tiering GPU/CPU/disk/GDS; CacheBlend non-prefix reuse is LOSSY. |
| InfiniGen (OSDI'24) | https://arxiv.org/abs/2406.19707 | [M] verify | Speculative top-token prefetch from host; up to 3x; LOSSY (token selection). |
| DeepSeek-V2 MLA | https://arxiv.org/abs/2405.04434 | | KV -93.3% vs MHA. MLA cache = 512 latent + 64 rope = 576 scalars/token/layer (V3: 61 layers x 576 x 2 B = ~70 KB/token BF16, 32k ctx = ~2.2 GB). Exact by construction. |
| KVTC (ICLR'26, 2511.01815), CacheGen (2310.07240) | | | transform + quant + entropy coding, 20x+; lossy. |

### C. Platform / kernels
| Item | Finding |
|---|---|
| GPUDirect Storage | NVIDIA GDS docs describe Linux VFS only; no Windows support [V]; forum reports cuFile refuses GeForce ("unsupported"), compat mode = POSIX via host RAM, no benefit [V]. -> NOT available. |
| DirectStorage 1.1 (MS) | GDeflate GPU decompression, 64 KB tiles, D3D12 queues [V]. CUDA interop is possible in principle: cuImportExternalMemory with D3D12_RESOURCE handle (needs DEDICATED flag) [V docs]. On GeForce the NVMe->VRAM path still stages via host memory (unverified). GDeflate is LZ-family: expect poor ratio on BF16/quant weights [E]. Defer; optional experiment. |
| Pinned memory on Windows | Forum reports: cudaHostAlloc/Register capped ~50% of RAM on Win10/11 (8 GB of 16) [V forum]. Real budget is far lower given OS + model cache. |
| Marlin (IST-DASLab, arXiv 2408.11743) | [M] Apache-2.0. FP16xINT4, near-ideal 4x to batch 16-32, sm_80+/Ada OK, cp.async pipelines, striped partitioning. Sparse-Marlin for 2:4. |
| Machete (vLLM/Neural Magic) | Hopper wgmma/TMA only -> useless on sm_89. |
| FlashInfer (MLSys'25 best paper, 2501.01005) | [M] Apache-2.0. CUDA-graph-compatible load-balanced attention scheduling, paged/block-sparse KV, MLA support. Ada supported for attention. |
| MegaBlocks (MLSys'23, 2211.15841) | Block-sparse/grouped-GEMM dropless MoE (training oriented, [M] Apache-2.0). Idea (sort tokens by expert, grouped GEMM) is what matters; at batch 1 decode it degenerates to top-K GEMVs. |
| Ada facts | sm_89: mma.sync (incl. FP8 e4m3/e5m2), cp.async, ldmatrix; NO wgmma/TMA/clusters; ~99 KB opt-in smem/block; 32 MB L2. FA3/Machete/CUTLASS-Hopper paths do not apply; FA2-style is the target. |

## 2. Quantitative analysis for this laptop

Assumptions [E unless noted]: NVMe Gen4 sequential ~5 GB/s effective for large aligned reads (measure; repo has a baseline); PCIe 4.0 x8 (AD107) ~12 GB/s practical H2D from pinned memory; VRAM 272 GB/s; DDR5 ~60 GB/s; 16 GB RAM.

Ratio r (compressed/raw) by format (raw = what ships):
| Format | r | Source / status |
|---|---|---|
| BF16 | 0.67-0.68 | DFloat11, ZipNN, DietGPU [V] |
| FP8 e4m3 | 0.85-0.90 | ECF8 real models [V] |
| MXFP4/NVFP4 | ~0.98-1.0 (values raw; scales ~ -5%) | 2508.19263 [V] |
| GPTQ/AWQ int4 | ~0.95-1.0 expected | paper 2606.15789 claims far lower; unverified, measure |
| GGUF Q4_K/Q5_K | ~0.93-1.0 expected (quant codes near-uniform, scale/min fields small gain) | no source found; measure |

Pay-off rule. Let B = source bandwidth (NVMe 5, or RAM->PCIe 12 GB/s) in compressed bytes/s and D = decoder output rate in raw bytes/s.
- Pipelined (read tile i+1 while decoding tile i): throughput = min(B/r, D). Wins whenever D > B; gain = 1/r up to D.
- Serial: wins only if D > B/(1-r): BF16 D > 3.1B (15.6 GB/s at B=5); FP8 D > 7-10B (35-50 GB/s). So always pipeline.
- Expected NVMe-bound gains: BF16 x1.47, FP8 x1.11-1.18, 4-bit ~x1.0-1.07. Compression is therefore a BF16/FP8-checkpoint feature; for MXFP4/int4/Q4_K it is nearly irrelevant for bytes, but per-tile raw fallback costs nothing.
- GPU decode ceiling: read r + write 1 bytes per output byte -> 272/1.68 = ~160 GB/s bandwidth bound; DietGPU-class A100 numbers (250-600 GB/s of 1.5-2 TB/s) scale by BW to ~35-100 GB/s here; plan on 30-60 GB/s. Well above NVMe (5) and PCIe (12 -> 17.6 raw-equivalent for BF16). Decode of a 44 MB FP8 expert (~1.5 ms) is small vs 3.7 ms PCIe / 8.8 ms NVMe for the same expert.
- CPU decode: ZipNN's 80 GB/s is a 16-worker server number [V]; on 12700H expect ~10-25 GB/s [E] but it burns cores and DRAM bandwidth needed for CPU-side experts. Use CPU decode only as fallback/converter.
- Biggest win is the RAM tier: keep tiles compressed in RAM (16 GB, ~8-9 GB cache) -> BF16 effective capacity x1.47, FP8 x1.15, and decode on GPU after H2D of the compressed bytes (PCIe carries r*bytes). VRAM cache holds decoded (compute-ready) tiles.
- Order-of-magnitude reality check: a 671B-class FP8 MoE reads ~350 MB/layer/token of routed experts if all cold; at 5 GB/s that is seconds/token, so hit rate (locality, prefetch), not codec, dominates; the codec is a 10-18% multiplier for FP8.
- Expert dedup/delta: experts in pretrained MoEs are separately trained; published methods are lossy (SVD/low-rank/merge). Exact byte/integer deltas only help upcycled models. Offline test (cheap): per-tensor entropy of (expert_i - expert_j) vs entropy of expert_i; adopt only if bits drop >5%.

## 3. Decisions for LM Atomic

### Codecs (native C/CUDA, no external deps)
1. **RAW** (codec 0): always available, mandatory fallback per tile. Covers MXFP4/int4/Q4_K nibble data.
2. **EXPSPLIT-HUFF** (codec 1), primary for BF16 and FP8: split into planes: exponent plane (BF16: 8-bit exponent; FP8 e4m3: 4-bit exponent, keep sign+mantissa 4 bits raw) and sign+mantissa plane stored raw (BF16: 1+7 bits packed to bytes). Exponent plane Huffman-coded with length limit 11-12 bits, per-tensor shared code table (256 x u8 lengths, canonical codes) stored once per tensor, 2-level LUT in shared memory for decode (DFloat11-style; ~2.6 bit exponent -> BF16 10.7 bit). Use a per-tile gap array (start bit offset per 256-symbol lane group) so decoders don't need a count pass.
3. **EXPSPLIT-rANS** (codec 2), optional second implementation (12-bit probs, 32 interleaved streams per tile, DietGPU/Recoil-style) for tiles where it beats Huffman by >1%; also used for scale/min planes of Q4_K/MXFP4 (tables per tensor). Only add after codec 1 ships and benchmarks show gain.
4. **DELTA-SCALE** (codec 3, later): scale planes (FP8 E8M0/E4M3 block scales, Q4_K d/dmin) coded with byte-delta + Huffman; expected <=5% file gain; low priority.
Skip: nvCOMP (proprietary), DirectStorage/GDeflate (LZ ratio poor, adds D3D12), fused decompress-GEMM (ZipGEMM/DFloat11 style; the streaming design decodes once per load, amortization is with weight reuse across the tokens of a layer; revisit for resident attention/dense weights only), Zstd on payload (use only for metadata).

### Tile format (LMAT tile)
- Logical tile = 64 KiB of uncompressed payload, aligned to quantization blocks (whole rows of BF16/FP8 elements or whole Q4_K super-blocks/MXFP4 blocks; tail tile shorter). 64 KiB matches GDeflate/ZipNN chunk scale (256 KiB in ZipNN; we go smaller for finer per-tile fallback and 24-SM parallelism).
- Expert record = tensors (gate/up/down) each a run of tiles; expert file range starts at 4 KiB boundary (unbuffered I/O alignment); a tile directory (u64 offset, u32 comp_len) per expert enables one contiguous read per expert.
- Tile header (16 B): u8 codec, u8 elem_type, u16 table_id (index into per-tensor table set), u32 raw_len, u32 comp_len, u32 xxh32/crc32c of the RAW (decoded) bytes. Payload: [gap array][exponent stream][sign+mantissa plane]. Per-tile codec selection at conversion time: pick smallest of {RAW, HUFF, rANS}; choose HUFF/rANS only if comp_len <= 0.97 * raw_len (saves nothing otherwise and decode costs GPU time).
- Checksums: per-tile checksum of decoded bytes (verified in a `verify` tool and optionally in a sampled/debug GPU pass); per-tensor SHA-256 of the original checkpoint bytes recorded in the manifest, and the converter must round-trip decode all tiles on the CPU before declaring the file valid. Decoders are integer-only, so exactness is by construction; tests compare against source safetensors/GGUF bytes.

### Exact-reconstruction cache
- Content-addressed: key = (manifest_hash, layer, expert, tensor, tile_index) with the decoded-tile checksum as integrity tag. Tiers: NVMe (compressed file) -> RAM (compressed tiles in pinned/pageable ring) -> VRAM (decoded, compute-ready expert slabs, LRU/frequency-aware). Decode always writes into a VRAM slab; a cache hit never re-decodes. Never store lossy reconstructions in this cache; lossy modes (e.g. lower-precision experts) get a separate namespace tagged with mode id so Exact runs can never read them.
- Pinned staging: small ring (2-4 x expert size, ~0.5-1 GB) of cudaHostAlloc buffers; overlapped ReadFile (FILE_FLAG_NO_BUFFERING|OVERLAPPED, IOCP) directly into pinned buffers; do NOT cudaHostRegister the whole model (Windows ~50% RAM cap, and pinning starves the OS on a 16 GB box).

### KV exact-prefix-reuse validity key
Block (e.g. 64 tokens) hash chain: `key_i = H(key_{i-1}, token_ids_i)` where the root seeds `H(weights_manifest_hash, tokenizer_hash + chat-template id, arch/RoPE config incl. theta/scaling/YaRN, KV dtype+layout+quant config, attention kernel id + reduction/split config, canonical prefill chunk size, sliding-window/sink config, adapter id, exact/lossy mode tag, engine build version)`. Reuse rules for Exact mode: (a) prefix must start at position 0 (no position-independent/CacheBlend reuse - lossy); (b) prefill must use a fixed canonical chunking (e.g. 512 tokens, block aligned) and batch-invariant kernels, otherwise KV computed in different chunkings is not bit-identical; (c) cached KV stored raw or via lossless exponent-split (KV exponent entropy ~3 bit -> ~0.70 for BF16) only; (d) validation test: recompute and memcmp on a sample of hits in CI. For MLA models store only the 576-wide latent (exact and 60x smaller than MHA-equivalent). Lossy KV (KIVI/KVQuant/KVTuner-style 3-4 bit) are optional modes with their own namespace and a measured perplexity/task delta report.

### GPU kernels / execution techniques to adopt
- Attention: FlashAttention-2-style tiled kernels for sm_89 (cp.async double buffering, mma.sync, ~99 KB smem); paged KV blocks (matches prefix hashing); split-KV for long-context decode; fixed reduction order in Exact mode.
- Quantized weights: Marlin-style FP16xINT4 kernel design (cp.async pipeline, offline weight reshuffle for ldmatrix layout) for GPTQ/AWQ/MXFP4-like resident weights; dequant fused in registers. GEMV-specialized path for batch 1 (memory bound at 272 GB/s).
- MoE: at decode, top-K expert GEMVs launched from a device-side pointer table (slot -> VRAM slab); at prefill/batch use sort-by-expert grouped GEMM (MegaBlocks/CUTLASS grouped idea) - custom, no library dependency.
- CUDA Graphs: capture per-layer decode segments (attention + router, then expert compute) with device-side indirection so routing changes do not need re-instantiation; useful on WDDM where launch overhead is high. Use conditional nodes only if needed (CUDA 13.4 has them). Expert H2D/decode stays outside graphs on separate streams (copy stream + decode stream + compute stream, events for slab readiness), prefetching layer N+1 experts (router output known early) while layer N computes.
- Skip: Machete/TMA/wgmma/FA3 (Hopper only), persistent megakernels (later, if launch overhead measured dominant), GDS (unavailable), DirectStorage (defer, optional experiment), fused decompress-GEMM (later, only for resident weights).

### Build order
1. Converter + CPU reference EXPSPLIT-HUFF + verify tool (round-trip vs source bytes). 2. GPU Huffman decoder benchmarked on the 4060 (target >=30 GB/s output). 3. Pipelined reader with pinned ring; measure NVMe and H2D. 4. Per-tile fallback stats on real MXFP4/int4/Q4_K models to settle the "4-bit ~incompressible" assumption (and test the 2606.15789 claim). 5. rANS only if measured gain justifies.
