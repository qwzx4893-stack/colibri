#ifndef LMA_BACKEND_CUDA_H
#define LMA_BACKEND_CUDA_H

#include <stddef.h>
#include <stdint.h>

/* LMA_CUDA_DLLEXPORT marks functions exported from lma_cuda.dll on Windows.
 * Define LMA_CUDA_BUILDING_DLL when compiling the .cu into the DLL (so the
 * functions are __declspec(dllexport)); the host loader does NOT include this
 * header's declarations — it resolves symbols at runtime via GetProcAddress. */
#if defined(_WIN32) && defined(LMA_CUDA_BUILDING_DLL)
#define LMA_CUDA_DLLEXPORT __declspec(dllexport)
#else
#define LMA_CUDA_DLLEXPORT
#endif


#ifdef __cplusplus
extern "C" {
#endif

#define LMA_CUDA_MAX_DEVICES 16

/* Weight formats the generic per-element device decoder (weight_at,
 * backend_cuda.cu) can actually decode: f32, int8-row, int4 nibbles (fmt=2 and
 * the grouped fmt=4, same packing), int2, and fmt=8 (fp8-e4m3 raw bytes,
 * decoded through the c_e4m3 LUT -- absorb-path support; absorb_scale supplies
 * its per-128x128-block scale). Nothing else.
 *
 * WHY THIS IS A PREDICATE AND NOT A COMMENT. weight_at used to END in the int2
 * decode as an unguarded fall-through, so ANY other format handed to it -- a
 * format with a different element width, a different scale geometry, or no
 * device decoder at all -- was silently read as 2-bit values and returned
 * plausible-looking numbers. The CPU twins refuse the same input loudly:
 * qt_addrow and qt_matvec_rows (glm52.c) both exit(1) naming the function and
 * the fmt. Two backends given identical unsupported input, one refusing and one
 * fabricating, is the defect -- not the missing decoder.
 *
 * Defined here, in the host header, rather than inside the .cu: the host gates
 * that keep unsupported formats off the device (absorb_fmt_ok and friends) and
 * the device-side backstop must agree by construction rather than by two people
 * writing the same list twice, and a CPU test build can then unit-test the truth
 * table without a GPU or a CUDA toolchain (tests/test_cuda_fmt_guard.c) -- the
 * same arrangement glm52.c uses for metal_fused_fmt_ok.
 *
 * NOT a statement about which formats the CUDA BACKEND supports: quant_matmul
 * has its own explicit branches for fmt=6 (E8/IQ3) and fmt=7 (MXFP4) that
 * never route through weight_at (and its own fmt=8 branch for the dense path
 * -- weight_at's fmt=8 branch serves the absorb kernels, which share the same
 * c_e4m3 LUT). This predicate is scoped to weight_at's own dispatch, which is
 * what the absorb and grouped-expert kernels decode through.
 *
 * fmt=8 CAVEAT, stated because the truth table alone cannot carry it: a fmt=8
 * decode additionally requires the e4m3 LUT to have been published to the
 * configured devices (lma_cuda_fp8_set_lut). The exact mechanism, so the
 * claim cannot outrun it: lma_cuda_fp8_set_lut copies the table into every
 * context live AT CALL TIME and sets a process-wide flag; the flag gates
 * fmt=8 uploads (lma_cuda_tensor_upload refuses until it is set).
 * lma_cuda_shutdown clears the flag; lma_cuda_init never writes it. That is
 * enough because init will not rebuild contexts underneath a live set: a
 * re-init naming the same device set returns success and leaves the contexts,
 * and the table published to them, untouched, while one naming a different
 * device set is refused before any context is rebuilt. So the device set
 * cannot widen past what the last publish covered without going through
 * shutdown, and no fmt=8 LmaCudaTensor can reach a kernel whose device has
 * an unwritten table. This predicate
 * deliberately does not restate that gate: it answers "does weight_at have a
 * decode branch for this fmt", which is the question the launch-site gates
 * and the device-side __trap() backstop share. */
static inline int lma_cuda_weight_at_supported(int fmt) {
    return fmt == 0 || fmt == 1 || fmt == 2 || fmt == 3 || fmt == 4 || fmt == 8;
}

/* The two decisions the fmt=8 LUT gate rests on, as pure predicates. They live
 * here rather than inline in backend_cuda.cu so a host-side test can pin them
 * with no CUDA toolchain and no GPU (tests/test_cuda_lut_gate.c). backend_cuda.cu
 * calls BOTH at the real decision sites, so the test pins the engine's own
 * logic rather than a second copy that could drift from it -- which is the
 * failure this factoring exists to prevent, the gate having no CI reach
 * otherwise. */

/* Does the upload gate admit this tensor? Only fmt=8 needs the published
 * table; every other format decodes without one. */
static inline int lma_cuda_fp8_gate_admits(int fmt, int lut_ready) {
    return fmt != 8 || lut_ready != 0;
}

/* What lma_cuda_init must do with a request while a device set may be live.
 * BUILD: nothing is live, build the contexts. ACCEPT: the same set is already
 * live -- return success and touch nothing, so the table published to those
 * contexts stays valid. REFUSE: a different set is live -- refuse before
 * rebuilding anything, so the set cannot widen past the last publish. */
enum { LMA_CUDA_INIT_BUILD = 0, LMA_CUDA_INIT_ACCEPT = 1, LMA_CUDA_INIT_REFUSE = -1 };
static inline int lma_cuda_init_disposition(int nctx, int count,
                                             const int *want, const int *live) {
    int i;
    if (nctx <= 0) return LMA_CUDA_INIT_BUILD;
    if (count != nctx) return LMA_CUDA_INIT_REFUSE;
    for (i = 0; i < count; i++) if (want[i] != live[i]) return LMA_CUDA_INIT_REFUSE;
    return LMA_CUDA_INIT_ACCEPT;
}

/* Opaque, persistent device copy of one resident quantized tensor. */
typedef struct LmaCudaTensor LmaCudaTensor;

/* Devices are CUDA ordinals, not positions in the input list.
 * Repeating the same ordered list preserves active contexts. Changing an
 * active list returns 0 without replacing it; release tensors and shut down
 * before selecting a different list. Init/shutdown require caller serialization. */
LMA_CUDA_DLLEXPORT int lma_cuda_init(const int *devices, int count);
LMA_CUDA_DLLEXPORT void lma_cuda_shutdown(void);
/* Number of CUDA devices visible to this process, before a device list is
 * selected. Returns 0 when the runtime cannot discover any device. */
LMA_CUDA_DLLEXPORT int lma_cuda_available_device_count(void);
LMA_CUDA_DLLEXPORT int lma_cuda_device_count(void);
LMA_CUDA_DLLEXPORT int lma_cuda_device_at(int index);
LMA_CUDA_DLLEXPORT int lma_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes);
LMA_CUDA_DLLEXPORT int lma_cuda_device_integrated(int device);
/* device < 0 returns aggregate statistics for all configured devices. */
LMA_CUDA_DLLEXPORT void lma_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes);
LMA_CUDA_DLLEXPORT void lma_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms);
/* Per-device form of lma_cuda_group_stats; unknown devices return zeros. */
LMA_CUDA_DLLEXPORT void lma_cuda_group_stats_device(
    int device, uint64_t *calls, uint64_t *experts, uint64_t *rows,
    double *h2d_ms, double *kernel_ms, double *d2h_ms);

/* Publish the E8 codebook (quant.h's e8_grid, 256x4 bytes) to every configured
 * device. Must be called after lma_cuda_init and before any fmt=6 upload; the
 * backend keeps no copy of the table so it cannot drift from the CPU decoder. */
LMA_CUDA_DLLEXPORT int lma_cuda_e8_set_grid(const void *grid);

/* Publish the fmt=8 e4m3 decode table (quant.h's E4M3_LUT, 256 f32) the same
 * way. Must be called after lma_cuda_init; fmt=8 uploads are refused until it
 * succeeds, because kernels would decode against a zero-initialized table. */
LMA_CUDA_DLLEXPORT int lma_cuda_fp8_set_lut(const float *lut);

/* Upload without executing, so capacity failures happen during model startup. */
LMA_CUDA_DLLEXPORT int lma_cuda_tensor_upload_g(LmaCudaTensor **tensor,
        const void *weights, const float *scales,
        int fmt, int I, int O, int device, int gs);
LMA_CUDA_DLLEXPORT int lma_cuda_tensor_upload(LmaCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);
#ifdef LMA_ANS
/* Experimental Linux-only GPU-resident entropy tier. The archive remains in
 * VRAM and is decoded into per-device scratch immediately before a grouped
 * expert launch. */
LMA_CUDA_DLLEXPORT int lma_cuda_tensor_upload_compressed(LmaCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);
#endif

/*
 * y[S,O] = x[S,I] @ W[O,I]^T.
 * fmt matches QT in glm.c: 0=f32, 1=int8, 2=int4, 3=int2, 4=grouped int4.
 * gs is the group size for fmt=4 (0 for all other formats).
 * The first successful call uploads W and its scales; later calls reuse it.
 * Returns 1 on success and 0 when CUDA is not initialized or the format is invalid.
 */
/* y[S,O] = x[S,I] @ dequant_mxfp4(W[O,I])^T, fmt=7 (OCP microscaling FP4).
 * q4 is [O, ceil(I/2)] e2m1 nibbles, e8s is [O, ceil(I/32)] ue8m0 exponents --
 * BYTES, not floats, which is why this is not folded into lma_cuda_matmul.
 * Stateless: weights are uploaded per call, matching the streaming expert tier
 * Kimi K3 uses. Returns 1 on success, 0 (y untouched) to fall back to CPU. */
LMA_CUDA_DLLEXPORT int lma_cuda_matmul_mxfp4(float *y, const float *x,
                                               const unsigned char *q4,
                                               const unsigned char *e8s,
                                               int S, int I, int O);

/* Streaming Kimi expert: down(SiTU(gate(x), up(x))). Weights are MXFP4
 * host buffers; intermediate activations remain on device. No weight cache.
 * Returns 0 on failure; callers must accumulate y only after success. */
LMA_CUDA_DLLEXPORT int lma_cuda_expert_mxfp4(float *y, const float *x,
        const unsigned char *gate_w, const unsigned char *gate_s,
        const unsigned char *up_w, const unsigned char *up_s,
        const unsigned char *down_w, const unsigned char *down_s,
        int S, int D, int I, float b1, float b2);

LMA_CUDA_DLLEXPORT int lma_cuda_matmul(LmaCudaTensor **tensor,
                     float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device, int gs);

/* Fused expert pipeline: y = down(silu(gate(x)) * up(x)).  All three tensors
 * must already be resident on one device.  Activations cross PCIe once in
 * each direction instead of once per matrix. */
LMA_CUDA_DLLEXPORT int lma_cuda_expert_mlp(LmaCudaTensor *gate, LmaCudaTensor *up,
                         LmaCudaTensor *down, float *y, const float *x, int S);

/* Prefill-oriented shared expert path.  INT4 weights stay packed in global
 * memory, activations are converted to FP16 per tile, and Tensor Cores
 * accumulate into FP32.  Unlike LMA_CUDA_TC_INT4 this does not quantize the
 * activation to INT4. */
LMA_CUDA_DLLEXPORT int lma_cuda_shared_mlp_w4a16(LmaCudaTensor *gate, LmaCudaTensor *up,
                               LmaCudaTensor *down, float *y,
                               const float *x, int S);

/* Packed group of same-shaped experts. Inputs and outputs contain sum(rows)
 * consecutive [D] rows in call order. */
/* Async issue/take split of the group call below (Inc.4): issue launches on the
 * device stream and returns; take syncs and returns the pinned result rows (valid
 * until the next issue on that device). Small totals only (<=8 rows); one
 * outstanding issue per device. */
LMA_CUDA_DLLEXPORT int lma_cuda_expert_group_issue(LmaCudaTensor *const *gates,
                               LmaCudaTensor *const *ups,
                               LmaCudaTensor *const *downs,
                               const int *rows, int count, const float *x);
LMA_CUDA_DLLEXPORT const float *lma_cuda_expert_group_take(int device);

LMA_CUDA_DLLEXPORT int lma_cuda_expert_group(LmaCudaTensor *const *gates,
                           LmaCudaTensor *const *ups,
                           LmaCudaTensor *const *downs,
                           const int *rows, int count,
                           float *y, const float *x);
/* Same operation, but force the small-batch grouped kernel family when
 * pin_small_batch is nonzero.  Speculative verification uses this to keep
 * CUDA on the same numeric family as S=1 regardless of accepted draft depth. */
LMA_CUDA_DLLEXPORT int lma_cuda_expert_group_pinned(LmaCudaTensor *const *gates,
                           LmaCudaTensor *const *ups,
                           LmaCudaTensor *const *downs,
                           const int *rows, int count,
                           float *y, const float *x, int pin_small_batch);

/* Decode-only MLA weight-absorption core for one token. kv_b is [H*(Q+V),K]. */
LMA_CUDA_DLLEXPORT int lma_cuda_attention_absorb(LmaCudaTensor *kv_b,float *ctx,const float *q,
                               const float *latent,const float *rope,int H,int Q,
                               int R,int V,int K,int T,float attention_scale);

/* Causal MLA absorption for S contiguous rows from one sequence.  The KV
 * arrays contain T rows ending at the final query; query s attends T-S+s+1
 * rows.  One transfer and one launch replace S host round-trips. */
LMA_CUDA_DLLEXPORT int lma_cuda_attention_absorb_batch(LmaCudaTensor *kv_b,float *ctx,const float *q,
                                     const float *latent,const float *rope,int S,
                                     int H,int Q,int R,int V,int K,int T,
                                     float attention_scale);

/* Same attention batch followed immediately by resident o_proj on the same
 * device.  Only the final [S,D] tensor crosses back to the host. */
LMA_CUDA_DLLEXPORT int lma_cuda_attention_project_batch(LmaCudaTensor *kv_b,LmaCudaTensor *o_proj,
                                      float *out,const float *q,const float *latent,
                                      const float *rope,int S,int H,int Q,int R,
                                      int V,int K,int T,float attention_scale);

LMA_CUDA_DLLEXPORT int lma_cuda_attention_project_ragged(LmaCudaTensor *kv_b,LmaCudaTensor *o_proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int max_t,float attention_scale);

LMA_CUDA_DLLEXPORT void lma_cuda_tensor_free(LmaCudaTensor *tensor);
LMA_CUDA_DLLEXPORT size_t lma_cuda_tensor_bytes(const LmaCudaTensor *tensor);
LMA_CUDA_DLLEXPORT size_t lma_cuda_alloc_footprint(size_t bytes);
LMA_CUDA_DLLEXPORT size_t lma_cuda_tensor_vram(const LmaCudaTensor *tensor);
LMA_CUDA_DLLEXPORT int lma_cuda_tensor_device(const LmaCudaTensor *tensor);

/* Replace a resident tensor's contents without reallocating its device slot. */
LMA_CUDA_DLLEXPORT int lma_cuda_tensor_update(LmaCudaTensor *tensor,
                            const void *weights, const float *scales);

/* ---- resident-pipeline primitives (Inc.0): device-pointer entry points ---- */
LMA_CUDA_DLLEXPORT float *lma_cuda_pipe_scratch(int device,int slot,size_t bytes);
LMA_CUDA_DLLEXPORT void *lma_cuda_pipe_alloc(int device,size_t bytes);
LMA_CUDA_DLLEXPORT void lma_cuda_pipe_free(int device,void *p);
LMA_CUDA_DLLEXPORT int lma_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes);
LMA_CUDA_DLLEXPORT int lma_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes);
LMA_CUDA_DLLEXPORT int lma_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev,
                           const float *w_dev,int S,int D,float eps);
LMA_CUDA_DLLEXPORT int lma_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows,
                        int stride,int offset,int R,int heads,float theta);
LMA_CUDA_DLLEXPORT int lma_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n);
LMA_CUDA_DLLEXPORT int lma_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n);
LMA_CUDA_DLLEXPORT int lma_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev,
                            const int *rows_dev,int nrows,int D);
LMA_CUDA_DLLEXPORT int lma_cuda_pipe_gemm(LmaCudaTensor *t,float *y_dev,const float *x_dev,int S);
LMA_CUDA_DLLEXPORT int lma_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev,
                             const float *w_dev,int S,int D,float eps,
                             int xstride,int ystride);
LMA_CUDA_DLLEXPORT int lma_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows,
                             int stride,int offset,int R,int heads,float theta);
LMA_CUDA_DLLEXPORT int lma_cuda_expert_group_resident_issue(LmaCudaTensor *const *gates,
        LmaCudaTensor *const *ups, LmaCudaTensor *const *downs,
        const float *weights, int count,
        int home_device, const float *x_src_dev, float *partial_slot_dev);
LMA_CUDA_DLLEXPORT int lma_cuda_expert_group_resident_take(int home_device,const int *devices,
        int n_issued,float *slots_dev,float *acc_dev,int D);
LMA_CUDA_DLLEXPORT int lma_cuda_pipe_router(int device,const float *x_dev,
        const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,
        float topp,int norm_topk,float routed_scale,
        int *idx_host,float *w_host,int *keff_host);
LMA_CUDA_DLLEXPORT int lma_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src,
                          int spitch,int width,int height);
LMA_CUDA_DLLEXPORT int lma_cuda_attention_project_batch_dev(LmaCudaTensor *kv_b,LmaCudaTensor *o_proj,
        float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
LMA_CUDA_DLLEXPORT int lma_cuda_attention_absorb_batch_dev(LmaCudaTensor *kv_b_shard,float *ctx_dev,
        const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
LMA_CUDA_DLLEXPORT int lma_cuda_attention_absorb_kvdev(LmaCudaTensor *kv_b,float *ctx,const float *q,
        const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T,
        float scale);
LMA_CUDA_DLLEXPORT int lma_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev,
                             const float *src,size_t bytes);
LMA_CUDA_DLLEXPORT int lma_cuda_attention_project_batch_dev_out(LmaCudaTensor *kv_b,LmaCudaTensor *o_proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
LMA_CUDA_DLLEXPORT int lma_cuda_pipe_sync(int device);

#ifdef __cplusplus
}
#endif

#endif
