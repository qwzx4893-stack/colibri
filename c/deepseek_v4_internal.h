#ifndef LMA_DEEPSEEK_V4_INTERNAL_H
#define LMA_DEEPSEEK_V4_INTERNAL_H

/*
 * Internal DeepSeek-V4 API. Not a stability commitment.
 * External callers should use deepseek_v4.h (engine / config / prompt).
 */
#include "deepseek_v4.h"

#include <stdio.h>
#include "tensor.h"
#include "expert_store.h"
#include "expert_store_registry.h"
#include "native_quant.h"
#include "native_quant_batch.h"
#include "native_quant_dual.h"
#include "native_quant_fp4_rows16.h"
#include "st.h"

/* Worker count for the persistent expert-loader pool in the block pipeline
 * (deepseek_v4_block_pipeline.c). Shared here so the CLI can size the OpenMP
 * team around the loaders instead of scheduling compute onto their CPUs. */
#ifndef LMA_V4_EXPERT_LOADER_COUNT
#define LMA_V4_EXPERT_LOADER_COUNT 3
#endif

#define LMA_ST_MAX_RANK ST_MAX_RANK
#define LMA_ST_BF16 0
#define LMA_ST_F16 1
#define LMA_ST_F32 2
#define LMA_ST_U8 3
#define LMA_ST_I8 3
#define LMA_ST_F8_E4M3 4
#define LMA_ST_F8_E8M0 5
#define LMA_ST_I64 6

typedef int LmaSafetensorsDType;
typedef st_tensor LmaSafetensorsTensor;
typedef shards LmaSafetensorsIndex;

typedef struct {
    LmaTensorView view;
    void *data_allocation;
    void *scale_allocation;
} LmaOwnedTensor;

typedef struct {
    float *data;
    uint64_t count;
    int rank;
    int64_t shape[LMA_ST_MAX_RANK];
} LmaFloatTensor;

int lma_st_index_open(LmaSafetensorsIndex **out, const char *directory,
                       char *error, size_t error_size);
void lma_st_index_close(LmaSafetensorsIndex *index);
size_t lma_st_tensor_count(const LmaSafetensorsIndex *index);
size_t lma_st_shard_count(const LmaSafetensorsIndex *index);
const char *lma_st_shard_path(const LmaSafetensorsIndex *index, int shard);
const LmaSafetensorsTensor *lma_st_find(const LmaSafetensorsIndex *index,
                                         const char *name);
int lma_st_tensor_shard(const LmaSafetensorsIndex *index,
                         const LmaSafetensorsTensor *tensor);
int lma_st_read_tensor(const LmaSafetensorsIndex *index,
                        const LmaSafetensorsTensor *tensor, void *destination);
int lma_st_read_at(const LmaSafetensorsIndex *index, int shard,
                    uint64_t offset, size_t length, void *destination);
/* Large, transient SSD read: prefer the index's O_DIRECT twin and use an
 * aligned bounce buffer, falling back to the ordinary buffered path. */
int lma_st_read_at_streaming(const LmaSafetensorsIndex *index, int shard,
                              uint64_t offset, size_t length,
                              void *destination);
int lma_st_streaming_direct_available(const LmaSafetensorsIndex *index,
                                       int shard);
int lma_st_prefetch_at(const LmaSafetensorsIndex *index, int shard,
                        uint64_t offset, size_t length);
const char *lma_st_dtype_name(LmaSafetensorsDType dtype);

/* ==== begin dual-SSD mirror (LMA_MODEL_MIRROR / SNAP_MIRROR) ==== */

/* Registers the read replicas listed in LMA_MODEL_MIRROR (or SNAP_MIRROR) on
 * `index` and derives the per-drive expert read split from LMA_DISK_WEIGHTS
 * or a startup bandwidth probe (glm52.c mirror_setup semantics). Runs after
 * the index is open and before any expert load. Returns 1 when a usable mirror
 * is active, 0 when none, -1 on error. */
int lma_st_mirror_setup(LmaSafetensorsIndex *index, const char *model_dir,
                         int experts_per_layer);
int lma_st_streaming_direct_available_rep(const LmaSafetensorsIndex *index,
                                           int shard, int rep);

/* Replica (0 = primary, 1..nrep-1 = mirrors) serving expert (layer, eid). */
int lma_st_expert_route(int layer, int eid);
int lma_st_mirror_active(void);
int lma_st_mirror_nrep(void);

/* Rep-aware reads: route to the replica fd (falling back to the primary when
 * the shard is absent there or on read error) and account bytes per drive. */
int lma_st_read_at_rep(const LmaSafetensorsIndex *index, int shard, int rep,
                        uint64_t offset, size_t length, void *destination);
int lma_st_read_at_streaming_rep(const LmaSafetensorsIndex *index, int shard,
                                  int rep, uint64_t offset, size_t length,
                                  void *destination);
int lma_st_prefetch_at_rep(const LmaSafetensorsIndex *index, int shard,
                            int rep, uint64_t offset, size_t length);

/* Per-drive I/O telemetry (bytes / read count), index [0] primary, [r] mirror. */
extern uint64_t g_v4_mir_bytes[1 + ST_MAX_MIR];
extern uint64_t g_v4_mir_nread[1 + ST_MAX_MIR];
/* ==== end dual-SSD mirror ==== */

int lma_tensor_load_fp8(LmaOwnedTensor *output,
                         const LmaSafetensorsIndex *index,
                         const char *prefix, char *error, size_t error_size);
void lma_owned_tensor_free(LmaOwnedTensor *tensor);
int lma_tensor_load_f32(LmaFloatTensor *output,
                         const LmaSafetensorsIndex *index,
                         const char *name, char *error, size_t error_size);
void lma_float_tensor_free(LmaFloatTensor *tensor);

typedef struct LmaV4Engine LmaV4Engine;

/* Runtime-selected full DSpark profile, shared with the separately compiled
 * generation unit. */
extern int lma_v4_full_dspark_wanted;
double lma_v4_dspark_cache_gb(void);

/* ==== begin deepseek_v4_math.h ==== */

#ifdef __cplusplus
extern "C" {
#endif

int lma_v4_hc_split_sinkhorn(float *pre, float *post, float *comb,
                              const float *mixes, const float scale[3],
                              const float *base, int hc, int iterations,
                              float eps);

int lma_v4_hc_pre(float *output, float *post, float *comb,
                   const float *input, const float *hc_fn,
                   const float scale[3], const float *base,
                   int hc, int dimension, int iterations,
                   float norm_eps, float hc_eps);

int lma_v4_hc_post(float *output, const float *branch,
                    const float *residual, const float *post,
                    const float *comb, int hc, int dimension);

int lma_v4_rmsnorm(float *output, const float *input, const float *weight,
                    int dimension, float eps);

int lma_v4_rope_precompute(float *cosines, float *sines,
                            int dimension, int sequence_length,
                            int original_sequence_length, float base,
                            float factor, int beta_fast, int beta_slow);

int lma_v4_rope_precompute_range(float *cosines, float *sines,
                                  int dimension, int start_position,
                                  int sequence_length,
                                  int original_sequence_length, float base,
                                  float factor, int beta_fast, int beta_slow);

int lma_v4_rope_position(float *cosines, float *sines,
                          int dimension, int position,
                          int original_sequence_length, float base,
                          float factor, int beta_fast, int beta_slow);

int lma_v4_rope_apply(float *vectors, int vector_count, int dimension,
                       const float *cosines, const float *sines, int inverse);

int lma_v4_route(float *weights, int *indices, const float *hidden,
                  const float *gate, const float *bias,
                  const int *forced_indices, int experts, int dimension,
                  int topk, float route_scale);

int lma_v4_swiglu(float *output, const float *gate, const float *up,
                   int dimension, float limit);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_math.h ==== */

/* ==== begin deepseek_v4_layer.h ==== */

#include <stddef.h>
#include <stdint.h>

/* amalgamated: deepseek_v4_config.h */

#ifdef __cplusplus
extern "C" {
#endif

#define LMA_V4_MAX_LAYER_TENSORS 48
#define LMA_V4_MAX_TENSOR_NAME 160

typedef struct {
    char name[LMA_V4_MAX_TENSOR_NAME];
    LmaSafetensorsDType dtype;
    int rank;
    int64_t shape[LMA_ST_MAX_RANK];
    /* Dense FP8 weights may be transposed inside each 8-row tile after load.
     * This is an in-memory execution layout only; checkpoint bytes and scales
     * remain unchanged. */
    int packed_rows8;
} LmaDeepSeekV4TensorSpec;

typedef struct {
    int layer;
    int compression_ratio;
    int uses_hash_router;
    int has_compressor;
    int has_indexer;
    size_t tensor_count;
    LmaDeepSeekV4TensorSpec tensors[LMA_V4_MAX_LAYER_TENSORS];
} LmaDeepSeekV4LayerPlan;

typedef struct {
    size_t tensor_count;
    uint64_t total_bytes;
    uint64_t bf16_bytes;
    uint64_t f32_bytes;
    uint64_t fp8_weight_bytes;
    uint64_t fp8_scale_bytes;
    uint64_t i64_bytes;
} LmaDeepSeekV4LayerStats;

typedef struct {
    LmaDeepSeekV4LayerPlan plan;
    LmaDeepSeekV4LayerStats stats;
    void *data[LMA_V4_MAX_LAYER_TENSORS];
    /* Optional per-tensor backend-resident mirrors (Dsv4CudaTensor* on the CUDA
     * tier). Aligned 1:1 with plan.tensors[]; owned by the engine's GPU tier. */
    void *gpu[LMA_V4_MAX_LAYER_TENSORS];
} LmaDeepSeekV4LayerWeights;

int lma_v4_layer_plan(LmaDeepSeekV4LayerPlan *plan,
                       const LmaDeepSeekV4Config *config, int layer,
                       char *error, size_t error_size);
int lma_v4_layer_validate(const LmaDeepSeekV4LayerPlan *plan,
                           const LmaSafetensorsIndex *index,
                           LmaDeepSeekV4LayerStats *stats,
                           char *error, size_t error_size);
int lma_v4_layer_load(LmaV4Engine *engine,
                       LmaDeepSeekV4LayerWeights *weights,
                       const LmaDeepSeekV4Config *config,
                       const LmaSafetensorsIndex *index, int layer,
                       char *error, size_t error_size);
void lma_v4_layer_free(LmaV4Engine *engine,
                        LmaDeepSeekV4LayerWeights *weights);
const void *lma_v4_layer_data(const LmaDeepSeekV4LayerWeights *weights,
                               const char *name,
                               const LmaDeepSeekV4TensorSpec **spec);

/* Backend-mirror accessors: the gpu handle attached to the tensor named
 * "layers.<N>.<suffix>.weight", or NULL when the tier did not upload it. */
void *lma_v4_layer_gpu(const LmaDeepSeekV4LayerWeights *weights,
                        const char *suffix);
int lma_v4_layer_gpu_set(LmaDeepSeekV4LayerWeights *weights,
                          const char *suffix, void *handle);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_layer.h ==== */

/* ==== begin deepseek_v4_sparse_attention.h ==== */

#ifdef __cplusplus
extern "C" {
#endif

int lma_v4_sparse_attention_ref(float *output, const float *queries,
                                 const float *kv, const float *sinks,
                                 const int *indices, int heads,
                                 int head_dimension, int kv_count, int topk,
                                 float softmax_scale);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_sparse_attention.h ==== */

/* ==== begin deepseek_v4_kv_cache.h ==== */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LmaDeepSeekV4KVCache LmaDeepSeekV4KVCache;

int lma_v4_kv_cache_create(LmaDeepSeekV4KVCache **cache,
                            int window_size, int compression_ratio,
                            int head_dimension, int max_context);
void lma_v4_kv_cache_reset(LmaDeepSeekV4KVCache *cache);
void lma_v4_kv_cache_destroy(LmaDeepSeekV4KVCache *cache);
int lma_v4_kv_cache_put_window(LmaDeepSeekV4KVCache *cache,
                                int position, const float *kv);
int lma_v4_kv_cache_put_compressed(LmaDeepSeekV4KVCache *cache,
                                    int position, const float *kv);
int lma_v4_kv_cache_indices(const LmaDeepSeekV4KVCache *cache,
                             int position, int *indices, size_t capacity);
const float *lma_v4_kv_cache_values(const LmaDeepSeekV4KVCache *cache);
int lma_v4_kv_cache_value_count(const LmaDeepSeekV4KVCache *cache);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_kv_cache.h ==== */

/* ==== begin deepseek_v4_attention_cache.h ==== */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LmaDeepSeekV4AttentionCache LmaDeepSeekV4AttentionCache;

int lma_v4_attention_cache_create(LmaDeepSeekV4AttentionCache **cache,
                                   int window_size, int compression_ratio,
                                   int head_dimension, int max_context);
void lma_v4_attention_cache_reset(LmaDeepSeekV4AttentionCache *cache);
void lma_v4_attention_cache_destroy(LmaDeepSeekV4AttentionCache *cache);

/* query is [heads, head_dimension]. window_kv and compressed_kv have one
 * head_dimension vector each. compressed_kv is required at ratio boundaries. */
int lma_v4_attention_cache_step(LmaDeepSeekV4AttentionCache *cache,
                                 float *output, const float *query,
                                 const float *window_kv,
                                 const float *compressed_kv,
                                 const float *sinks, int heads,
                                 int position, float softmax_scale);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_attention_cache.h ==== */

/* ==== begin deepseek_v4_attention.h ==== */

#include <stddef.h>

/* amalgamated: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_layer.h */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LmaDeepSeekV4WindowAttentionState
    LmaDeepSeekV4WindowAttentionState;

int lma_v4_window_attention_prepare(LmaDeepSeekV4WindowAttentionState *state,
                                     const LmaDeepSeekV4LayerWeights *weights,
                                     const LmaDeepSeekV4Config *config,
                                     char *error, size_t error_size);
int lma_v4_window_attention_create(LmaDeepSeekV4WindowAttentionState **state,
                                    const LmaDeepSeekV4Config *config);
void lma_v4_window_attention_reset(LmaDeepSeekV4WindowAttentionState *state);
void lma_v4_window_attention_destroy(LmaDeepSeekV4WindowAttentionState *state);

/* Correctness-first single-KV attention. Compressed layers may use this at
 * position zero, before any compressed KV/indexer candidate exists. */
int lma_v4_attention_token_ref(float *output,
                                const LmaDeepSeekV4LayerWeights *weights,
                                const LmaDeepSeekV4Config *config,
                                const float *input, int position,
                                char *error, size_t error_size);
int lma_v4_attention_window_token_ref(
    float *output, LmaDeepSeekV4WindowAttentionState *state,
    const LmaDeepSeekV4LayerWeights *weights,
    const LmaDeepSeekV4Config *config, const float *input, int position,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_attention.h ==== */

/* ==== begin deepseek_v4_attention_batch.h ==== */

/* amalgamated: deepseek_v4_attention.h */

int lma_v4_attention_window_batch_ref(
    float *outputs, LmaDeepSeekV4WindowAttentionState *state,
    const LmaDeepSeekV4LayerWeights *weights,
    const LmaDeepSeekV4Config *config, const float *inputs,
    int start_position, int batch, char *error, size_t error_size);
/* ==== end deepseek_v4_attention_batch.h ==== */

/* ==== begin deepseek_v4_attention_transaction.h ==== */

/* amalgamated: deepseek_v4_attention.h */

typedef struct LmaV4AttentionSnapshot LmaV4AttentionSnapshot;

int lma_v4_attention_snapshot_create(
    const LmaDeepSeekV4WindowAttentionState *state,
    LmaV4AttentionSnapshot **output);
int lma_v4_attention_snapshot_restore(
    LmaDeepSeekV4WindowAttentionState *state,
    const LmaV4AttentionSnapshot *snapshot);
void lma_v4_attention_snapshot_destroy(LmaV4AttentionSnapshot *snapshot);
/* Disk (de)serialization of a snapshot (little-endian raw fields + arrays).
 * write returns 0 on success; read allocates *output (0 on success). */
int lma_v4_attention_snapshot_write(const LmaV4AttentionSnapshot *snapshot,
                                     FILE *stream);
int lma_v4_attention_snapshot_read(FILE *stream,
                                    LmaV4AttentionSnapshot **output);
/* ==== end deepseek_v4_attention_transaction.h ==== */

/* ==== begin deepseek_v4_compressor.h ==== */

#include <stddef.h>

/* amalgamated: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_layer.h */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LmaDeepSeekV4CompressorState LmaDeepSeekV4CompressorState;

typedef struct {
    const char *prefix;
    int head_dimension;
    int rotate_fp4;
} LmaDeepSeekV4CompressorOptions;

int lma_v4_compressor_create(LmaDeepSeekV4CompressorState **state,
                              const LmaDeepSeekV4LayerWeights *weights,
                              const LmaDeepSeekV4Config *config,
                              char *error, size_t error_size);
int lma_v4_compressor_create_with_options(
    LmaDeepSeekV4CompressorState **state,
    const LmaDeepSeekV4LayerWeights *weights,
    const LmaDeepSeekV4Config *config,
    const LmaDeepSeekV4CompressorOptions *options,
    char *error, size_t error_size);
void lma_v4_compressor_reset(LmaDeepSeekV4CompressorState *state);
int lma_v4_compressor_bind_weights(LmaDeepSeekV4CompressorState *state,
                                    const LmaDeepSeekV4LayerWeights *weights,
                                    char *error, size_t error_size);
void lma_v4_compressor_destroy(LmaDeepSeekV4CompressorState *state);

/* Processes one decode token. produced is set to one only when a complete
 * compression window emits a KV vector. output may be NULL on other steps. */
int lma_v4_compressor_step(LmaDeepSeekV4CompressorState *state,
                            float *output, int *produced,
                            const float *input, int position,
                            char *error, size_t error_size);

/* Like lma_v4_compressor_step, but consumes precomputed wkv/wgate matvec
 * rows (projection_dim floats each, e.g. batched on the GPU) instead of
 * projecting input itself. State updates are identical. */
int lma_v4_compressor_advance(LmaDeepSeekV4CompressorState *state,
                               float *output, int *produced,
                               const float *kv_proj, const float *gate_proj,
                               int position, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_compressor.h ==== */

/* ==== begin deepseek_v4_compressor_snapshot.h ==== */

/* amalgamated: deepseek_v4_compressor.h */

typedef struct LmaV4CompressorSnapshot LmaV4CompressorSnapshot;

int lma_v4_compressor_snapshot_create(
    const LmaDeepSeekV4CompressorState *state,
    LmaV4CompressorSnapshot **output);
int lma_v4_compressor_snapshot_restore(
    LmaDeepSeekV4CompressorState *state,
    const LmaV4CompressorSnapshot *snapshot);
void lma_v4_compressor_snapshot_destroy(LmaV4CompressorSnapshot *snapshot);
int lma_v4_compressor_snapshot_write(const LmaV4CompressorSnapshot *snapshot,
                                      FILE *stream);
int lma_v4_compressor_snapshot_read(FILE *stream,
                                     LmaV4CompressorSnapshot **output);
/* ==== end deepseek_v4_compressor_snapshot.h ==== */

/* ==== begin deepseek_v4_indexer.h ==== */

#include <stddef.h>

/* amalgamated: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_layer.h */

typedef struct LmaDeepSeekV4Indexer LmaDeepSeekV4Indexer;

int lma_v4_indexer_create(LmaDeepSeekV4Indexer **state,
                           const LmaDeepSeekV4LayerWeights *weights,
                           const LmaDeepSeekV4Config *config,
                           int max_context, char *error, size_t error_size);
int lma_v4_indexer_bind_weights(LmaDeepSeekV4Indexer *state,
                                 const LmaDeepSeekV4LayerWeights *weights,
                                 char *error, size_t error_size);
void lma_v4_indexer_reset(LmaDeepSeekV4Indexer *state);
void lma_v4_indexer_destroy(LmaDeepSeekV4Indexer *state);

/* Updates the overlap compressor, then returns compressed-cache ordinals in
 * descending index score order. query_rank is the normalized q_lora vector. */
int lma_v4_indexer_step(LmaDeepSeekV4Indexer *state, int *indices,
                         int index_capacity, const float *query_rank,
                         const float *input, int position,
                         char *error, size_t error_size);
/* Like lma_v4_indexer_step, but consumes precomputed rows of the indexer
 * compressor's wkv/wgate projections for this position. */
int lma_v4_indexer_step_projected(LmaDeepSeekV4Indexer *state, int *indices,
                                   int index_capacity, const float *query_rank,
                                   const float *input, int position,
                                   const float *kv_proj, const float *gate_proj,
                                   char *error, size_t error_size);
/* Batched prefill split of the step: advance per token (in order), then
 * select for the whole chunk at once. */
int lma_v4_indexer_advance(LmaDeepSeekV4Indexer *state, const float *input,
                            int position, const float *kv_proj,
                            const float *gate_proj, char *error,
                            size_t error_size);
int lma_v4_indexer_select_batch(LmaDeepSeekV4Indexer *state, int *indices,
                                 int index_capacity, const float *query_ranks,
                                 const float *inputs, int start_position,
                                 int batch, const int *counts, int *selected,
                                 char *error, size_t error_size);
const float *lma_v4_indexer_compressed_values(
    const LmaDeepSeekV4Indexer *state);
int lma_v4_indexer_compressed_count(const LmaDeepSeekV4Indexer *state);
/* ==== end deepseek_v4_indexer.h ==== */

/* ==== begin deepseek_v4_indexer_snapshot.h ==== */

/* amalgamated: deepseek_v4_indexer.h */

typedef struct LmaV4IndexerSnapshot LmaV4IndexerSnapshot;

int lma_v4_indexer_snapshot_create(const LmaDeepSeekV4Indexer *state,
                                    LmaV4IndexerSnapshot **output);
int lma_v4_indexer_snapshot_restore(LmaDeepSeekV4Indexer *state,
                                     const LmaV4IndexerSnapshot *snapshot);
void lma_v4_indexer_snapshot_destroy(LmaV4IndexerSnapshot *snapshot);
int lma_v4_indexer_snapshot_write(const LmaV4IndexerSnapshot *snapshot,
                                   FILE *stream);
int lma_v4_indexer_snapshot_read(FILE *stream, LmaV4IndexerSnapshot **output);
/* ==== end deepseek_v4_indexer_snapshot.h ==== */

/* ==== begin deepseek_v4_expert.h ==== */

#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

int lma_v4_expert_forward_ref(float *output, const LmaExpertView *expert,
                               const float *input, float route_weight,
                               float swiglu_limit);

/* Batch-major routed-expert forward.  The row-major FP4 path streams each
 * matrix once for all items; unsupported packed layouts fall back to the
 * scalar entry point without changing its numerical contract. */
int lma_v4_expert_forward_batch_ref(float *outputs,
                                     const LmaExpertView *expert,
                                     const float *inputs,
                                     const float *route_weights,
                                     int batch, float swiglu_limit);

int lma_v4_shared_expert_forward_ref(float *output,
                                      const LmaTensorView *gate,
                                      const LmaTensorView *down,
                                      const LmaTensorView *up,
                                      const float *input,
                                      float swiglu_limit);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_expert.h ==== */

/* ==== begin deepseek_v4_expert_store.h ==== */

#include <stddef.h>
#include <stdint.h>

#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LmaDeepSeekV4ExpertStoreOptions {
    const char *model_dir;
    int layers;
    int experts_per_layer;
    uint64_t cache_bytes;
    /* Optional hot-pin policy (-1 / 0 => implementation default). */
    int pin_slots_per_layer;
    uint64_t repin_interval;
    /* Internal range executors must not benchmark arbitrary bytes from a
     * complete checkpoint while opening one layer slice. */
    int skip_mirror_setup;
} LmaDeepSeekV4ExpertStoreOptions;

int lma_deepseek_v4_expert_store_open(
    const LmaDeepSeekV4ExpertStoreOptions *options,
    LmaExpertStore **store,
    char *error,
    size_t error_size);

/* Plain SSD implementation underneath the optional hot-row/autopin wrapper.
 * Segment adapters use it so opening a layer range never warms experts from
 * layers outside that range.  It remains an internal symbol, not public ABI. */
int lma_deepseek_v4_expert_store_open_base(
    const LmaDeepSeekV4ExpertStoreOptions *options,
    LmaExpertStore **store,
    char *error,
    size_t error_size);

/* Batched CPU prefill only: `layer` borrows the complete expert-cache pool;
 * a negative value restores ordinary per-layer miss allocation for decode.
 * Alternative registered ExpertStore backends safely ignore the request. */
void lma_v4_expert_store_prefill_pool(LmaExpertStore *store, int layer);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_expert_store.h ==== */

/* ==== begin deepseek_v4_block.h ==== */

#include <stddef.h>

/* amalgamated: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_attention.h */
/* amalgamated: deepseek_v4_layer.h */
#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

int lma_v4_block_token_ref(float *output_hc,
                            const LmaDeepSeekV4LayerWeights *weights,
                            const LmaDeepSeekV4Config *config,
                            LmaExpertStore *experts,
                            const float *input_hc, int token, int position,
                            char *error, size_t error_size);
int lma_v4_block_window_token_ref(
    float *output_hc, LmaDeepSeekV4WindowAttentionState *attention,
    const LmaDeepSeekV4LayerWeights *weights,
    const LmaDeepSeekV4Config *config, LmaExpertStore *experts,
    const float *input_hc, int token, int position,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_block.h ==== */

/* ==== begin deepseek_v4_block_batch.h ==== */

/* amalgamated: deepseek_v4_attention.h */
/* amalgamated: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_layer.h */
#include "expert_store.h"

int lma_v4_block_window_batch_ref(
    float *outputs_hc, LmaDeepSeekV4WindowAttentionState *attention,
    const LmaDeepSeekV4LayerWeights *weights,
    const LmaDeepSeekV4Config *config, LmaExpertStore *experts,
    const float *inputs_hc, const int *tokens, int start_position, int batch,
    char *error, size_t error_size);
/* ==== end deepseek_v4_block_batch.h ==== */

/* ==== begin deepseek_v4_resource_plan.h ==== */

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t available_bytes;
    uint64_t user_limit_bytes;
    uint64_t maximum_layer_bytes;
    uint64_t runtime_other_bytes;
    uint64_t expert_record_bytes;
    int sparse_layers;
    int routed_topk;
    int experts_per_layer;
} LmaDeepSeekV4ResourceInputs;

typedef struct {
    uint64_t os_available_bytes;
    uint64_t planner_available_bytes;
    uint64_t system_reserve_bytes;
    uint64_t runtime_reserve_bytes;
    uint64_t minimum_expert_bytes;
    uint64_t expert_cache_bytes;
    uint64_t projected_bytes;
    int slots_per_layer;
} LmaDeepSeekV4ResourcePlan;

typedef struct {
    uint64_t available_bytes;
    uint64_t fixed_bytes;
    uint64_t dense_bytes;
    uint64_t minimum_expert_bytes;
} LmaDeepSeekV4ResidentTierInputs;

typedef struct {
    uint64_t dense_bytes;
    int dense_resident;
} LmaDeepSeekV4ResidentTierPlan;

uint64_t lma_v4_os_available_memory(void);
int lma_v4_resource_plan_compute(
    LmaDeepSeekV4ResourcePlan *plan,
    const LmaDeepSeekV4ResourceInputs *inputs,
    char *error, size_t error_size);
int lma_v4_resident_tier_plan(
    LmaDeepSeekV4ResidentTierPlan *plan,
    const LmaDeepSeekV4ResidentTierInputs *inputs,
    char *error, size_t error_size);
/* ==== end deepseek_v4_resource_plan.h ==== */

/* ==== begin deepseek_v4_head_cache.h ==== */

#include <stddef.h>
#include <stdint.h>

int lma_v4_head_cache_probe(const LmaSafetensorsIndex *index, uint64_t *bytes,
                             char *error, size_t error_size);
int lma_v4_head_cache_load(LmaV4Engine *engine,
                            const LmaSafetensorsIndex *index,
                            char *error, size_t error_size);
uint64_t lma_v4_head_cache_bytes(const LmaV4Engine *engine);
const void *lma_v4_head_cache_data(const LmaV4Engine *engine,
                                    int shard, uint64_t offset, size_t length);
/* ==== end deepseek_v4_head_cache.h ==== */


/* Runtime options live on LmaV4Engine. */
typedef struct {
    const char *target_model_dir;
    uint64_t memory_limit_bytes;
    int context_tokens;
    int dense_resident;
    uint64_t target_expert_cache_bytes;
    int pin_slots_per_layer;
    uint64_t repin_interval;
    uint64_t dspark_reserve_bytes;
} LmaDeepSeekV4RuntimeOptions;

enum { LMA_V4_RESIDENT_MAX_LAYERS = 128 };

/* engine open: seconds spent building target_index (printed by the auto
 * store planner as the v4_open line). Defined in the engine unit. */
extern double g_v4_open_index_seconds;

#ifdef LMA_V4_GPU_TIER
/* Provided by the LMA_V4_UNIT_GPU translation unit. Compiled in only on the
 * Windows CUDA build; every call site elsewhere is guarded by LMA_V4_GPU_TIER
 * so non-GPU objects never reference these symbols. */
int lma_v4_gpu_engine_open(LmaV4Engine *engine);
void lma_v4_gpu_engine_close(LmaV4Engine *engine);
int lma_v4_gpu_layer_upload(LmaV4Engine *engine, int layer,
                             LmaDeepSeekV4LayerWeights *weights);
int lma_v4_gpu_fp8_matvec(const LmaTensorView *w, float *output,
                           const float *input);
int lma_v4_gpu_fp8_matmul_batch(const LmaTensorView *w, float *outputs,
                                 const float *inputs, int batch);
int lma_v4_gpu_kv_ring_append(const LmaDeepSeekV4LayerWeights *weights,
                               const float *rows, int start_pos, int count,
                               int window, int dim);
int lma_v4_gpu_kv_comp_append(const LmaDeepSeekV4LayerWeights *weights,
                               const float *rows, int start_idx, int count,
                               int dim);
int lma_v4_gpu_kv_cache_sync(const LmaDeepSeekV4LayerWeights *weights,
                              const float *cpu_ring, int window, int head_dim,
                              int start_position, const float *compressed,
                              int comp_total);
void lma_v4_gpu_kv_cache_advance(const LmaDeepSeekV4LayerWeights *weights,
                                  const float *rows, int start_position,
                                  int count, int window, int head_dim,
                                  int comp_total);
void lma_v4_gpu_kv_cache_poison(const LmaDeepSeekV4LayerWeights *weights);
void lma_v4_gpu_kv_cache_invalidate_all(void);
int lma_v4_gpu_sparse_attention_batch_cached(
    const LmaDeepSeekV4LayerWeights *weights, float *attended,
    const float *q, const float *chunk, int chunk_start, const float *sinks,
    const int *meta, int abs_base, int comp_limit, int heads, int head_dim,
    int batch);
int lma_v4_gpu_fp8_ref_matmul(const LmaDeepSeekV4LayerWeights *weights,
                               const LmaTensorView *w, const float *x_qdq,
                               int tokens, float *y);
int lma_v4_gpu_indexer_score_batch(
    const LmaDeepSeekV4LayerWeights *weights, float *scores,
    const float *queries, const float *keys, const float *head_w,
    const int *counts, int tokens, int heads, int dim, int count);
int lma_v4_gpu_sparse_attention_batch_cached_idx(
    const LmaDeepSeekV4LayerWeights *weights, float *attended,
    const float *q, const float *chunk, int chunk_start, const float *sinks,
    const int *meta, const int *sel, int selstride, int abs_base,
    int comp_limit, int heads, int head_dim, int batch);
int lma_v4_gpu_moe_batch_wanted(void);
void lma_v4_gpu_moe_batch_release(void);
void lma_v4_gpu_moe_batch_hint(int total_fresh_tokens);
int lma_v4_gpu_moe_batch_union(float *outputs,
                                const LmaDeepSeekV4LayerWeights *weights,
                                const LmaDeepSeekV4Config *config,
                                LmaExpertStore *store,
                                const float *inputs, const int *tokens,
                                int batch);
int lma_v4_gpu_matvec_grouped(const LmaTensorView *w, float *output,
                               const float *input, int groups);
/* Batched GPU attention offloads for prefill (LMA_CUDA_ATTN_BATCH=1).
 * Every entry returns non-zero on any refusal so the caller can fall back to
 * the CPU reference for the whole chunk. */
int lma_v4_gpu_attn_batch_wanted(void);
/* Runs both bf16 projection matrices (wkv_key/wgate_key mirrors) over the
 * whole chunk: kv_proj/gate_proj receive [batch][rows-of-mirror]. */
int lma_v4_gpu_compressor_project_batch(
    const LmaDeepSeekV4LayerWeights *weights, const char *wkv_key,
    const char *wgate_key, int expected_rows, float *kv_proj,
    float *gate_proj, const float *inputs, int batch);
/* Sparse window attention over a linear KV slab; contract described at
 * dsv4_cuda_sparse_attn_batch. */
int lma_v4_gpu_sparse_attention_batch(
    const LmaDeepSeekV4LayerWeights *weights, float *attended,
    const float *q, const float *values, const float *sinks, const int *meta,
    int value_rows, int comp_base, int heads, int head_dim, int batch);
/* Grouped wo_a + wo_b over the whole chunk through the fp8-bf16 wo_a mirror.
 * q_width = heads*head_dim (context row), hidden = output row. */
int lma_v4_gpu_attention_wo_batch(
    const LmaDeepSeekV4LayerWeights *weights, float *outputs,
    const float *attended, int groups, int q_width, int hidden, int batch);
/* Batched mHC: whole-chunk normalized_hc_pre / lma_v4_hc_post through the
 * hc_<branch>_fn/scale/base and norm f32 mirrors (hc must be 4, hidden 4096).
 * posts is [batch][hc], combs [batch][hc*hc]; layouts match the CPU arrays. */
int lma_v4_gpu_mhc_pre_norm_batch(
    const LmaDeepSeekV4LayerWeights *weights, const char *branch,
    const char *norm_key, float *posts, float *combs, float *normalized,
    const float *inputs_hc, int hc, int hidden, int batch);
int lma_v4_gpu_mhc_post_batch(
    const LmaDeepSeekV4LayerWeights *weights, float *outputs_hc,
    const float *branch, const float *residual_hc, const float *posts,
    const float *combs, int hc, int hidden, int batch);
/* MoE router offload: mirrors the bf16 route contract of lma_v4_route_bf16.
 * Uses the resident layer's uploaded f32 gate/bias mirrors through
 * dsv4_cuda_route (which hardcodes 256 experts / top-k 6); any shape or
 * mirror mismatch returns non-zero so the caller falls back to the CPU route. */
int lma_v4_gpu_route(float *route_weights, int *indices, const float *input,
                      const LmaDeepSeekV4LayerWeights *weights,
                      const float *bias, const int *forced_indices,
                      int experts, int dimension, int topk, float route_scale);
/* Best-effort fp4 mirror attach for a routed expert. Returns 0 when all three
 * of view->gate/up/down now carry Dsv4CudaTensor* handles (cached on first
 * use), non-zero when the tier is inactive or the expert must stay on the CPU
 * fp4 path. Only block_rows==1 views are mirrored; rows16-packed slots are
 * skipped. */
int lma_v4_gpu_expert_attach(LmaExpertStore *store, LmaExpertView *view);
/* lookup-only twin: reports residency, never uploads (hybrid q* split) */
int lma_v4_gpu_expert_peek(LmaExpertStore *store, LmaExpertView *view);
/* DSV4_HYBRID=1 gate plus its cross-unit counters/EMAs: defined in the block
 * unit, read by the serve unit's per-turn stderr line. */
int lma_v4_hybrid_enabled(void);
/* async attach (enqueue only; drain closes the pipeline before compute) */
int lma_v4_gpu_expert_attach_async(LmaExpertStore *store,
                                    LmaExpertView *view);
int lma_v4_gpu_expert_drain(LmaExpertStore *store);
extern double g_v4_hyb_fill_bw, g_v4_hyb_host_bw;
extern unsigned long long g_v4_hyb_gpu_n, g_v4_hyb_cpu_n;
extern unsigned long long g_v4_hyb_upload_n, g_v4_hyb_skip_n;
/* Dspark (MTP) resident-expert mirrors: a separate bounded LRU so drafting can
 * never evict the target model's learned expert mirrors. ensure() lazily
 * allocates the cache on the first V4_MTP_GPU=1 draft; attach mirrors one
 * block_rows==1 fp4 expert view into it (returns non-zero to stay on CPU). */
int lma_v4_gpu_dspark_mirrors_ensure(LmaV4Engine *engine);
int lma_v4_gpu_dspark_expert_attach(void *cache, LmaExpertView *view);
#endif

struct LmaV4Engine {
    LmaDeepSeekV4Config config;
    LmaDeepSeekV4RuntimeOptions runtime;
    LmaSafetensorsIndex *target_index;
    LmaExpertStore *experts;
    LmaV4EngineMemorySummary summary;
    struct {
        unsigned char *data;
        uint64_t bytes;
        uint64_t offset;
        int shard;
    } head_cache;
    struct {
        LmaDeepSeekV4LayerWeights layers[LMA_V4_RESIDENT_MAX_LAYERS];
        unsigned char ready[LMA_V4_RESIDENT_MAX_LAYERS];
        const LmaSafetensorsIndex *index;
        uint64_t total_bytes;
    } dense_resident;
    /* Optional CUDA tier (compiled in only when the engine build defines
     * LMA_V4_GPU_TIER on Windows). enabled is 1 only after the loader resolved
     * lma_cuda_dsv4.dll and dsv4_cuda_init succeeded; matvec dispatch then
     * short-circuits the dense fp8 projections through backend_cuda_dsv4.cu. */
    struct {
        int enabled;
        int device;
        unsigned char layer_ready[LMA_V4_RESIDENT_MAX_LAYERS];
        long long uploaded_bytes;
        /* Optional opaque V4GpuExpertMirrorCache* for the dspark/MTP draft
         * experts (separate bounded LRU; see dspark_mirrors_ensure). NULL
         * unless V4_MTP_GPU=1 and the tier opened successfully. */
        void *dspark_mirrors;
    } gpu;
    struct {
        uint16_t *markov_w1;
        uint16_t *markov_w2;
        uint64_t bytes;
        int rank;
        int block_size;
        int stage;
        int enabled;
    } dspark;
    char *owned_target_model_dir;
    int owns_experts;
    int owns_index;
    int active_sessions; /* sessions created against this engine */
};

/* Session ownership helpers shared by production session code and tests. */
void lma_v4_engine_attach_session(LmaV4Engine *engine);
void lma_v4_engine_detach_session(LmaV4Engine *engine);

#include "tok.h"
#include "kv_prefix.h"

struct LmaV4Session {
    LmaV4Engine *engine;
    LmaDeepSeekV4Config config;
    LmaDeepSeekV4WindowAttentionState **attention;
    float *state;
    float *next;
    float *hidden;
    int *prompt_ids;
    int *generated;
    int max_prompt_tokens;
    int max_new_tokens_cap;
    int prompt_count;
    int generated_count;
    Tok tokenizer;
    int tokenizer_ready;
    char *text;
    int text_length;
    /* Token ids this session's attention state already holds, prompt and
     * generated alike, in the shared format glm52.c/inkling.c/kimi_k3.c use.
     * A follow-up request whose prompt starts with exactly these ids continues
     * from that position instead of re-prefilling it. */
    kv_prefix fed;
    int prefix_reused;   /* reuse length of the request in flight, for stats */
    uint64_t spec_attempts;
    uint64_t spec_drafted;
    uint64_t spec_accepted;
    int spec_disabled;
    /* Prompt-end capture for SUBMIT pin=1: the ids fed and the head scores
     * that predict the token after them. A later prompt that starts with
     * exactly these ids gets its first fresh token's predictor from here; that
     * token is the one a closed-set caller asks about (docs/brio.md). The
     * attention state itself goes to a v4_ckpt slot; this is the part the
     * snapshot does not hold. */
    int *pin_ids;
    int pin_len;
    float *pin_scores;
    /* Scratch for the numeric channel, one hidden row and one row of head
     * scores, allocated on first use and freed with the session so the many
     * early returns of generate() leave nothing behind. */
    float *echo_hidden;
    float *echo_scores;
};

/* RAM-tiered expert open used by lma_v4_engine_open (replaces ld --wrap).
 * `config` is the model geometry (== engine->config on the engine_open path);
 * it is forwarded by the backend registry so engine-less callers (the standalone
 * CLI) can use routed backends that only need geometry, not the engine. */
int lma_v4_expert_store_open_planned(
    LmaV4Engine *engine,
    const LmaDeepSeekV4Config *config,
    const LmaDeepSeekV4ExpertStoreOptions *options,
    LmaExpertStore **store,
    char *error,
    size_t error_size);

/* Internal accessors — not part of the experimental public API. */
LmaSafetensorsIndex *lma_v4_engine_target_index(LmaV4Engine *engine);
LmaExpertStore *lma_v4_engine_expert_store(LmaV4Engine *engine);

/* Head-cache aware safetensors read (engine NULL => plain lma_st_read_at). */
int lma_st_read_at_engine(LmaV4Engine *engine,
                           const LmaSafetensorsIndex *index, int shard,
                           uint64_t offset, size_t length, void *destination);

#ifdef LMA_V4_TEST_HOOKS
/*
 * Fault-injection / counters for ownership tests only.
 * Compile ownership objects with -DLMA_V4_TEST_HOOKS; production objects omit this.
 */
extern int lma_v4_test_fail_expert_store_open;
extern int lma_v4_test_skip_expert_store_open;
extern int lma_v4_test_closed_owned_index;
extern void (*lma_v4_test_expert_read_hook)(LmaExpertKey key);
extern void (*lma_v4_test_expert_wait_hook)(LmaExpertKey key);
extern uint64_t lma_v4_test_fp4_batch_calls;
extern uint64_t lma_v4_test_expert_victim_probes;
int lma_v4_test_expert_slot_index(LmaExpertStore *store, LmaExpertKey key);
void lma_v4_test_reset_direct_io_stats(void);
uint64_t lma_v4_test_direct_reads(void);
uint64_t lma_v4_test_direct_fallbacks(void);
/* Point missing O_DIRECT twins at a dup of the buffered fd so tests can
 * exercise the direct-window path on filesystems that refuse O_DIRECT. */
int lma_v4_test_force_streaming_direct(LmaExpertStore *store);

LmaV4Session *lma_v4_test_session_bare_create(LmaV4Engine *engine);
void lma_v4_test_session_bare_destroy(LmaV4Session *session);
#endif /* LMA_V4_TEST_HOOKS */

#endif /* LMA_DEEPSEEK_V4_INTERNAL_H */
