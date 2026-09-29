#ifndef LMA_SEGMENT_RUNTIME_H
#define LMA_SEGMENT_RUNTIME_H

/*
 * Public, engine-neutral layer-segment runtime.
 *
 * LM Atomic owns model math, weights, accelerators and sequence state.  A
 * distributed caller owns transport, placement, leases and network session
 * identifiers.  This API is the boundary between the two: adapters keep all
 * engine-specific objects opaque while callers can open a layer range, create
 * isolated sessions, run activations through it and stream snapshots.
 *
 * Version 1 is additive and unused by the existing CLI/serve paths.  A
 * consumer registers adapters explicitly during process initialization and
 * before the first lookup, after which the registry is read-only and safe for
 * concurrent lookup.  No compiler-specific constructor mechanism is needed.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LMA_SEGMENT_ABI_VERSION 1u
#define LMA_SEGMENT_ENGINE_ID_CAP 64u
#define LMA_SEGMENT_STATE_SCHEMA_CAP 128u
#define LMA_SEGMENT_NUMERIC_CLASS_CAP 96u

typedef struct LmaSegmentEngine LmaSegmentEngine;
typedef struct LmaSegmentSession LmaSegmentSession;

typedef enum {
    LMA_SEGMENT_DTYPE_INVALID = 0,
    LMA_SEGMENT_DTYPE_F32 = 1,
    LMA_SEGMENT_DTYPE_F16 = 2,
    LMA_SEGMENT_DTYPE_BF16 = 3,
} LmaSegmentDType;

enum {
    LMA_SEGMENT_CAP_TOKEN_IDS = UINT64_C(1) << 0,
    LMA_SEGMENT_CAP_SNAPSHOT = UINT64_C(1) << 1,
    LMA_SEGMENT_CAP_RANGE_NATIVE = UINT64_C(1) << 2,
    LMA_SEGMENT_CAP_MULTI_SESSION = UINT64_C(1) << 3,
    LMA_SEGMENT_CAP_CPU = UINT64_C(1) << 8,
    LMA_SEGMENT_CAP_CUDA = UINT64_C(1) << 9,
    LMA_SEGMENT_CAP_HIP = UINT64_C(1) << 10,
    LMA_SEGMENT_CAP_METAL = UINT64_C(1) << 11,
    LMA_SEGMENT_CAP_VULKAN = UINT64_C(1) << 12,
};

#define LMA_SEGMENT_CAP_BACKEND_MASK                                      \
    (LMA_SEGMENT_CAP_CPU | LMA_SEGMENT_CAP_CUDA | LMA_SEGMENT_CAP_HIP | \
     LMA_SEGMENT_CAP_METAL | LMA_SEGMENT_CAP_VULKAN)

/* Filled by the adapter during engine_open, after it has inspected the model.
 * Fixed-size identity fields make the record safe to copy across a local ABI
 * boundary. state_width is the activation width on the segment wire; it need
 * not equal the model hidden size (DeepSeek V4 mHC is one example). */
typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t flags;
    char engine_id[LMA_SEGMENT_ENGINE_ID_CAP];
    char state_schema[LMA_SEGMENT_STATE_SCHEMA_CAP];
    char numeric_class[LMA_SEGMENT_NUMERIC_CLASS_CAP];
    uint32_t state_dtype;
    uint32_t state_width;
    uint32_t max_batch_rows;
    uint32_t max_context_tokens;
    uint32_t num_layers;
    uint32_t reserved_u32[7];
    uint64_t reserved_u64[4];
} LmaSegmentCapabilities;

/* layer_begin is inclusive and layer_end is exclusive.  backend_mask == 0
 * lets the adapter select its normal local policy.  memory_limit_bytes == 0
 * uses the adapter's ordinary automatic budget. */
typedef struct {
    uint32_t struct_size;
    const char *model_dir;
    uint32_t layer_begin;
    uint32_t layer_end;
    uint32_t context_tokens;
    uint32_t reserved_u32;
    uint64_t memory_limit_bytes;
    uint64_t backend_mask;
    const void *resource_plan;
    size_t resource_plan_size;
} LmaSegmentEngineOptions;

typedef struct {
    uint32_t struct_size;
    uint32_t context_tokens;
    uint64_t memory_limit_bytes;
    uint64_t reserved_u64[4];
} LmaSegmentSessionOptions;

typedef int (*LmaSegmentCancelFn)(void *user_data);

/* input and output each hold rows * state_width values of the advertised
 * dtype.  token_ids is either NULL/0 or exactly rows entries.  Engines whose
 * capabilities include LMA_SEGMENT_CAP_TOKEN_IDS require the latter. */
typedef struct {
    uint32_t struct_size;
    uint32_t rows;
    uint64_t position;
    const int32_t *token_ids;
    size_t token_count;
    const void *input;
    size_t input_bytes;
    void *output;
    size_t output_bytes;
    LmaSegmentCancelFn should_cancel;
    void *cancel_user_data;
    uint64_t reserved_u64[4];
} LmaSegmentRunRequest;

/* Snapshot formats are private to the adapter and named by state_schema.
 * Streaming callbacks avoid a second full-state allocation in LM Atomic or in
 * a network daemon.  A callback returns zero on success, non-zero to abort. */
typedef int (*LmaSegmentWriteFn)(void *user_data, const void *data, size_t size);
typedef int (*LmaSegmentReadFn)(void *user_data, void *data, size_t size);

typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    const char *engine_id;

    int (*engine_open)(void **engine_impl,
                       LmaSegmentCapabilities *capabilities,
                       const LmaSegmentEngineOptions *options,
                       char *error, size_t error_size);
    void (*engine_destroy)(void *engine_impl);
    int (*session_create)(void *engine_impl, void **session_impl,
                          const LmaSegmentSessionOptions *options,
                          char *error, size_t error_size);
    void (*session_destroy)(void *session_impl);
    int (*session_run)(void *session_impl,
                       const LmaSegmentRunRequest *request,
                       char *error, size_t error_size);
    int (*session_snapshot)(void *session_impl, LmaSegmentWriteFn write_fn,
                            void *write_user_data,
                            char *error, size_t error_size);
    int (*session_restore)(void *session_impl, LmaSegmentReadFn read_fn,
                           void *read_user_data,
                           char *error, size_t error_size);

    void (*reserved_fn[8])(void);
} LmaSegmentAdapter;

/* Registering is an initialization-time operation. Duplicate engine IDs and
 * malformed/currently incompatible adapters are rejected. Complete all
 * registration before lookups or execution begin. */
int lma_segment_adapter_register(const LmaSegmentAdapter *adapter);
const LmaSegmentAdapter *lma_segment_adapter_lookup(const char *engine_id);
int lma_segment_adapter_count(void);

int lma_segment_engine_open(const char *engine_id,
                             const LmaSegmentEngineOptions *options,
                             LmaSegmentEngine **engine,
                             char *error, size_t error_size);
/* The caller sets struct_size to its allocation size. Bytes beyond the ABI
 * version known by this runtime are zeroed for forward-compatible callers. */
int lma_segment_engine_capabilities(const LmaSegmentEngine *engine,
                                     LmaSegmentCapabilities *capabilities,
                                     char *error, size_t error_size);
/* Refuses to close an engine while sessions are alive. */
int lma_segment_engine_close(LmaSegmentEngine *engine,
                              char *error, size_t error_size);

int lma_segment_session_create(LmaSegmentEngine *engine,
                                const LmaSegmentSessionOptions *options,
                                LmaSegmentSession **session,
                                char *error, size_t error_size);
void lma_segment_session_destroy(LmaSegmentSession *session);
int lma_segment_run(LmaSegmentSession *session,
                     const LmaSegmentRunRequest *request,
                     char *error, size_t error_size);
int lma_segment_snapshot(LmaSegmentSession *session,
                          LmaSegmentWriteFn write_fn, void *write_user_data,
                          char *error, size_t error_size);
int lma_segment_restore(LmaSegmentSession *session,
                         LmaSegmentReadFn read_fn, void *read_user_data,
                         char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* LMA_SEGMENT_RUNTIME_H */
