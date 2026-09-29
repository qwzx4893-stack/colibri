#ifndef LMA_TENSOR_H
#define LMA_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Storage/execution formats understood by a model backend.  A view describes
 * bytes owned by a model or ExpertStore; it never owns or frees those bytes. */
typedef enum {
    LMA_TENSOR_F32 = 0,
    LMA_TENSOR_Q8_ROW,
    LMA_TENSOR_Q4_ROW,
    LMA_TENSOR_Q2_ROW,
    LMA_TENSOR_FP8_E4M3_BLOCK,
    LMA_TENSOR_FP4_NATIVE_BLOCK,
    LMA_TENSOR_INT8_BLOCK,
    LMA_TENSOR_INT4_BLOCK
} LmaTensorFormat;

typedef enum {
    LMA_SCALE_NONE = 0,
    LMA_SCALE_F32,
    LMA_SCALE_UE8M0
} LmaScaleFormat;

typedef struct {
    LmaTensorFormat format;
    LmaScaleFormat scale_format;
    const void *data;
    const void *scales;
    size_t data_bytes;
    size_t scale_bytes;
    int64_t rows;
    int64_t columns;
    uint32_t block_rows;
    uint32_t block_columns;
    /* Optional backend-resident mirror of this weight (e.g. a Dsv4CudaTensor*).
     * NULL on the CPU-only paths; owned and freed by the backend tier, never by
     * the view. */
    void *gpu;
} LmaTensorView;

#ifdef __cplusplus
}
#endif

#endif
