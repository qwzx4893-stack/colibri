#ifndef LMA_EDGE_ADAPTER_INTERNAL_H
#define LMA_EDGE_ADAPTER_INTERNAL_H

/* Engine-private helpers shared by the built-in adapters. */

#include "edge_runtime.h"

#include <stdio.h>

static int lma_edge_adapter_error(char *error, size_t error_size,
                                   const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
    return -1;
}

static void lma_edge_capability_string(char *output, size_t output_size,
                                        const char *value) {
    if (!output || !output_size) return;
    snprintf(output, output_size, "%s", value ? value : "");
}

static int lma_edge_argmax(const float *values, uint32_t count,
                            int32_t *token, float *score) {
    if (!values || !count || !token) return -1;
    uint32_t best = 0;
    for (uint32_t item = 1; item < count; item++)
        if (values[item] > values[best]) best = item;
    *token = (int32_t)best;
    if (score) *score = values[best];
    return 0;
}

#endif /* LMA_EDGE_ADAPTER_INTERNAL_H */
