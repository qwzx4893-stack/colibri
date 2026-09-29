#ifndef LMA_SEGMENT_ADAPTER_INTERNAL_H
#define LMA_SEGMENT_ADAPTER_INTERNAL_H

/* Shared implementation details for the engine-owned Segment adapters.
 * This is not a public ABI.  Snapshot payloads remain adapter-private and are
 * accepted only after the engine/range/context identity has been checked. */

#include "segment_runtime.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LMA_SEGMENT_SNAPSHOT_VERSION 1u
#define LMA_SEGMENT_HASH_INIT UINT64_C(1469598103934665603)

typedef struct {
    unsigned char magic[8];
    uint32_t format_version;
    uint32_t header_size;
    char engine_id[LMA_SEGMENT_ENGINE_ID_CAP];
    uint32_t layer_begin;
    uint32_t layer_end;
    uint32_t context_tokens;
    uint32_t position;
    uint64_t payload_bytes;
    uint64_t payload_hash;
    uint64_t reserved[4];
} LmaSegmentSnapshotHeader;

typedef struct {
    void *data;
    size_t size;
} LmaSegmentStateSpan;

static int lma_segment_adapter_error(char *error, size_t error_size,
                                      const char *format, ...) {
    if (error && error_size) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, error_size, format, args);
        va_end(args);
    }
    return -1;
}

static int lma_segment_size_mul(size_t left, size_t right, size_t *result) {
    if (!result || (right && left > SIZE_MAX / right)) return -1;
    *result = left * right;
    return 0;
}

static uint64_t lma_segment_hash_update(uint64_t hash, const void *data,
                                         size_t size) {
    const unsigned char *bytes = (const unsigned char *)data;
    if (!hash) hash = LMA_SEGMENT_HASH_INIT;
    for (size_t index = 0; index < size; index++) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int lma_segment_stream_write(LmaSegmentWriteFn write_fn,
                                     void *user_data, const void *data,
                                     size_t size, char *error,
                                     size_t error_size) {
    if (size && write_fn(user_data, data, size))
        return lma_segment_adapter_error(error, error_size,
                                          "segment snapshot write failed");
    return 0;
}

static int lma_segment_stream_read(LmaSegmentReadFn read_fn,
                                    void *user_data, void *data, size_t size,
                                    char *error, size_t error_size) {
    if (size && read_fn(user_data, data, size))
        return lma_segment_adapter_error(error, error_size,
                                          "segment snapshot read failed");
    return 0;
}

static void lma_segment_snapshot_header_init(
    LmaSegmentSnapshotHeader *header, const char *engine_id,
    uint32_t layer_begin, uint32_t layer_end, uint32_t context_tokens,
    uint32_t position, uint64_t payload_bytes, uint64_t payload_hash) {
    static const unsigned char magic[8] = {'C','O','L','I','S','E','G','1'};
    memset(header, 0, sizeof(*header));
    memcpy(header->magic, magic, sizeof(magic));
    header->format_version = LMA_SEGMENT_SNAPSHOT_VERSION;
    header->header_size = (uint32_t)sizeof(*header);
    snprintf(header->engine_id, sizeof(header->engine_id), "%s", engine_id);
    header->layer_begin = layer_begin;
    header->layer_end = layer_end;
    header->context_tokens = context_tokens;
    header->position = position;
    header->payload_bytes = payload_bytes;
    header->payload_hash = payload_hash;
}

static int lma_segment_snapshot_header_valid(
    const LmaSegmentSnapshotHeader *header, const char *engine_id,
    uint32_t layer_begin, uint32_t layer_end, uint32_t context_tokens,
    uint64_t payload_bytes, char *error, size_t error_size) {
    static const unsigned char magic[8] = {'C','O','L','I','S','E','G','1'};
    if (!header || memcmp(header->magic, magic, sizeof(magic)) ||
        header->format_version != LMA_SEGMENT_SNAPSHOT_VERSION ||
        header->header_size != sizeof(*header) ||
        strncmp(header->engine_id, engine_id,
                sizeof(header->engine_id)) ||
        header->layer_begin != layer_begin || header->layer_end != layer_end ||
        header->context_tokens != context_tokens ||
        header->position > context_tokens ||
        header->payload_bytes != payload_bytes)
        return lma_segment_adapter_error(
            error, error_size, "segment snapshot identity is incompatible");
    return 0;
}

static void lma_segment_capability_string(char *output, size_t capacity,
                                           const char *value) {
    if (!output || !capacity) return;
    snprintf(output, capacity, "%s", value ? value : "");
}

static int lma_segment_spans_size(const LmaSegmentStateSpan *spans,
                                   size_t count, size_t *bytes) {
    if (!bytes || (count && !spans)) return -1;
    size_t total = 0;
    for (size_t index = 0; index < count; index++) {
        if (spans[index].size > SIZE_MAX - total) return -1;
        total += spans[index].size;
    }
    *bytes = total;
    return 0;
}

static uint64_t lma_segment_spans_hash(const LmaSegmentStateSpan *spans,
                                        size_t count) {
    uint64_t hash = LMA_SEGMENT_HASH_INIT;
    for (size_t index = 0; index < count; index++)
        hash = lma_segment_hash_update(hash, spans[index].data,
                                        spans[index].size);
    return hash;
}

static int lma_segment_spans_write(const LmaSegmentStateSpan *spans,
                                    size_t count, LmaSegmentWriteFn write_fn,
                                    void *user_data, char *error,
                                    size_t error_size) {
    for (size_t index = 0; index < count; index++)
        if (lma_segment_stream_write(write_fn, user_data, spans[index].data,
                                      spans[index].size, error, error_size))
            return -1;
    return 0;
}

typedef int (*LmaSegmentPayloadValidateFn)(
    const LmaSegmentStateSpan *spans, size_t count,
    const unsigned char *payload, size_t payload_bytes,
    void *user_data, char *error, size_t error_size);

/* Read into a compact temporary buffer, validate the complete payload, then
 * commit to live state.  A failed or truncated restore therefore leaves the
 * conversation byte-for-byte unchanged.  The optional validator lets an
 * adapter reject semantically impossible state while it is still staged. */
static int lma_segment_spans_restore_checked(
    const LmaSegmentStateSpan *spans, size_t count, uint64_t expected_hash,
    LmaSegmentReadFn read_fn, void *read_user_data,
    LmaSegmentPayloadValidateFn validate_fn, void *validate_user_data,
    char *error, size_t error_size) {
    size_t bytes;
    if (lma_segment_spans_size(spans, count, &bytes))
        return lma_segment_adapter_error(error, error_size,
                                           "segment snapshot size overflow");
    unsigned char *payload = bytes ? (unsigned char *)malloc(bytes) : NULL;
    if (bytes && !payload)
        return lma_segment_adapter_error(error, error_size,
                                           "out of memory restoring segment state");
    if (lma_segment_stream_read(read_fn, read_user_data, payload, bytes,
                                 error, error_size)) {
        free(payload);
        return -1;
    }
    if (lma_segment_hash_update(LMA_SEGMENT_HASH_INIT, payload, bytes) !=
        expected_hash) {
        free(payload);
        return lma_segment_adapter_error(error, error_size,
                                           "segment snapshot checksum mismatch");
    }
    if (validate_fn && validate_fn(spans, count, payload, bytes,
                                   validate_user_data, error, error_size)) {
        free(payload);
        return -1;
    }
    const unsigned char *cursor = payload;
    for (size_t index = 0; index < count; index++) {
        if (spans[index].size) {
            memcpy(spans[index].data, cursor, spans[index].size);
            cursor += spans[index].size;
        }
    }
    free(payload);
    return 0;
}

static int lma_segment_spans_restore(const LmaSegmentStateSpan *spans,
                                      size_t count, uint64_t expected_hash,
                                      LmaSegmentReadFn read_fn,
                                      void *user_data, char *error,
                                      size_t error_size) {
    return lma_segment_spans_restore_checked(
        spans, count, expected_hash, read_fn, user_data, NULL, NULL,
        error, error_size);
}

#endif /* LMA_SEGMENT_ADAPTER_INTERNAL_H */
