/*
 * Pluggable expert-store backend registry — implementation.
 *
 * See expert_store_registry.h for the design. Registration happens from C
 * constructors at static-link time (before main), so the table is read-only
 * once the engine runs and no locking is needed.
 */

#include "expert_store_registry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Keep the public registry symbols external even under -flto: backends live
 * in separately-compiled object files (e.g. a custom backend linked in later)
 * and call register() from their own constructors, so these symbols must
 * survive LTO inlining. Without this, gcc drops them when the only caller in
 * the current link is inside the same LTO set. */
#if defined(__has_attribute)
#  if __has_attribute(externally_visible)
#    define LMA_ESR_EXTERNALLY_VISIBLE __attribute__((externally_visible))
#  else
#    define LMA_ESR_EXTERNALLY_VISIBLE
#  endif
#elif defined(__GNUC__)
#  define LMA_ESR_EXTERNALLY_VISIBLE __attribute__((externally_visible))
#else
#  define LMA_ESR_EXTERNALLY_VISIBLE
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define LMA_ESR_EXPORT LMA_ESR_EXTERNALLY_VISIBLE __attribute__((used))
#else
#  define LMA_ESR_EXPORT
#endif

/* The built-in on-disk/mmap backend, defined in the LMA_V4_UNIT_EXPERT_STORE_AUTO
 * amalgamation unit of deepseek_v4.c. Declared here (not via its header, which
 * is a large amalgamated translation unit) so this module stays standalone. */
int lma_v4_expert_store_open_planned(
    LmaV4Engine *engine,
    const LmaDeepSeekV4Config *config,
    const LmaDeepSeekV4ExpertStoreOptions *options,
    LmaExpertStore **output,
    char *error, size_t error_size);

#define LMA_EXPERT_STORE_MAX_BACKENDS 8

static struct {
    const char *name;
    LmaExpertStoreBackendOpenFn open_fn;
} g_backends[LMA_EXPERT_STORE_MAX_BACKENDS];
static int g_backend_count;

LMA_ESR_EXPORT
int lma_expert_store_backend_register(const char *name,
                                       LmaExpertStoreBackendOpenFn open_fn) {
    if (!name || !open_fn) return -1;
    for (int i = 0; i < g_backend_count; i++) {
        if (strcmp(g_backends[i].name, name) == 0) {
            g_backends[i].open_fn = open_fn; /* last-wins override */
            return 0;
        }
    }
    if (g_backend_count >= LMA_EXPERT_STORE_MAX_BACKENDS) return -1;
    g_backends[g_backend_count].name = name;
    g_backends[g_backend_count].open_fn = open_fn;
    g_backend_count++;
    return 0;
}

LMA_ESR_EXPORT
int lma_expert_store_backend_count(void) {
    return g_backend_count;
}

LMA_ESR_EXPORT
LmaExpertStoreBackendOpenFn
lma_expert_store_backend_lookup(const char *name) {
    if (!name) return NULL;
    for (int i = 0; i < g_backend_count; i++)
        if (strcmp(g_backends[i].name, name) == 0)
            return g_backends[i].open_fn;
    return NULL;
}

LMA_ESR_EXPORT
int lma_expert_store_backend_open_selected(
    LmaV4Engine *engine,
    const LmaDeepSeekV4Config *config,
    const LmaDeepSeekV4ExpertStoreOptions *options,
    LmaExpertStore **output,
    char *error, size_t error_size) {
    const char *name = getenv("LMA_EXPERT_STORE");
    if (!name || !*name) name = "auto";
    LmaExpertStoreBackendOpenFn fn = lma_expert_store_backend_lookup(name);
    if (!fn) {
        if (error && error_size)
            snprintf(error, error_size,
                     "expert store backend '%s' is not registered "
                     "(set LMA_EXPERT_STORE to a linked backend; "
                     "default is 'auto')",
                     name);
        return -1;
    }
    return fn(engine, config, options, output, error, error_size);
}

/* Register the built-in on-disk/mmap backend at static-link time so the
 * default (LMA_EXPERT_STORE unset) path is unchanged from before this seam. */
__attribute__((constructor))
static void lma_register_auto_backend(void) {
    lma_expert_store_backend_register("auto",
                                       lma_v4_expert_store_open_planned);
}