#ifndef LMA_EXPERT_STORE_REGISTRY_H
#define LMA_EXPERT_STORE_REGISTRY_H

/*
 * Pluggable expert-store backend registry.
 *
 * The DeepSeek-V4 engine opens its expert store through a backend selected by
 * the LMA_EXPERT_STORE environment variable (default "auto" = the built-in
 * on-disk/mmap store). A backend registers a name + an open function matching
 * LmaExpertStoreBackendOpenFn. Backends ship as separate object files and
 * register themselves at link time via a C constructor, so the engine source
 * carries no backend-specific branching — adding a backend (e.g. a networked
 * store) is a link-time concern, not an engine edit.
 *
 * The built-in "auto" backend (lma_v4_expert_store_open_planned) is registered
 * by this module's own constructor, so with LMA_EXPERT_STORE unset the engine
 * behaves exactly as before this seam existed.
 */

#include <stddef.h>
#include "expert_store.h"
#include "deepseek_v4.h" /* LmaDeepSeekV4Config (engine-less CLI path passes it) */

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations. The concrete structs live in deepseek_v4_internal.h;
 * the registry only forwards opaque pointers to the chosen backend, so a
 * backend object file that needs field access includes that header itself. */
typedef struct LmaV4Engine LmaV4Engine;
typedef struct LmaDeepSeekV4ExpertStoreOptions LmaDeepSeekV4ExpertStoreOptions;

/* Open a store for the given engine + config + options. `engine` carries the
 * runtime/planning state the built-in "auto" backend needs; `config` carries
 * the model geometry every backend needs and is the ONLY thing a backend
 * without an engine (e.g. a remote store opened from the standalone CLI
 * path, which has no LmaV4Engine) can rely on. `engine` may be NULL for
 * backends that only consume `config`; the "auto" backend requires a non-NULL
 * engine. Same return contract as lma_v4_expert_store_open_planned: zero on
 * success with *output written, non-zero on failure with a message. */
typedef int (*LmaExpertStoreBackendOpenFn)(
    LmaV4Engine *engine,
    const LmaDeepSeekV4Config *config,
    const LmaDeepSeekV4ExpertStoreOptions *options,
    LmaExpertStore **output,
    char *error, size_t error_size);

/* Register a backend under `name`. Intended to be called from a constructor in
 * the backend's object file (so registration happens at static-link time,
 * before main). Names match case-sensitively; a later registration with the
 * same name replaces the earlier (last-wins, letting a build deliberately
 * override "auto"). Returns 0 on success, -1 if name/open_fn is NULL or the
 * registry is full. */
int lma_expert_store_backend_register(const char *name,
                                       LmaExpertStoreBackendOpenFn open_fn);

/* Number of backends currently registered (mainly for tests/diagnostics). */
int lma_expert_store_backend_count(void);

/* Look up a backend's open function by name, or NULL if not registered. */
LmaExpertStoreBackendOpenFn
lma_expert_store_backend_lookup(const char *name);

/* Open a store via the backend selected by the LMA_EXPERT_STORE env var
 * (default "auto"). Returns zero on success; non-zero with an error message if
 * the selected backend is not registered or its open fails. This is what the
 * engine calls in place of a hardcoded open. */
int lma_expert_store_backend_open_selected(
    LmaV4Engine *engine,
    const LmaDeepSeekV4Config *config,
    const LmaDeepSeekV4ExpertStoreOptions *options,
    LmaExpertStore **output,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
#endif /* LMA_EXPERT_STORE_REGISTRY_H */