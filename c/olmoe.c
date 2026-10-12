/* Motore di inferenza OLMoE in C puro, con EXPERT-STREAMING dal disco.
 * Porting del motore Python (engine.py). Obiettivo Stadio A: produrre gli STESSI
 * token id del riferimento (ref.json) -> valida il core prima di scalare a GLM-5.2.
 *
 * Densa (embed, attn, router, norme, lm_head) residente in RAM (float32).
 * Expert letti dal disco on-demand via pread, cache LRU per-layer; le pagine
 * restano nel page cache cosi' i miss LRU non tornano su disco (EXPERT_DROP=1
 * ripristina fadvise(DONTNEED) per macchine con poca RAM).
 * Matmul multi-thread con OpenMP (niente BLAS).
 *
 * ENV VARS:
 *   PILOT=0/1/2/3 : 0=no prefetch, 1=1-layer lookahead, 2=2-layer, 3=3-layer lookahead
 *   HOT=N         : pin top-N hot experts per layer permanently (never evict)
 *   WARMUP=N      : tokens before hot pinning activates (default 5)
 *   WIDE=N        : prefetch top-K*N candidates (default 1, try 2 or 3)
 *   SMOOTH=F      : EMA coefficient for routing momentum (default 0.3, range 0.0-0.95)
 *   CONF_LIMIT=F  : cumulative gate probability threshold for prefetch cutoff (default 0.92)
 *   PILOT_EVICT_GUARD=0/1 : 1=enable LFRU prefetch eviction guard (default), 0=disable
 *   EXPERT_DROP=0/1: 1=fadvise(DONTNEED) after each expert read (old behaviour,
 *                    for RAM-tight boxes); 0=keep pages cached (default)
 *   ROUTE_TRACE=<path>: log every routing decision (one line per moe call,
 *                    position and layer: "<call> <row> <layer> <id>:<gate> ...")
 *                    for offline analysis — tools/route_pairs.py,
 *                    tools/route_coupling_report.py, tools/residency_sim.py.
 *                    Measurement only: it cannot change which experts run.
 *   (expert queue is sorted by eid for SSD read locality)
 *
 * LM ATOMIC RUNTIME (real, wired, decode-path additions — Exact Mode: none of
 * this changes which experts are routed or their gate weights; it only changes
 * WHEN and WHERE their weights are fetched from disk, which is why the ref.json
 * token-exact tests and greedy-decode determinism are unaffected):
 *   PREDICT_BETA=F   : weight (0.0-1.0) given to the learned co-activation
 *                    predictor vs. the original gate-logit softmax when
 *                    choosing PILOT>=2 lookahead candidates (default 0.4,
 *                    0.0 = old gate-only behaviour). See coact2/coact3 below.
 *   PILOT_WORKERS=N  : number of concurrent prefetch worker threads draining
 *                    the shared pilot queue (default 2; was hardcoded to 1).
 *                    Lets several speculative NVMe reads be in flight at once,
 *                    matching the measured multi-thread NVMe queue-depth
 *                    advantage (lma_hwprof: qd8 beats qd1 by ~1.4-4.6x on
 *                    this class of drive) instead of serializing them.
 *   SCHED_COST=0/1   : 1 (default) = evict by a cost-aware score (frequency +
 *                    recency + reuse likelihood from the co-activation table),
 *                    0 = the original pure-recency LRU. Applies to both the
 *                    demand-miss path (expert_get) and the pilot path
 *                    (pilot_realload), replacing two independent LRU scans
 *                    with one scorer.
 *   HOT_REEVAL_TOK=N : re-evaluate the hot-pinned working set every N decode
 *                    tokens using a decayed (not lifetime-total) frequency
 *                    window, so it can un-pin/re-pin if the session's topic
 *                    drifts (default 0 = old one-shot-after-WARMUP behaviour,
 *                    kept as the default so HOT=N alone reproduces prior runs
 *                    exactly; set e.g. 200 to enable adaptation).
 *   LMA_BYTES_LOG=1  : report external bytes moved (NVMe reads) per generated
 *                    token to stderr at the end of a CLI generate/PPL run.
 *   LMA_GIANT_STICK=F: Physical Giant Emulator ONLY (default 0, no effect on
 *                    any real model/checkpoint). See g_giant_stick's own
 *                    comment near its declaration.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#ifndef _WIN32
#include <dlfcn.h>
#endif
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <unistd.h>
#endif
#include "cli_args.h"
#include "st.h"
#ifdef _OPENMP
#include <omp.h>   /* omp_set_num_threads/omp_get_max_threads per omp_tune.h */
#endif
#include "omp_tune.h"
#include "route_trace.h"                    /* shared routing telemetry (#700) */
#include "moe_router.h"
#include "kv_prefix.h"
#include "pin_pool.h"                       /* piu scatti annidati */   /* riuso del prefisso tra turni (shared) */
#include "serve_codec.h"
#include "serve_budget.h"
#ifdef LMA_SEGMENT_ADAPTER
#include "segment_runtime.h"
#include "segment_adapters.h"
#include "segment_adapter_internal.h"
#endif
#ifdef LMA_EDGE_ADAPTER
#include "edge_runtime.h"
#include "edge_adapters.h"
#include "tok.h"
#include "edge_tok_internal.h"
#endif

#ifdef _WIN32
#include <windows.h>
#define sleep_ms(ms) Sleep(ms)
#else
#define sleep_ms(ms) usleep((ms) * 1000)
#endif



/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim;
    int n_experts, topk, inter, vocab;
    float theta, eps; int norm_topk;
    int stop_ids[8], n_stop;   /* unused (no model-specific hardcoded stops) — chat mode's
                                 * stop set comes entirely from the tokenizer's own special-
                                 * token flags via sample.h's stops_arm_tok */
} Cfg;

/* ---------- pesi densi per-layer ---------- */
typedef struct {
    float *in_ln, *post_ln, *q, *k, *v, *o, *qn, *kn, *gate;
    uint16_t *q_h, *k_h, *v_h, *o_h;   /* FP16-resident twins of q/k/v/o (LMA_DENSE_F16); when set the FP32 pointer is NULL */
} Layer;

/* ---------- cache LRU degli expert (pesi QUANTIZZATI) ----------
 * Ogni weight [out,in] tenuto come int8 (per-riga) + scala float per riga.
 * Cosi' la RAM-cache scende da 4 byte/param (f32) a 1 byte/param: e' il
 * meccanismo che fa stare GLM-5.2 nei 15 GB. dequant-on-use nel matmul. */
/* pinned=1 means this slot is strongly preferred to keep (hot expert); it will
 * not be evicted during normal LRU eviction, but may be displaced under extreme
 * cache pressure when all slots are pinned or in-flight. */
typedef struct { int eid; int pinned; int8_t *g, *u, *d; float *gs, *us, *ds; uint64_t used; } Slot;
typedef struct {
    Slot *slots;
    int *slot_by_expert;                  /* expert id -> resident slot, -1 if absent */
    int n, cap;
} LCache;

typedef struct {
    Cfg c;
    shards S;
    int quant_bits;
    float *embed, *lm_head, *final_norm;
    uint16_t *lm_head_h;    /* FP16-resident lm_head (LMA_DENSE_F16); lm_head is NULL when set */
    Layer *L;
    LCache *cache;          /* [n_layers] */
    uint64_t clock, hits, miss;
    /* Nanoseconds spent inside expert reads, summed across threads. The reads
     * run unlocked and in parallel, so this is an atomic counter rather than a
     * plain double: a per-turn delta of it is what the PROF line reports. */
    uint64_t disk_ns;
    float **K, **V; int kv_len, max_t;
    /* What the cached keys and values were built from, so a serve turn that
     * resends the transcript prefills only the new tail. Recorded where the
     * tokens are fed (see kv_prefix.h), never derived from a counter. */
    kv_prefix kvp;
    double dense_load_s;
    /* IMPROVEMENT 2: expert frequency heatmap */
    uint32_t **freq;                   /* per-layer expert counts, owned by route_trace.h */
    uint8_t **ehit;                    /* experts routed this turn, for HITS (dashboard Brain) */
    int freq_token_count, hot_pinned, hot_n, warmup_tokens;
    int token_count;
    /* PREDICTION IMPROVEMENT A: per-layer EMA of gate logits across tokens.
     * momentum_logits[l*E .. (l+1)*E-1] = EMA of gate outputs for layer l.
     * Used exclusively by the PILOT prefetcher to stabilise routing predictions
     * across tokens; does NOT affect actual MoE routing (pr is unchanged). */
    float *momentum_logits; /* [n_layers * n_experts], EMA of gate logits */
    float pilot_smooth;     /* SMOOTH env: EMA coefficient 0.0-0.9 (default 0.3) */
    uint8_t *is_pinned;     /* [n_layers * n_experts], 1 if expert is globally pinned */
    uint8_t *is_queued;     /* [n_layers * n_experts], 1 if expert is currently in the prefetch queue */
    float pilot_conf_limit; /* CONF_LIMIT env: cumulative gate probability threshold (e.g. 0.92) */
    uint64_t *last_access;  /* [n_layers * n_experts], clock time when expert was last accessed */

    /* LM ATOMIC: external bytes actually moved from disk (weights + scales),
     * the central "External Bytes Moved / Generated Token" metric. Same
     * atomic-counter reasoning as disk_ns: reads run unlocked and parallel. */
    uint64_t disk_bytes;

    /* LM ATOMIC Predictive Expert Engine: online cross-layer co-activation
     * counts, trained ONLY from real (post-router) selections, S==1 decode
     * only (prefill has no lookahead to speed up). coact2[l*E*E + a*E + b] =
     * how many times real layer l picked expert a and real layer l+2 picked
     * expert b (coact3: layer l -> l+3). Used to blend with pilot_prefetch's
     * gate-logit score for deeper (PILOT>=2) lookahead, where the existing
     * gate-on-stale-x approximation is weakest. Depth-1 lookahead is left on
     * the original gate-only path: at that call site the source layer for a
     * 2-hop transition (i-1) is already the freshest coact evidence available
     * (see pilot_prefetch), so the same coact2 table doubles as the depth-1
     * signal via gap=2 from i-1, with no separate coact1 table needed. */
    uint32_t *coact2, *coact3;      /* [n_layers * n_experts * n_experts] each */
    float predict_beta;             /* PREDICT_BETA env, default 0.4 */
    /* Rolling window of the last 3 layers' REAL (post-router) selections,
     * S==1 only, so a later layer's prediction can condition on an earlier
     * layer's actual (not speculative) pick. Slot reused round-robin. */
    int win_layer[3], win_valid[3], win_k[3];
    int win_idx[3][16];

    /* LM ATOMIC Multi-Tier (NVMe/RAM) Deadline Scheduler: SCHED_COST env,
     * default on. Reuses lfru_score's frequency+recency shape but is now the
     * PRIMARY eviction scorer for both expert_get's demand-miss path and
     * pilot_realload's speculative path, not just pilot's guard. */
    int sched_cost;

    /* LM ATOMIC Session-Adaptive Working Set: HOT_REEVAL_TOK env. When >0,
     * pin_hot_experts_adaptive() re-derives the hot set every N tokens from a
     * decayed window instead of running once and freezing forever. */
    int hot_reeval_tok;
    uint32_t **freq_window;         /* per-layer decayed usage window, same shape as freq */

    /* LM ATOMIC predictor precision/recall telemetry: pf_issued = speculative
     * loads actually performed (not dropped by the LFRU guard or a duplicate
     * check); pf_used = how many of those were consumed by a real expert_get
     * hit before eviction — i.e. the prediction paid for itself. Mirrors the
     * Digital Twin's prefetch_useful_frac so the two are comparable. */
    uint64_t pf_issued, pf_used;
    uint8_t *pf_flag;               /* [n_layers * n_experts], 1 = resident copy came from a pilot load */

    /* LM ATOMIC GPU decode path (olmoe_cuda.cu, loaded from lma_olmoe_cuda.dll). */
    int gpu_on;                      /* 1 = S==1 decode runs on the GPU */
    int gpu_max_t;                   /* device KV capacity in positions; beyond it decode falls back to the CPU */
    int gpu_kv_len;                  /* device KV holds positions [0, gpu_kv_len) */
    int gpu_n_slots, gpu_all_resident;
    int *gpu_slot_of;                /* [L*E] host mirror of the device (layer,expert)->slot table, -1 = absent */
    shards S4; int s4_open;          /* optional int4 overlay (LMA_EXPERT_OVERLAY): compiled w4/s4 expert tensors */
    unsigned char gpu_fmt[128];       /* per-layer expert storage: 0 = int8 (Exact), 1 = int4 group-32 (High-Fidelity, lossy) */
    int gpu_np[2];                   /* slots per pool (0 = int8 pool, 1 = int4 pool) */
    int *gpu_owner[2];               /* [pool][slot] layer*E+expert, -1 = free */
    uint64_t *gpu_use[2], gpu_clock;
    float *gpu_kv_stage;             /* [L][2][D] */
    uint64_t gpu_tokens, gpu_miss, gpu_fetch_syncs;
    double gpu_s;                    /* wall seconds spent in gpu_decode */
    float gpu_max_dlogit; uint64_t gpu_verify_n, gpu_verify_agree;   /* LMA_GPU_VERIFY statistics */
    uint64_t gpu_pf_rows, gpu_pf_calls; double gpu_pf_s;             /* GPU prefill: rows, calls, wall seconds */
    float gpu_pf_max_dlogit; uint64_t gpu_pf_verify_n, gpu_pf_verify_agree;
    int gpu_win_n; double gpu_win_s; uint64_t gpu_win_miss;          /* circuit-breaker health window (32 decode tokens) */
    double gpu_peak_d2d_gbs, gpu_peak_h2d_gbs, gpu_bytes_tok;         /* measured ceilings; weight bytes read per token */
    char gpu_trip_reason[96];
    /* LM ATOMIC Physical Giant Emulator ONLY (see g_giant_stick above). */
    int *giant_prev_idx;             /* [n_layers * topk] */
    uint8_t *giant_prev_valid;       /* [n_layers] */
    uint64_t giant_rng;
} Model;

static pthread_mutex_t g_pilot_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_pilot_cv = PTHREAD_COND_INITIALIZER; /* broadcast on every publish */
static struct { int l, e; } pilot_q[4096];
static volatile unsigned pilot_r = 0, pilot_w = 0;
static Model *pilot_m = NULL;
static int g_pilot = 0;
static int g_wide  = 1;  /* IMPROVEMENT 4: top-K * g_wide candidates prefetched */
static int g_pilot_evict_guard = 1; /* PILOT_EVICT_GUARD=0 to disable LFRU prefetch eviction guard */
static int g_expert_drop = 0;       /* EXPERT_DROP=1 restores fadvise(DONTNEED) after expert reads */
static int g_fused3 = 0;            /* FUSED3=1 (with IDOT=1): faster AVX2 form of the IDOT path
                                     * (fused_simd.h). Bit-identical to IDOT=1 but LOSSY vs the
                                     * default path (int8-quantized activations): High-Fidelity
                                     * option only, never Exact Mode. OFF by default. */
static int g_pilot_workers = 2;     /* PILOT_WORKERS env: concurrent prefetch worker threads */
/* LM ATOMIC Physical Giant Emulator ONLY: LMA_GIANT_STICK=0.0-1.0 (default 0,
 * meaning zero effect on any real model). Blends real router picks with the
 * previous row's picks at the same layer, mirroring lmatomic/twin.py's
 * SyntheticRouting `stick` parameter, so a synthetic giant model built with
 * random dense/gate weights (which routes ~uniformly, i.e. already the
 * "hostile" regime) can also be driven at favorable/typical locality to
 * measure whether the real scheduler/predictor benefit from it. Never used
 * outside an explicit giant-emulation run; real inference never sets this. */
static float g_giant_stick = 0.0f;

static uint64_t lfru_score(uint32_t heat, uint64_t last, uint64_t clock) {
    uint64_t age = (clock > last) ? (clock - last) : 0;
    uint64_t recent = (age < 255) ? (255 - age) : 0;
    return ((uint64_t)heat << 8) | recent;
}

/* LM ATOMIC Multi-Tier Deadline Scheduler (NVMe/RAM tiers — this engine has no
 * VRAM tier, see olmoe.c's header comment): the eviction score a slot must
 * beat to survive. Higher = keep longer. With SCHED_COST=0 this reduces to
 * `used_clock` alone, i.e. byte-for-byte the original pure-recency LRU
 * comparison — this function is a superset, not a replacement, of the old
 * behaviour. Eviction order never changes which bytes get loaded for a given
 * expert id, only which resident slot is reused first, so this cannot change
 * a single generated token (Exact Mode is unaffected). */
static uint64_t evict_keep_score(Model *m, int layer, int eid, uint64_t used_clock) {
    if (!m->sched_cost || !m->freq || !m->freq[layer] || !m->last_access)
        return used_clock;
    uint32_t heat = m->freq[layer][eid];
    uint64_t last = m->last_access[layer * m->c.n_experts + eid];
    return (lfru_score(heat, last, m->clock) << 16) | (used_clock & 0xFFFFu);
}

static void pilot_prefetch(Model *m, int lnext, const float *x, int S);
static void *pilot_worker(void *arg);
static void ensure_pilot_worker_started(Model *m);
static void slot_ensure_allocated(Model *m, Slot *s);
static void record_real_selection(Model *m, int layer, const int *idx, int K);
static void enqueue_prefetch(Model *m, int layer, int eid);

#ifdef LMA_CACHE_INDEX_TEST
static uint64_t g_slot_index_probes;
#endif

/* Runtime callers hold g_pilot_mx.  The defensive eid check is intentional:
 * an index bug must degrade to a miss, never serve another expert's weights. */
static Slot *slot_indexed(Model *m, int layer, int eid) {
    if (layer < 0 || layer >= m->c.n_layers || eid < 0 ||
        eid >= m->c.n_experts) return NULL;
    LCache *lc = &m->cache[layer];
    if (!lc->slot_by_expert) return NULL;
#ifdef LMA_CACHE_INDEX_TEST
    g_slot_index_probes++;
#endif
    int i = lc->slot_by_expert[eid];
    if (i < 0 || i >= lc->n || lc->slots[i].eid != eid) return NULL;
    return &lc->slots[i];
}

static void cache_unindex(Model *m, int layer, Slot *s) {
    LCache *lc = &m->cache[layer];
    int eid = s->eid, i = (int)(s - lc->slots);
    if (lc->slot_by_expert && eid >= 0 && eid < m->c.n_experts &&
        lc->slot_by_expert[eid] == i)
        lc->slot_by_expert[eid] = -1;
}

static void cache_hide(Model *m, int layer, Slot *s) {
    cache_unindex(m, layer, s);
    s->eid = -1;
}

static void cache_publish(Model *m, int layer, Slot *s, int eid) {
    LCache *lc = &m->cache[layer];
    cache_unindex(m, layer, s);
    s->eid = eid;
    if (lc->slot_by_expert && eid >= 0 && eid < m->c.n_experts)
        lc->slot_by_expert[eid] = (int)(s - lc->slots);
}

/* A slot being read keeps the index entry of the expert it is loading, marked
 * -(eid+2) as in glm52.c's ecache_reserve: lookups still miss and eviction
 * still skips it (eid < 0), but a second loader of the same expert can see the
 * read in flight instead of starting another one into another slot. */
static void cache_reserve(Model *m, int layer, Slot *s, int eid) {
    LCache *lc = &m->cache[layer];
    cache_hide(m, layer, s);
    s->eid = -(eid + 2);
    if (lc->slot_by_expert && eid >= 0 && eid < m->c.n_experts)
        lc->slot_by_expert[eid] = (int)(s - lc->slots);
}

/* Caller holds g_pilot_mx. */
static int slot_in_flight(Model *m, int layer, int eid) {
    if (layer < 0 || layer >= m->c.n_layers || eid < 0 ||
        eid >= m->c.n_experts) return 0;
    LCache *lc = &m->cache[layer];
    if (!lc->slot_by_expert) return 0;
    int i = lc->slot_by_expert[eid];
    return i >= 0 && i < lc->n && lc->slots[i].eid == -(eid + 2);
}

static void ensure_pilot_worker_started(Model *m) {
    if (!pilot_m) {
        pilot_m = m;
        /* LM ATOMIC Asynchronous Expert Pipeline: several workers draining the
         * same queue so several speculative NVMe reads overlap, instead of
         * serializing every prefetch through one thread. Dequeue is a short
         * mutex-protected claim (pilot_dequeue); only the actual pread runs
         * concurrently across workers — see pilot_worker(). */
        int n = g_pilot_workers > 0 ? g_pilot_workers : 1;
        if (n > 16) n = 16;
        for (int i = 0; i < n; i++) {
            pthread_t t;
            if (pthread_create(&t, NULL, pilot_worker, NULL) != 0) {
                fprintf(stderr, "Error: Failed to create pilot prefetch worker thread\n");
                exit(1);
            }
            pthread_detach(t);
        }
    }
}

/* ---------- utility ---------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
#if defined(__APPLE__)
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0*1024.0); }  /* macOS: byte */
#else
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0); }        /* Linux: KB */
#endif
/* Quanta RAM il sistema offre ancora, in GB. Serve a dimensionare la cache
 * degli esperti quando nessuno ha scelto un numero: senza questa, il default
 * e' una costante che non sa nulla ne' del modello ne' della macchina.
 *
 * Fuori da Linux e dai sistemi con _SC_AVPHYS_PAGES ritorna 0, il che rende il
 * budget automatico pari a cio' che il processo gia' tiene: la cache risulta
 * minima invece che sbagliata, e --ram (o --cap) resta la via esplicita. */
static double mem_available_gb(void) {
    /* compat.h's probe knows Linux (MemAvailable), macOS (host_statistics64)
     * and Windows (GlobalMemoryStatusEx). The Linux-only version that lived
     * here returned 0 on the other two, and 0 sized the expert cache to one
     * slot per layer: 2.5x slower without --ram, on every Windows and macOS
     * benchmark taken since (#1500). */
    double avail = compat_mem_available_gb();
    if (avail > 0.0) return avail;
    /* Not measurable here: say so once and fall back to half the physical RAM
     * where that is known, else to a small fixed budget, rather than to a
     * cache that streams every expert from disk on every token. */
    static int noted = 0;
    double total = 0.0;
#ifdef _WIN32
    double a2 = 0.0; compat_meminfo(&total, &a2);
#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page > 0) total = (double)pages * (double)page / 1e9;
#endif
    double fallback = total > 0.0 ? total * 0.5 : 8.0;
    if (!noted) {
        noted = 1;
        fprintf(stderr, "[olmoe] could not measure available RAM on this platform; assuming %.1f GB "
                        "(%s). Pass --ram <GB> to set the budget explicitly.\n",
                fallback, total > 0.0 ? "half the physical RAM" : "a fixed default");
    }
    return fallback;
}
static float *falloc(int64_t n) { float *p = malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM %ld\n",(long)n);exit(1);} return p; }

/* chat mode only (main()'s CHAT=1 path): sampling temperature/top-p and the
 * tokenizer + stop-set machinery. g_temp<=0 -> greedy (sample.h's pick_tok).
 * Declared before #include "sample.h" — it references these by name, and
 * Cfg/falloc above, without its own extern declarations. */
static float g_temp = 0.7f;   /* TEMP env overrides */
static float g_nuc  = 0.95f;  /* NUCLEUS env overrides */
#include "sample.h"

/* LM ATOMIC Exact-Mode bandwidth reduction: the checkpoint stores the dense
 * attention/lm_head weights as FP16, and this engine used to expand them to FP32
 * in RAM, doubling the bytes streamed from DRAM per token (~1.5 GB/token of the
 * ~2.3 GB a cached decode step reads). FP16->FP32 is exact, so keeping them FP16
 * resident and converting in registers changes no value. matmul_h reproduces the
 * exact reduction structure GCC emits for matmul() below (verified in the
 * assembly): ONE 8-lane accumulator, acc = acc + (x*w) per lane as separate
 * multiply and add (no FMA in the vector body), then lanes 0..7 summed
 * sequentially from 0.0f. Same order, same roundings => bit-identical results;
 * tests/ + the A/B token comparison prove it rather than assume it.
 * Enabled by LMA_DENSE_F16 (default 1) only for the standalone engine and only
 * when I % 8 == 0 and the tensor really is FP16; anything else stays FP32. */
static int g_dense_f16 = 0;
#if defined(__AVX2__) && defined(__F16C__)
#include <immintrin.h>
static void matmul_h(float *y, const float *x, const uint16_t *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint16_t *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            __m256 acc = _mm256_setzero_ps();
            for (int i = 0; i < I; i += 8) {
                __m256 wv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i)));
                acc = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(xs + i), wv), acc);
            }
            float lane[8]; _mm256_storeu_ps(lane, acc);
            float r = 0.f;
            for (int k = 0; k < 8; k++) r += lane[k];
            y[(int64_t)s * O + o] = r;
        }
    }
}
#define LMA_HAVE_MATMUL_H 1
#endif

/* y[S,O] = x[S,I] @ W^T,  W e' [O,I] row-major */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            #pragma omp simd reduction(+:acc)
            for (int i = 0; i < I; i++) acc += xs[i] * w[i];
            y[(int64_t)s * O + o] = acc;
        }
    }
}

/* y[1,O] = x[1,I] @ W^T con W quantizzato: q[O,I] int8 + scala per riga.
 * W[o,i] ~= q[o,i]*scale[o]  ->  y[o] = scale[o] * sum_i x[i]*q[o,i].
 * Su ARM: attivazione quantizzata Q8_0 (scala per blocco di 16) + dot int8
 * NEON (sdot dove c'e' dotprod) — stessa famiglia IDOT di glm.c, IDOT=0 per
 * la via scalare byte-esatta. Misurato 2.7x end-to-end su M5.
 *
 * NB: la quantizzazione delle ATTIVAZIONI rende questo percorso non
 * equivalente alla via scalare (issue #1044). Vale per NEON come per AVX2:
 * IDOT e' opt-in (IDOT=1), non piu' attivo di default. */
#if defined(__ARM_NEON)
#include <arm_neon.h>
static inline int32_t dot_i8_16(const int8_t *a, const int8_t *b) {
    int32x4_t acc = vdupq_n_s32(0);
    int8x16_t va = vld1q_s8(a), vb = vld1q_s8(b);
#if defined(__ARM_FEATURE_DOTPROD)
    acc = vdotq_s32(acc, va, vb);
#else
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va),  vget_low_s8(vb)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
#endif
    return vaddvq_s32(acc);
}
#define HAVE_FAST_DOT_I8 1
#elif defined(__AVX2__)
#include <immintrin.h>
/* x86 counterpart of the NEON path above (was scalar-only here before —
 * the only fast path was ARM, so x86 boxes silently used the scalar
 * fallback even when AVX2 was available).
 * Sign-extend both int8 vectors to int16 (exact, no precision loss) then
 * madd+horizontal-sum in int32: pure integer arithmetic, so THIS DOT is
 * bit-for-bit identical to a scalar int8 dot, just vectorized.
 *
 * That exactness does NOT extend to the branch that calls it. matmul_q's IDOT
 * path quantizes the ACTIVATIONS to Q8_0 per 16-block before calling this,
 * which the scalar fallback does not do -- so the two paths differ. This
 * comment previously read as if it covered the whole path, which is how
 * issue #1044 stayed invisible. See the note at the idot default below. */
static inline int32_t dot_i8_16(const int8_t *a, const int8_t *b) {
    __m256i va16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)a));
    __m256i vb16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)b));
    __m256i prod = _mm256_madd_epi16(va16, vb16);           /* 8 x int32, adjacent pairs summed */
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(prod), _mm256_extracti128_si256(prod, 1));
    __m128i hi64   = _mm_unpackhi_epi64(sum128, sum128);
    __m128i sum64  = _mm_add_epi32(sum128, hi64);
    __m128i hi32   = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sum32  = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}
#define HAVE_FAST_DOT_I8 1
#elif defined(__SSE4_1__)
#include <immintrin.h>
#include "sse41_kernels.h"
/* Sandy Bridge-EP path: AVX 1.0 only, no FMA, no AVX-2.
 * 16 int8 dot via two 8-wide SSE2 sign-extend + SSE4.1 madd pairs.
 * Bit-for-bit identical to the AVX2 version above (just 2x 128-bit ops
 * instead of 1x 256-bit op). NO FMA here -- this branch targets Sandy Bridge
 * which has no FMA -- so use explicit mul+add for the inner accumulation. */
static inline int32_t dot_i8_16(const int8_t *a, const int8_t *b) {
    __m128i va_lo = _mm_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)a));        /* lower 8 int8 -> 8 int16 */
    __m128i vb_lo = _mm_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)b));
    __m128i va_hi = _mm_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a + 8)));   /* upper 8 int8 -> 8 int16 */
    __m128i vb_hi = _mm_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b + 8)));
    __m128i p_lo = _mm_madd_epi16(va_lo, vb_lo);   /* 4 x int32 from 8 int16 pairs */
    __m128i p_hi = _mm_madd_epi16(va_hi, vb_hi);   /* 4 x int32 from 8 int16 pairs */
    __m128i sum = _mm_add_epi32(p_lo, p_hi);
    /* horizontal reduce 4 x int32 -> 1 x int32 */
    __m128i hi64   = _mm_unpackhi_epi64(sum, sum);
    __m128i sum64  = _mm_add_epi32(sum, hi64);
    __m128i hi32   = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
#define HAVE_FAST_DOT_I8 1
#endif
/* Test-only hook, compiled out of the shipping binary.
 *
 * matmul_q reads IDOT once into a static, so a test cannot exercise both paths
 * in one process without it. Guarded by OLMOE_TESTING so production builds have
 * neither the global nor the branch: tests/test_olmoe_matmul_q.c defines it. */
#ifdef OLMOE_TESTING
int matmul_q_idot_force = -1;
static inline void matmul_q_reset_for_test(void) { matmul_q_idot_force = -1; }
#endif

#if defined(__AVX2__)
#include "fused_simd.h"   /* FUSED3=1: quant_x_q8_avx2 + matmul_q_idot{,_pair}_v3 (bit-exact) */
#endif

static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(HAVE_FAST_DOT_I8)
    /* IDOT is OPT-IN. It quantizes the ACTIVATIONS to Q8_0 per 16-block (below),
     * which the scalar fallback never does, so the two paths are not numerically
     * equivalent: measured 5/12 vs 12/12 matching tokens against a reference
     * (issue #1044). Before 2c4e9de x86 had no fast dot at all, so IDOT=1 fell
     * through to the exact scalar path and x86 was token-exact by accident; that
     * commit silently made every AVX2 box lossy by default. Defaulting to off
     * restores the behaviour users actually had.
     *
     * Cost of turning it on: ~+5.8e-4 relative error per dot from the activation
     * quantization (glm52.c:910-915 measures the same mechanism at ~+0.117
     * nats/token on GLM, which is why GLM keeps q/k/v off IDOT). On AVX2-only
     * hardware it is also not faster: 11.01 vs 10.67 tok/s measured on an
     * i5-9600K, n=4 interleaved. Set IDOT=1 to opt in knowingly. */
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = (e && *e == '1'); }
#ifdef OLMOE_TESTING
    if (matmul_q_idot_force >= 0) idot = matmul_q_idot_force;
#endif
    if (idot && I % 16 == 0 && I <= 4096) {
        int nb = I / 16; int8_t xi[4096]; float xs[256];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b*16;
            float am = 0.f; for (int i = 0; i < 16; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f/s;
            for (int i = 0; i < 16; i++) xi[b*16+i] = (int8_t)lrintf(xb[i]*inv);
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            float acc = 0.f;
            for (int b = 0; b < nb; b++) acc += xs[b]*(float)dot_i8_16(xi+b*16, w+b*16);
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        #pragma omp simd reduction(+:acc)
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}


/* rmsnorm su una riga di lunghezza D, in-place su out (out puo' essere == x) */
static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}

static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ---------- caricamento ---------- */
/* config.json arrives from an untrusted mirror: a missing key was a NULL-deref
 * (json_get(...)->num), so require each dimension present + numeric. */
static double req_num(jval *r, const char *k){
    jval *v=json_get(r,k);
    if(!v||v->t!=J_NUM){ fprintf(stderr,"config.json: missing or non-numeric \"%s\"\n",k); exit(1); }
    return v->num;
}
static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n<0 || n>(256L<<20)){ fprintf(stderr,"%s: config.json missing or larger than 256 MB\n",path); exit(1); }  /* SEC-9 */
    char *buf = malloc((size_t)n+1); if(!buf){ fprintf(stderr,"OOM reading %s\n",path); exit(1); }
    if(fread(buf,1,(size_t)n,f)!=(size_t)n){ fprintf(stderr,"%s: short read\n",path); exit(1); } buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);
    c->hidden    = (int)req_num(r,"hidden_size");
    c->n_layers  = (int)req_num(r,"num_hidden_layers");
    c->n_heads   = (int)req_num(r,"num_attention_heads");
    c->n_kv_heads= (int)req_num(r,"num_key_value_heads");
    c->n_experts = (int)req_num(r,"num_experts");
    c->topk      = (int)req_num(r,"num_experts_per_tok");
    c->inter     = (int)req_num(r,"intermediate_size");
    c->vocab     = (int)req_num(r,"vocab_size");
    /* range-check so bad dims can't drive a later malloc(inter*hidden) / div-by-zero */
    if(c->hidden<1||c->hidden>(1<<20) || c->n_heads<1||c->n_heads>(1<<16) ||
       c->inter<1||c->inter>(1<<24) || c->vocab<1||c->vocab>(1<<24) ||
       c->n_layers<1||c->n_layers>4096 || c->n_experts<1||c->n_experts>(1<<20) ||
       c->n_kv_heads<1 || c->topk<1||c->topk>c->n_experts){
        fprintf(stderr,"config.json: dimension out of range\n"); exit(1); }
    c->head_dim  = c->hidden / c->n_heads;
    jval *th = json_get(r,"rope_theta");  c->theta = th ? (float)th->num : 10000.f;
    jval *ep = json_get(r,"rms_norm_eps"); c->eps   = ep ? (float)ep->num : 1e-5f;
    jval *nt = json_get(r,"norm_topk_prob"); c->norm_topk = (nt && nt->t==J_BOOL) ? nt->boolean : 0;
    free(buf); free(arena);
}

static float *load_t(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);   /* densa: niente DONTNEED, resta residente */
    return p;
}

/* FP16-resident load (see matmul_h): returns NULL and fills *h when the tensor is
 * FP16 with an inner dimension divisible by 8 and LMA_DENSE_F16 is on; otherwise
 * loads the usual FP32 copy and leaves *h NULL. */
static float *load_t_dense(Model *m, const char *name, uint16_t **h) {
    *h = NULL;
#ifdef LMA_HAVE_MATMUL_H
    if (g_dense_f16) {
        st_tensor *t = st_find(&m->S, name);
        if (t && t->dtype == 1 && t->rank == 2 && (t->shape[1] % 8) == 0 && t->nbytes == t->numel * 2) {
            uint16_t *p = malloc((size_t)t->nbytes);
            if (!p) { fprintf(stderr, "OOM %s\n", name); exit(1); }
            st_read_raw(&m->S, name, p, 0);
            *h = p;
            return NULL;
        }
    }
#endif
    return load_t(m, name);
}

/* y = x @ W^T with either the FP32 or the FP16-resident copy of W. */
static inline void mm_dense(float *y, const float *x, const float *W, const uint16_t *Wh,
                            int S, int I, int O) {
#ifdef LMA_HAVE_MATMUL_H
    if (Wh) { matmul_h(y, x, Wh, S, I, O); return; }
#endif
    matmul(y, x, W, S, I, O);
}

/* LM ATOMIC: close the loop from lmatomic/agent.py's offline analysis into
 * this engine's own runtime knobs. Reads <snap>/agent_profile.json (written
 * by `lmatomic agent`) for strategy.pin_hot_fraction — a crude substring
 * scan, not a JSON parser, because this is the ONE field consumed today and
 * a missing/malformed/absent file must never be fatal (it is optional
 * measured guidance, not a required input): any parse failure returns -1.0
 * and the caller falls back to HOT=0 (disabled) exactly as before this
 * existed. Never overrides an explicit HOT= the user set. */
static float lma_agent_pin_hot_fraction(const char *snap) {
    char path[600];
    snprintf(path, sizeof(path), "%s/agent_profile.json", snap);
    FILE *f = fopen(path, "rb");
    if (!f) return -1.0f;
    char buf[8192];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;
    const char *key = "\"pin_hot_fraction\"";
    char *p = strstr(buf, key);
    if (!p) return -1.0f;
    p += strlen(key);
    while (*p && (*p == ':' || *p == ' ' || *p == '\t')) p++;
    char *end = NULL;
    float v = strtof(p, &end);
    if (end == p) return -1.0f;
    if (v < 0.0f || v > 1.0f) return -1.0f;
    return v;
}

static void model_init_range(Model *m, const char *snap, int cap, int bits,
                             int layer_begin, int layer_end,
                             int load_boundaries, int init_telemetry) {
    memset(m, 0, sizeof(*m));
    m->quant_bits = bits;
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    if (layer_end == 0) layer_end = c->n_layers;
    if (layer_begin < 0 || layer_end > c->n_layers ||
        layer_begin >= layer_end) {
        fprintf(stderr, "invalid OLMoE layer range [%d,%d) for %d layers\n",
                layer_begin, layer_end, c->n_layers);
        exit(1);
    }
    double t0 = now_s();
    if (load_boundaries) {
        m->embed      = load_t(m, "model.embed_tokens.weight");
        m->lm_head    = load_t_dense(m, "lm_head.weight", &m->lm_head_h);
        m->final_norm = load_t(m, "model.norm.weight");
    }
    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[256];
    for (int i = layer_begin; i < layer_end; i++) {
        Layer *l = &m->L[i];
        #define LD(field, suffix) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t(m,nm)
        LD(in_ln,  "input_layernorm.weight");
        LD(post_ln,"post_attention_layernorm.weight");
        #define LDD(field, hfield, suffix) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t_dense(m,nm,&l->hfield)
        LDD(q, q_h, "self_attn.q_proj.weight"); LDD(k, k_h, "self_attn.k_proj.weight");
        LDD(v, v_h, "self_attn.v_proj.weight"); LDD(o, o_h, "self_attn.o_proj.weight");
        #undef LDD
        LD(qn,"self_attn.q_norm.weight"); LD(kn,"self_attn.k_norm.weight");
        LD(gate, "mlp.gate.weight");
        #undef LD
    }
    /* cap <= 0 is "you decide", the sentinel the launcher sends when nobody
     * asked for a number (glm53 already reads it that way; #1443 asked for the
     * same here). Sized HERE rather than in main because the dense weights are
     * resident by this line: rss_gb() is a measurement, not a projection, which
     * is what made the same budget in kimi_k3 (#855) the simpler of the two.
     *
     * Until now "no explicit choice" arrived as a constant 8 slots per layer,
     * which knows nothing about the model or the machine. On a 16 GB box whose
     * entire expert set is 6.5 GB that constant costs a factor of five:
     * measured on a 1204-token prefill, cap 8 gives 22.8% expert hit rate and
     * 0.045 tok/s, cap 64 gives 99.4% and 0.215 tok/s. */
    if (cap <= 0) {
        double resident = rss_gb();
        double avail = mem_available_gb();
        const char *ram_env = getenv("RAM_GB");
        double ram_arg = ram_env ? atof(ram_env) : 0.0;
        /* An explicit --ram is a ceiling on the WHOLE process. Without it take
         * 88% of what the OS still offers and add what we already hold, the
         * same fraction and the same reason as the sibling engines: overshoot
         * means an OOM kill mid-generation, which is worse than a small cache. */
        double budget = ram_arg > 0.0 ? ram_arg : resident + avail * 0.88;
        /* The KV cache is allocated later, at the first request, so project it:
         * two tensors per layer of n_heads * max_t * head_dim floats. CTX caps
         * at 4096 because attention()'s score buffer does. */
        int max_t = getenv("CTX") ? atoi(getenv("CTX")) : 4096;
        if (max_t < 1 || max_t > 4096) max_t = 4096;
        double kv_gb = 2.0 * (double)c->n_layers * c->n_heads * max_t *
                       c->head_dim * sizeof(float) / 1e9;
        /* One slot holds one expert: three int8 matrices plus their row scales,
         * the same arithmetic the Segment adapter uses to turn a memory limit
         * into a cap. */
        double slot_gb = ((double)c->hidden * c->inter * 3.0 +
                          (double)(c->inter * 2 + c->hidden) * sizeof(float)) / 1e9;
        int layers = layer_end - layer_begin;
        if (layers < 1) layers = 1;
        double room = budget - resident - kv_gb - 0.5;   /* 0.5 GB: activations */
        int derived = room > 0.0 && slot_gb > 0.0
                    ? (int)(room / slot_gb / (double)layers) : 0;
        if (derived < 1) derived = 1;
        if (derived > c->n_experts) derived = c->n_experts;
        fprintf(stderr, "[cache] %d slots/layer of %d experts: %.1f GB budget "
                        "(%s), %.1f GB dense resident, %.1f GB projected KV, "
                        "%.0f MB per expert\n",
                derived, c->n_experts, budget,
                ram_arg > 0.0 ? "RAM_GB" : "88% of what the OS still offers",
                resident, kv_gb, slot_gb * 1000.0);
        cap = derived;
    }
    m->cache = calloc(c->n_layers, sizeof(LCache));
    for (int i = layer_begin; i < layer_end; i++) {
        m->cache[i].cap = cap;
        m->cache[i].slots = calloc(cap, sizeof(Slot));
        m->cache[i].slot_by_expert = malloc((size_t)c->n_experts * sizeof(int));
        if (!m->cache[i].slot_by_expert) { fprintf(stderr,"OOM expert cache index\n"); exit(1); }
        for (int e = 0; e < c->n_experts; e++) m->cache[i].slot_by_expert[e] = -1;
    }
    /* IMPROVEMENT 2: frequency heatmap for hot expert pinning */
    if (init_telemetry) {
        rt_init("olmoe", c->n_layers, c->n_experts);
        rt_drop_row(c->n_layers);                 /* every layer routes; no MTP row */
        m->freq = rt_counts_all();                /* read sites keep their shape */
        const char *up = getenv("LMA_USAGE");    /* optional history seed */
        if (up && *up) {
            int64_t h = rt_load(up);
            if (h > 0)
                fprintf(stderr,
                        "[USAGE] expert history: %lld selections (%s)\n",
                        (long long)h, up);
        }
    } else {
        /* A process can host several ranges. Keep their optional counters
         * detached from route_trace.h's process-global singleton. */
        m->freq = calloc((size_t)c->n_layers, sizeof(*m->freq));
    }
    m->hot_pinned = 0; m->freq_token_count = 0;
    if (getenv("HOT")) {
        m->hot_n = atoi(getenv("HOT"));
    } else {
        /* LM ATOMIC: no explicit HOT= — let the Model Agent's measured
         * strategy pick it, if one was built for this model (lmatomic agent
         * --model <snap> --out <snap>). Falls back to 0 (disabled, the
         * original default) exactly as before when no profile exists. */
        float frac = lma_agent_pin_hot_fraction(snap);
        if (frac > 0.0f) {
            m->hot_n = (int)(frac * c->n_experts + 0.5f);
            fprintf(stderr, "[AGENT] no HOT= set; using pin_hot_fraction=%.2f from agent_profile.json -> HOT=%d\n",
                    frac, m->hot_n);
        } else {
            m->hot_n = 0;
        }
    }
    m->warmup_tokens = getenv("WARMUP") ? atoi(getenv("WARMUP")) : 5;
    m->token_count = 0;
    /* PREDICTION A: routing momentum — EMA of gate logits across tokens.
     * Initialized to zero; first token sets EMA = fresh logits. */
    m->momentum_logits = calloc((size_t)c->n_layers * c->n_experts, sizeof(float));
    float sv = getenv("SMOOTH") ? (float)atof(getenv("SMOOTH")) : 0.3f;
    if (sv < 0.f) sv = 0.f; if (sv > 0.95f) sv = 0.95f;
    m->pilot_smooth = sv;
    m->is_pinned = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint8_t));
    m->is_queued = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint8_t));
    m->last_access = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint64_t));
    float cl = getenv("CONF_LIMIT") ? (float)atof(getenv("CONF_LIMIT")) : 0.92f;
    if (cl < 0.1f) cl = 0.1f; if (cl > 1.0f) cl = 1.0f;
    m->pilot_conf_limit = cl;

    /* LM ATOMIC Predictive Expert Engine: co-activation tables, allocated
     * whenever PREDICT_BETA != 0 (default on, small memory: 2 * n_layers *
     * n_experts^2 * 4 bytes — 512 KB for OLMoE's 16 layers * 64 experts). */
    float pb = getenv("PREDICT_BETA") ? (float)atof(getenv("PREDICT_BETA")) : 0.4f;
    if (pb < 0.f) pb = 0.f; if (pb > 1.f) pb = 1.f;
    m->predict_beta = pb;
    if (pb > 0.f) {
        size_t tn = (size_t)c->n_layers * c->n_experts * c->n_experts;
        m->coact2 = calloc(tn, sizeof(uint32_t));
        m->coact3 = calloc(tn, sizeof(uint32_t));
    }
    for (int i = 0; i < 3; i++) m->win_valid[i] = 0;

    /* LM ATOMIC Multi-Tier Deadline Scheduler: on by default, SCHED_COST=0
     * reproduces the original pure-recency LRU exactly (evict_keep_score). */
    m->sched_cost = getenv("SCHED_COST") ? atoi(getenv("SCHED_COST")) : 1;

    /* LM ATOMIC Asynchronous Expert Pipeline: worker pool size. */
    g_pilot_workers = getenv("PILOT_WORKERS") ? atoi(getenv("PILOT_WORKERS")) : 2;
    if (g_pilot_workers < 1) g_pilot_workers = 1;

    /* LM ATOMIC Session-Adaptive Working Set: 0 (default) reproduces the
     * original one-shot-after-WARMUP hot pinning exactly. */
    m->hot_reeval_tok = getenv("HOT_REEVAL_TOK") ? atoi(getenv("HOT_REEVAL_TOK")) : 0;
    if (m->hot_reeval_tok > 0) {
        m->freq_window = calloc((size_t)c->n_layers, sizeof(*m->freq_window));
        for (int i = 0; i < c->n_layers; i++) m->freq_window[i] = calloc((size_t)c->n_experts, sizeof(uint32_t));
    }
    m->pf_flag = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint8_t));
    m->pf_issued = 0; m->pf_used = 0;
    m->disk_bytes = 0;

    if (g_giant_stick > 0.f) {
        m->giant_prev_idx = calloc((size_t)c->n_layers * c->topk, sizeof(int));
        m->giant_prev_valid = calloc((size_t)c->n_layers, sizeof(uint8_t));
        m->giant_rng = 0x9E3779B97F4A7C15ULL;
    }

    m->dense_load_s = now_s() - t0;

    /* Persistent hot pinning belongs to the standalone runtime. A Segment
     * range must never enqueue expert reads for layers it does not own. */
    char pinpath[512];
    snprintf(pinpath, sizeof(pinpath), "%s/hot_pinned.bin", snap);
    FILE *pinf = init_telemetry ? fopen(pinpath, "rb") : NULL;
    if (pinf) {
        size_t expected_size = (size_t)c->n_layers * c->n_experts;
        if (fread(m->is_pinned, 1, expected_size, pinf) == expected_size) {
            m->hot_pinned = 1;
            printf("[HOT] Loaded persistent pinning from %s\n", pinpath);
            
            if (g_pilot) {
                ensure_pilot_worker_started(m);
                for (int l = 0; l < c->n_layers; l++) {
                    for (int e = 0; e < c->n_experts; e++) {
                        if (m->is_pinned[l * c->n_experts + e]) {
                            unsigned w = __atomic_load_n(&pilot_w, __ATOMIC_RELAXED);
                            unsigned r = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
                            if (w - r < 4096) {
                                pilot_q[w & 4095].l = l; pilot_q[w & 4095].e = e;
                                pthread_mutex_lock(&g_pilot_mx);
                                m->is_queued[l * c->n_experts + e] = 1;
                                pthread_mutex_unlock(&g_pilot_mx);
                                __atomic_store_n(&pilot_w, w + 1, __ATOMIC_RELEASE);
                            }
                        }
                    }
                }
                printf("[HOT] Pre-loading pinned experts into cache...\n");
                double t_wait = now_s();
                while (1) {
                    unsigned r = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
                    unsigned w = __atomic_load_n(&pilot_w, __ATOMIC_ACQUIRE);
                    if (r == w) break;
                    sleep_ms(2);
                }
                printf("[HOT] Pre-loaded in %.1fs!\n", now_s() - t_wait);
            }
        }
        fclose(pinf);
    }
}

static void model_init(Model *m, const char *snap, int cap, int bits) {
    model_init_range(m, snap, cap, bits, 0, 0, 1, 1);
}

static void slot_ensure_allocated(Model *m, Slot *s) {
    if (s->g) return;
    Cfg *c = &m->c;
    int64_t ng = (int64_t)c->inter * c->hidden;
    int64_t nd = (int64_t)c->hidden * c->inter;
    int8_t *w_block = malloc(ng + ng + nd);
    if (!w_block) {
        fprintf(stderr, "Error: Out of memory allocating slot weights block\n");
        exit(1);
    }
    s->g = w_block;
    s->u = w_block + ng;
    s->d = w_block + ng + ng;
    float *s_block = falloc(c->inter + c->inter + c->hidden);
    s->gs = s_block;
    s->us = s_block + c->inter;
    s->ds = s_block + c->inter + c->inter;
    s->pinned = 0;
}

#ifdef LMA_CACHE_INDEX_TEST
/* Model-free tests stand in for the disk read, so they can hold a load open
 * and count how many times each expert is read. */
static void (*g_test_expert_load)(Model *m, int layer, int eid, Slot *s);
#endif

static void load_expert_merged(Model *m, int layer, int eid, Slot *s) {
#ifdef LMA_CACHE_INDEX_TEST
    if (g_test_expert_load) { g_test_expert_load(m, layer, eid, s); return; }
#endif
    char nm[256], qsnm[256];
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.merged_weight", layer, eid);
    snprintf(qsnm, sizeof(qsnm), "model.layers.%d.mlp.experts.%d.qs", layer, eid);
    /* SEC: st_init skips its numel*esz==nbytes cross-check for dtype-3 (U8/I8)
     * tensors, so a crafted header from an untrusted mirror can declare nbytes
     * far larger than the config-sized destination and st_read_raw would write
     * past w_block (heap overflow); an oversized .qs likewise overruns s->gs.
     * slot_ensure_allocated sizes those buffers exactly ng+ng+nd bytes and
     * inter+inter+hidden floats, so require an exact match. Reject, never
     * repair — same contract as qt_resolve_fmt in the GLM engine. */
    Cfg *cc = &m->c;
    int64_t ng = (int64_t)cc->inter * cc->hidden, nd = (int64_t)cc->hidden * cc->inter;
    int64_t want_w = ng + ng + nd;
    int64_t want_s = (int64_t)cc->inter + cc->inter + cc->hidden;
    st_tensor *tw = st_find(&m->S, nm), *ts = st_find(&m->S, qsnm);
    if (!tw || tw->nbytes != want_w) {
        fprintf(stderr, "%s: expert weight is %lld bytes — expected %lld for [inter=%d,hidden=%d], "
                "refusing (untrusted container)\n", nm, (long long)(tw ? tw->nbytes : -1),
                (long long)want_w, cc->inter, cc->hidden); exit(1); }
    if (!ts || ts->numel != want_s) {
        fprintf(stderr, "%s: scale array is %lld elems — expected %lld, refusing (untrusted container)\n",
                qsnm, (long long)(ts ? ts->numel : -1), (long long)want_s); exit(1); }
    double started = now_s();
    st_read_raw(&m->S, nm, s->g, g_expert_drop);
    st_read_f32(&m->S, qsnm, s->gs, 0);  /* scales are F32; use typed reader for dtype safety */
    __atomic_fetch_add(&m->disk_ns, (uint64_t)((now_s() - started) * 1e9), __ATOMIC_RELAXED);
    /* LM ATOMIC: External Bytes Moved / Generated Token, the central metric —
     * every byte this engine actually reads from disk for an expert, weights
     * plus scales, counted at the one place both are read. */
    __atomic_fetch_add(&m->disk_bytes, (uint64_t)(want_w + want_s * (int64_t)sizeof(float)), __ATOMIC_RELAXED);
}

/* ---------- cache expert: ritorna i pesi quantizzati (q+scale) da cache o disco ---------- */
/* One byte per expert: routed in this turn or not. The dashboard's Brain tab
 * reads it as the HITS bitmap after every turn (serve_hits), and it is cleared
 * there. Outside OLMOE_NO_MAIN: expert_get is in the segment adapter object. */
static void ehit_mark(Model *m, int layer, int eid) {
    Cfg *c = &m->c;
    if (!m->ehit) {
        m->ehit = calloc((size_t)c->n_layers, sizeof(uint8_t *));
        for (int i = 0; i < c->n_layers; i++) m->ehit[i] = calloc((size_t)c->n_experts, 1);
    }
    if (layer >= 0 && layer < c->n_layers && eid >= 0 && eid < c->n_experts) m->ehit[layer][eid] = 1;
}

static void expert_get(Model *m, int layer, int eid, Slot **out) {
    LCache *lc = &m->cache[layer];
    pthread_mutex_lock(&g_pilot_mx);
    ehit_mark(m, layer, eid);          /* under the lock: the routing loop is parallel */
    Slot *hit = slot_indexed(m, layer, eid);
    /* The prefetcher usually reads the next layer's experts while this one
     * computes, so a routed expert is often already on its way: wait for that
     * read to publish rather than read the same bytes again into another slot. */
    while (!hit && slot_in_flight(m, layer, eid)) {
        pthread_cond_wait(&g_pilot_cv, &g_pilot_mx);
        hit = slot_indexed(m, layer, eid);
    }
    if (hit) {
        m->hits++; hit->used = ++m->clock; *out = hit;
        if (m->last_access) m->last_access[layer * m->c.n_experts + eid] = m->clock;
        if (m->pf_flag && m->pf_flag[layer * m->c.n_experts + eid]) {
            m->pf_flag[layer * m->c.n_experts + eid] = 0;   /* credit once, like the Twin's pf_used */
            m->pf_used++;
        }
        pthread_mutex_unlock(&g_pilot_mx);
        return;
    }
    m->miss++;
    Cfg *c = &m->c;
    Slot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
        slot_ensure_allocated(m, s);
    } else {
        /* Multi-tier scheduler eviction (SCHED_COST, default on): skip pinned
         * and in-flight (eid<0) slots, then pick the lowest evict_keep_score
         * rather than the oldest by raw recency alone — see evict_keep_score. */
        int lru = -1; uint64_t lru_sc = 0;
        for (int i = 0; i < lc->n; i++) {
            if (lc->slots[i].pinned || lc->slots[i].eid < 0) continue;
            uint64_t sc = evict_keep_score(m, layer, lc->slots[i].eid, lc->slots[i].used);
            if (lru < 0 || sc < lru_sc) { lru = i; lru_sc = sc; }
        }
        if (lru < 0) {
            /* All slots are pinned or in-flight; find oldest non-in-flight slot
             * (may be pinned, but never select one currently being loaded). */
            for (int i = 0; i < lc->n; i++) {
                if (lc->slots[i].eid < 0) continue; /* never evict in-flight */
                if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
            }
        }
        while (lru < 0) {
            /* EVERY slot is in flight: each buffer is owned by an unlocked pread
             * in the pilot worker (or a demand load) that will publish into it.
             * The old last resort (lru=0) stole such a slot mid-load — two writers
             * racing the same slab, then whichever published last decided the
             * expert id the resident bytes answered to. Wait for a publish instead
             * and rescan; in-flight always drains because a load either finishes
             * or the process is already dead in the water. */
            pthread_mutex_unlock(&g_pilot_mx);
            sleep_ms(1);
            pthread_mutex_lock(&g_pilot_mx);
            for (int i = 0; i < lc->n; i++) {
                if (lc->slots[i].eid < 0) continue;
                if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
            }
        }
        s = &lc->slots[lru];
        s->pinned = 0;
    }
    cache_reserve(m, layer, s, eid);
    s->used = ++m->clock;
    pthread_mutex_unlock(&g_pilot_mx);

    load_expert_merged(m, layer, eid, s);

    pthread_mutex_lock(&g_pilot_mx);
    cache_publish(m, layer, s, eid);
    s->pinned = m->is_pinned[layer * c->n_experts + eid];
    s->used = ++m->clock;
    if (m->last_access) m->last_access[layer * c->n_experts + eid] = m->clock;
    *out = s;
    pthread_cond_broadcast(&g_pilot_cv);
    pthread_mutex_unlock(&g_pilot_mx);
}

/* ---------- IMPROVEMENT 2: pin top-N hot experts per layer ---------- */
static void pin_hot_experts(Model *m) {
    Cfg *c = &m->c;
    if (m->hot_n <= 0 || m->hot_pinned) return;
    m->hot_pinned = 1;
    
    int is_dynamic = (m->hot_n >= 100);
    double thresh = is_dynamic ? (double)m->hot_n / 1000.0 : 0.0;
    
    int pinned_total = 0;
    for (int l = 0; l < c->n_layers; l++) {
        uint32_t *freq_l = m->freq[l];
        if (!freq_l) continue;                    /* a layer with no row cannot be ranked */

        uint64_t layer_total = 0;
        for (int e = 0; e < c->n_experts; e++) layer_total += freq_l[e];
        if (layer_total == 0) continue;

        int max_pin = m->cache[l].cap - 8;
        if (max_pin < 4) max_pin = 4;
        
        int hn = is_dynamic ? max_pin : (m->hot_n < c->n_experts ? m->hot_n : c->n_experts);
        if (hn > 256) hn = 256;
        int hot_eids[256];
        int actual_hn = 0;
        
        for (int k = 0; k < hn; k++) {
            int best = -1; uint32_t bv = 0;
            for (int e = 0; e < c->n_experts; e++) {
                int already = 0;
                for (int j = 0; j < k; j++) if (hot_eids[j] == e) { already = 1; break; }
                if (!already && freq_l[e] > bv) { bv = freq_l[e]; best = e; }
            }
            if (best < 0 || bv == 0) break;
            if (is_dynamic && bv < thresh * layer_total) break;
            hot_eids[k] = best;
            actual_hn++;
        }
        
        for (int k = 0; k < actual_hn; k++) {
            int eid = hot_eids[k];
            m->is_pinned[l * c->n_experts + eid] = 1;

            int found = 0;
            pthread_mutex_lock(&g_pilot_mx);
            Slot *resident = slot_indexed(m, l, eid);
            if (resident) { resident->pinned = 1; found = 1; }
            pthread_mutex_unlock(&g_pilot_mx);
            if (!found && g_pilot > 0) {
                /* Only enqueue when the prefetch worker is active (PILOT>0). */
                ensure_pilot_worker_started(m);
                unsigned w = __atomic_load_n(&pilot_w, __ATOMIC_RELAXED);
                unsigned r = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
                int gidx = l * c->n_experts + eid;
                pthread_mutex_lock(&g_pilot_mx);
                int already = m->is_queued[gidx];
                if (!already && w - r < 4096) {
                    pilot_q[w & 4095].l = l; pilot_q[w & 4095].e = eid;
                    m->is_queued[gidx] = 1;
                    __atomic_store_n(&pilot_w, w + 1, __ATOMIC_RELEASE);
                }
                pthread_mutex_unlock(&g_pilot_mx);
            }
            pinned_total++;
        }
    }
    /* stderr, not stdout: unlike the startup-time [HOT] messages above,
     * pin_hot_experts() runs from step() mid-request in serve mode, where
     * stdout is the strict SUBMIT/DATA/DONE wire protocol — a free-text line
     * there is a protocol-corrupting bug (confirmed: reproduces identically
     * on the pre-existing binary, not introduced by the runtime work here). */
    if (is_dynamic) {
        fprintf(stderr, "[HOT] Dynamic Pinned %d experts total (thresh=%.1f%%) after %d warmup tokens\n",
               pinned_total, thresh * 100.0, m->freq_token_count);
    } else {
        fprintf(stderr, "[HOT] Pinned %d experts (top-%d/layer) after %d warmup tokens\n",
               pinned_total, m->hot_n, m->freq_token_count);
    }
}

/* ---------- LM ATOMIC Session-Adaptive Working Set ---------- */
/* Re-derives the hot-pinned set every HOT_REEVAL_TOK tokens from a decayed
 * window (m->freq_window) instead of pin_hot_experts()'s one-shot lifetime
 * m->freq, and can UNPIN an expert that falls out of the window, not just add
 * new ones. Halves the window after each pass, so activity from many tokens
 * ago fades and a session that changes topic is not stuck with its first
 * warmup window's hot set forever. HOT_REEVAL_TOK=0 (default) never calls
 * this, so existing HOT=N runs are byte-for-byte unaffected. */
static void pin_hot_experts_adaptive(Model *m) {
    Cfg *c = &m->c;
    if (m->hot_n <= 0 || !m->freq_window) return;
    int is_dynamic = (m->hot_n >= 100);
    double thresh = is_dynamic ? (double)m->hot_n / 1000.0 : 0.0;
    int nE = c->n_experts > 4096 ? 4096 : c->n_experts;
    int pinned_total = 0, unpinned_total = 0;
    uint8_t want[4096];
    for (int l = 0; l < c->n_layers; l++) {
        uint32_t *freq_l = m->freq_window[l];
        if (!freq_l) continue;
        memset(want, 0, (size_t)nE);
        uint64_t layer_total = 0;
        for (int e = 0; e < nE; e++) layer_total += freq_l[e];
        if (layer_total > 0) {
            int max_pin = m->cache[l].cap - 8; if (max_pin < 4) max_pin = 4;
            int hn = is_dynamic ? max_pin : (m->hot_n < nE ? m->hot_n : nE);
            if (hn > 256) hn = 256;
            int hot_eids[256]; int actual_hn = 0;
            for (int k = 0; k < hn; k++) {
                int best = -1; uint32_t bv = 0;
                for (int e = 0; e < nE; e++) {
                    int already = 0;
                    for (int j = 0; j < k; j++) if (hot_eids[j] == e) { already = 1; break; }
                    if (!already && freq_l[e] > bv) { bv = freq_l[e]; best = e; }
                }
                if (best < 0 || bv == 0) break;
                if (is_dynamic && bv < thresh * layer_total) break;
                hot_eids[k] = best; actual_hn++;
            }
            for (int k = 0; k < actual_hn; k++) want[hot_eids[k]] = 1;
        }
        for (int e = 0; e < nE; e++) {
            int gidx = l * c->n_experts + e;
            if (want[e] == m->is_pinned[gidx]) continue;
            m->is_pinned[gidx] = want[e];
            pthread_mutex_lock(&g_pilot_mx);
            Slot *resident = slot_indexed(m, l, e);
            if (resident) resident->pinned = want[e];
            pthread_mutex_unlock(&g_pilot_mx);
            if (want[e]) pinned_total++; else unpinned_total++;
        }
        /* decay: halve so old activity fades and the hot set can drift */
        for (int e = 0; e < nE; e++) freq_l[e] -= freq_l[e] / 2;
    }
    if (pinned_total || unpinned_total)
        fprintf(stderr, "[HOT-ADAPT] +%d pinned, -%d unpinned (token %d)\n",
                pinned_total, unpinned_total, m->token_count);
}

/* ---------- RoPE su un vettore di una testa (head_dim) a posizione assoluta pos ---------- */
static void rope_head(float *x, int pos, const Cfg *c) {
    int h = c->head_dim / 2;
    for (int j = 0; j < h; j++) {
        float inv = powf(c->theta, -2.0f * j / c->head_dim);
        float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        float a = x[j], b = x[j+h];
        x[j]   = a*cs - b*sn;
        x[j+h] = b*cs + a*sn;
    }
}

/* attenzione sui token nuovi x[S,hidden]; pos_base = posizione assoluta del primo token nuovo */
static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out) {
    Cfg *c = &m->c; int H = c->n_heads, hd = c->head_dim, D = c->hidden;
    float *q = falloc((int64_t)S*D), *k = falloc((int64_t)S*D), *vv = falloc((int64_t)S*D);
    mm_dense(q, x, l->q, l->q_h, S, D, D);
    mm_dense(k, x, l->k, l->k_h, S, D, D);
    mm_dense(vv, x, l->v, l->v_h, S, D, D);
    /* qk-norm sull'intero vettore hidden, poi RoPE per testa */
    for (int s = 0; s < S; s++) {
        rmsnorm_row(q + (int64_t)s*D, q + (int64_t)s*D, l->qn, D, c->eps);
        rmsnorm_row(k + (int64_t)s*D, k + (int64_t)s*D, l->kn, D, c->eps);
        int pos = pos_base + s;
        for (int hh = 0; hh < H; hh++) { rope_head(q + (int64_t)s*D + hh*hd, pos, c); rope_head(k + (int64_t)s*D + hh*hd, pos, c); }
    }
    /* scrive k,v nella kv-cache alle posizioni pos_base..pos_base+S-1 */
    for (int s = 0; s < S; s++) for (int hh = 0; hh < H; hh++) {
        int t = pos_base + s;
        memcpy(m->K[layer] + ((int64_t)hh*m->max_t + t)*hd, k + (int64_t)s*D + hh*hd, hd*sizeof(float));
        memcpy(m->V[layer] + ((int64_t)hh*m->max_t + t)*hd, vv + (int64_t)s*D + hh*hd, hd*sizeof(float));
    }
    int Tk = pos_base + S;             /* numero di key totali disponibili */
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = falloc((int64_t)S*D);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int hh = 0; hh < H; hh++) {
        for (int s = 0; s < S; s++) {
            int qpos = pos_base + s;
            const float *qv = q + (int64_t)s*D + hh*hd;
            float sc[4096];
            for (int t = 0; t <= qpos; t++) {          /* causale: t <= qpos */
                const float *kv = m->K[layer] + ((int64_t)hh*m->max_t + t)*hd;
                float acc = 0; for (int dd = 0; dd < hd; dd++) acc += qv[dd]*kv[dd];
                sc[t] = acc * scale;
            }
            softmax_row(sc, qpos+1);
            float *cx = ctx + (int64_t)s*D + hh*hd;
            for (int dd = 0; dd < hd; dd++) cx[dd] = 0;
            for (int t = 0; t <= qpos; t++) {
                const float *vrow = m->V[layer] + ((int64_t)hh*m->max_t + t)*hd;
                float a = sc[t];
                for (int dd = 0; dd < hd; dd++) cx[dd] += a * vrow[dd];
            }
        }
    }
    (void)Tk;
    mm_dense(out, ctx, l->o, l->o_h, S, D, D);
    free(q); free(k); free(vv); free(ctx);
}

/* LM ATOMIC Predictive Expert Engine: record a real (post-router) selection
 * into the rolling 3-layer window, and — where the window already holds the
 * real selection from 2 or 3 layers back — fold the transition into coact2 /
 * coact3. Decode-only (see moe()'s S==1 call site); measurement/learning
 * only, never fed back into routing itself. */
static void record_real_selection(Model *m, int layer, const int *idx, int K) {
    Cfg *c = &m->c; int E = c->n_experts;
    if (K > 16) K = 16;
    /* LM ATOMIC Session-Adaptive Working Set: decayed usage window, bumped
     * unconditionally (independent of the co-activation tables below) so
     * HOT_REEVAL_TOK works even with PREDICT_BETA=0. */
    if (m->freq_window && m->freq_window[layer]) {
        for (int b = 0; b < K; b++) {
            int eb = idx[b];
            if (eb >= 0 && eb < E) m->freq_window[layer][eb]++;
        }
    }
    if (!m->coact2 || !m->coact3) return;
    for (int slot = 0; slot < 3; slot++) {
        if (!m->win_valid[slot]) continue;
        int gap = layer - m->win_layer[slot];
        uint32_t *tbl = (gap == 2) ? m->coact2 : (gap == 3) ? m->coact3 : NULL;
        if (!tbl) continue;
        int src_layer = m->win_layer[slot];
        for (int a = 0; a < m->win_k[slot]; a++) {
            int ea = m->win_idx[slot][a];
            if (ea < 0 || ea >= E) continue;
            uint32_t *row = tbl + (int64_t)src_layer * E * E + (int64_t)ea * E;
            for (int b = 0; b < K; b++) {
                int eb = idx[b];
                if (eb < 0 || eb >= E) continue;
                row[eb]++;
            }
        }
    }
    /* Push this layer's real selection into the window: fill an empty slot
     * first, else evict the one holding the smallest (oldest) layer id —
     * layers are always processed in increasing order within one pass. */
    int oldest = -1;
    for (int i = 0; i < 3; i++) if (!m->win_valid[i]) { oldest = i; break; }
    if (oldest < 0) {
        oldest = 0;
        for (int i = 1; i < 3; i++) if (m->win_layer[i] < m->win_layer[oldest]) oldest = i;
    }
    m->win_layer[oldest] = layer; m->win_valid[oldest] = 1; m->win_k[oldest] = K;
    for (int b = 0; b < K; b++) m->win_idx[oldest][b] = idx[b];
}

/* LM ATOMIC: enqueue (layer, eid) for the pilot worker pool if not already
 * resident, in flight, or queued — the same queue and the same cooperation
 * contract with expert_get() (slot_in_flight + condvar wait) that
 * pilot_prefetch() and pin_hot_experts() already use, factored out so the
 * Prefill Planner below can reuse it without duplicating the ring-buffer
 * push logic a third time. */
static void enqueue_prefetch(Model *m, int layer, int eid) {
    Cfg *c = &m->c;
    pthread_mutex_lock(&g_pilot_mx);
    int resident_or_inflight = slot_indexed(m, layer, eid) != NULL || slot_in_flight(m, layer, eid);
    int gidx = layer * c->n_experts + eid;
    int already_queued = m->is_queued[gidx];
    if (!resident_or_inflight && !already_queued) m->is_queued[gidx] = 1;
    pthread_mutex_unlock(&g_pilot_mx);
    if (resident_or_inflight || already_queued) return;
    unsigned w = __atomic_load_n(&pilot_w, __ATOMIC_RELAXED);
    unsigned r = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
    if (w - r < 4096) {
        pilot_q[w & 4095].l = layer; pilot_q[w & 4095].e = eid;
        __atomic_store_n(&pilot_w, w + 1, __ATOMIC_RELEASE);
    } else {
        pthread_mutex_lock(&g_pilot_mx);
        m->is_queued[gidx] = 0;
        pthread_mutex_unlock(&g_pilot_mx);
    }
}

/* MoE sui token x[S,hidden] -> out[S,hidden] */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk, I = c->inter;
    float *logits = falloc((int64_t)S*E);
    matmul(logits, x, l->gate, S, D, E);
    memset(out, 0, (int64_t)S*D*sizeof(float));
    float *g = falloc(I), *u = falloc(I), *hh = falloc(D);

    /* LM ATOMIC Prefill Planner: routing for every row of this call is
     * computed here, in one pass, before any expert is fetched — unchanged
     * from before except that idx/val now live in a per-call buffer instead
     * of a per-row stack array, so the prefetch-ahead pass below can see
     * every row's picks at once. */
    int *all_idx = malloc((size_t)S * K * sizeof(int));
    float *all_val = malloc((size_t)S * K * sizeof(float));
    uint8_t *needed = (S > 1) ? calloc((size_t)E, 1) : NULL;
    for (int s = 0; s < S; s++) {
        float *pr = logits + (int64_t)s*E;
        if (m->momentum_logits && m->pilot_smooth > 0.f) {
            float *ema = m->momentum_logits + (int64_t)layer * E;
            int is_zero = 1;
            for (int e = 0; e < E; e++) { if (ema[e] != 0.f) { is_zero = 0; break; } }
            if (is_zero) {
                for (int e = 0; e < E; e++) ema[e] = pr[e];
            } else {
                for (int e = 0; e < E; e++) {
                    ema[e] = (1.f - m->pilot_smooth) * pr[e] + m->pilot_smooth * ema[e];
                }
            }
        }

        softmax_row(pr, E);
        /* top-K indici (selezione parziale) */
        int *idx = all_idx + (int64_t)s*K; float *val = all_val + (int64_t)s*K;
        /* shared ranking (moe_router.h): all-NaN probabilities must not reach
         * expert_get() as -1 and then last_access[layer*E - 1] (heap write) */
        lma_router_topk(pr, NULL, NULL, E, K, idx, NULL, layer, 1);
        /* LM ATOMIC Physical Giant Emulator ONLY: blend in artificial
         * repeat-locality (see g_giant_stick's declaration). No-op (g_giant_stick
         * stays 0.0) for every real model and every existing test. */
        if (g_giant_stick > 0.f && m->giant_prev_valid && m->giant_prev_valid[layer]) {
            int *prev = m->giant_prev_idx + (int64_t)layer * K;
            for (int kk = 0; kk < K; kk++) {
                uint64_t x = m->giant_rng;
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                m->giant_rng = x;
                if ((double)(x >> 11) / (double)(1ULL << 53) < g_giant_stick)
                    idx[kk] = prev[kk];
            }
        }
        if (m->giant_prev_idx) {
            memcpy(m->giant_prev_idx + (int64_t)layer * K, idx, K * sizeof(int));
            m->giant_prev_valid[layer] = 1;
        }
        for (int kk = 0; kk < K; kk++) val[kk] = pr[idx[kk]];
        if (c->norm_topk) { float sm=0; for(int kk=0;kk<K;kk++) sm+=val[kk]; for(int kk=0;kk<K;kk++) val[kk]/=sm; }
        /* IMPROVEMENT 2 activation heatmap AND the ROUTE_TRACE stream, in one
         * call. The counters were the only thing this engine recorded, and it
         * recorded them HERE, before pinning activates — rt_count keeps that
         * placement exactly. The trace is the half olmoe never had: it emits a
         * line per (moe call, position, layer), so tools/route_pairs.py,
         * route_coupling_report.py and residency_sim.py can read this engine's
         * routing the same way they read GLM's. Until now olmoe announced
         * ROUTE_TRACE at startup and then wrote a zero-byte file, because
         * rt_init() opens the stream but nothing here ever called rt_trace():
         * every consumer silently saw "no data" instead of an error.
         *
         * Only rt_route() is unconditional: it is a no-op for the counts when
         * this engine has no counter row (the !hot_pinned guard below is
         * unchanged) and a no-op for the trace when ROUTE_TRACE is unset, so a
         * run without the variable behaves exactly as before. Measurement only,
         * never the computation: idx[] and val[] are the ids and the
         * post-normalisation gates the layer is about to apply. */
        if (!m->hot_pinned) rt_route(layer, s, idx, val, K);
        /* LM ATOMIC Predictive Expert Engine: train the co-activation table
         * from the REAL selection, decode-only (S==1 — prefill has S rows in
         * parallel and no single "current" row to condition later layers on). */
        if (S == 1) record_real_selection(m, layer, idx, K);
        if (needed) for (int kk = 0; kk < K; kk++) if (idx[kk] >= 0 && idx[kk] < E) needed[idx[kk]] = 1;
    }

    /* Issue every expert this whole batch will need, in one pass, to the
     * async pipeline BEFORE the per-row compute loop below touches any of
     * them — instead of the old row-by-row pattern where each row's
     * synchronous expert_get() blocked on a cold miss one expert at a time,
     * with zero overlap even across the up-to-K experts of a single row.
     * The union is bounded by E regardless of S (a layer only ever has E
     * distinct experts), so this never grows the prefetch queue beyond one
     * layer's worth, however long the prompt. Decode (S==1) skips this —
     * there is nothing to plan ahead of a single row, and the existing
     * PILOT lookahead already covers it. */
    if (needed) {
        ensure_pilot_worker_started(m);
        for (int e = 0; e < E; e++) if (needed[e]) enqueue_prefetch(m, layer, e);
        free(needed);
    }

    for (int s = 0; s < S; s++) {
        int *idx = all_idx + (int64_t)s*K; float *val = all_val + (int64_t)s*K;
        const float *xs = x + (int64_t)s*D;
        for (int kk = 0; kk < K; kk++) {
            Slot *e; expert_get(m, layer, idx[kk], &e);
#if defined(__AVX2__)
            /* FUSED3 is a faster implementation of matmul_q's IDOT branch, and is
             * bit-identical to IDOT=1 — NOT to the default path. IDOT quantizes the
             * ACTIVATIONS to int8 (Q8_0 per 16-block), so it is a lossy High-Fidelity
             * option (issue #1044), not an Exact-Mode one: measured on real OLMoE,
             * default vs IDOT=1 vs FUSED3=1 -> default differs from both, IDOT=1 ==
             * FUSED3=1. FUSED3 used to switch IDOT on silently (idot_moe defaulted
             * to 1 here) while documenting itself as exact; it now requires IDOT=1
             * explicitly, like matmul_q. Dims %16==0, <=4096 as for IDOT. */
            static int idot_moe = -1;
            if (idot_moe < 0) { const char *ie = getenv("IDOT"); idot_moe = (ie && *ie == '1'); }
            if (g_fused3 && idot_moe && D % 16 == 0 && D <= 4096 && I % 16 == 0 && I <= 4096) {
                matmul_q_idot_pair_v3(g, u, xs, e->g, e->gs, e->u, e->us, D, I);   /* gate+up share one quant of xs */
                for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
                matmul_q_idot_v3(hh, g, e->d, e->ds, I, D);                        /* down_proj [D,I] */
            } else
#endif
            {
            matmul_q(g, xs, e->g, e->gs, D, I);     /* gate_proj [I,D] */
            matmul_q(u, xs, e->u, e->us, D, I);     /* up_proj   [I,D] */
            for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
            matmul_q(hh, g, e->d, e->ds, I, D);     /* down_proj [D,I] */
            }
            float w = val[kk];
            float *os = out + (int64_t)s*D;
            for (int d = 0; d < D; d++) os[d] += w * hh[d];
        }
    }
    free(all_idx); free(all_val);
    free(logits); free(g); free(u); free(hh);
    /* Advance the trace call counter: once per moe() invocation, after all of
     * its rows are traced. rt_trace_end() is a no-op when no stream is open.
     *
     * Outside the row loop on purpose. A batch of S == 0 traces no rows and
     * must still consume a call id, or the ids stop being consecutive and
     * residency_sim.py rejects the trace outright ("trace lacks advancing GLM
     * call ids") rather than merging two forwards into one position space. GLM
     * and glm53 advance theirs the same way. */
    rt_trace_end();
}

/* PROF phases (#1449): wall time in attention, in the MoE blocks (expert
 * loads included) and in the head, plus positions forwarded. The serve loop
 * reports per-turn deltas; expert_matmul_s is the MoE time minus the disk
 * seconds measured inside it, clamped at zero; forwards counts step() calls,
 * so tokens per forward reads 1.0 for this engine. Timing only. */
static double g_prof_attn_s = 0.0, g_prof_moe_s = 0.0, g_prof_head_s = 0.0;
static long long g_prof_forwards = 0;

static void layers_forward_range(Model *m, float *x, int S, int pos_base,
                                 int layer_begin, int layer_end,
                                 int allow_prefetch) {
    Cfg *c = &m->c;
    int D = c->hidden;
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = layer_begin; i < layer_end; i++) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        double t_attn = now_s();
        attention(m, l, i, nrm, S, pos_base, tmp);
        g_prof_attn_s += now_s() - t_attn;
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        /* IMPROVEMENT 1: PILOT=1 -> 1-layer lookahead */
        if (allow_prefetch && g_pilot >= 1 && S <= 8 && i + 1 < c->n_layers)
            pilot_prefetch(m, i + 1, x, S);
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        double t_moe = now_s();
        moe(m, l, i, nrm, S, tmp);
        g_prof_moe_s += now_s() - t_moe;
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];

        /* PREDICTION IMPROVEMENT C (Residual gate trick):
         * PILOT=2 -> prefetch layer i+2 using completed state x (containing MoE residual). */
        if (allow_prefetch && g_pilot >= 2 && S <= 8 && i + 2 < c->n_layers)
            pilot_prefetch(m, i + 2, x, S);
        if (allow_prefetch && g_pilot >= 3 && S <= 8 && i + 3 < c->n_layers)
            pilot_prefetch(m, i + 3, x, S);
        
    }
    free(nrm); free(tmp);
}

/* Canale logprobs: coda numerica per token, lettura del prefill, fotografia
 * dello stato. Questo motore e ad attenzione pura, quindi riavvolgere vuol
 * dire solo dichiarare che il prefisso tenuto e quello fotografato: le righe
 * KV di quelle posizioni non le ha toccate nessuno. Si salvano gli id e il
 * vettore di logit finale, che e il predittore del primo token fresco. */
static int    g_echo_k = 0;
static const char *g_echo_id = NULL;
static LmaPinPool g_pins;             /* piu scatti annidati, vedi pin_pool.h */
static const float *g_pin_logit = NULL; /* logit dello scatto rimesso, se c'e */
static int    g_pin_use_logit = 0;
static Tok   *g_echo_tok = NULL;

static void olmoe_echo(const char *id, int pos, int token, const float *lo, int V, int k){
    char tail[1024]; lma_logprob_tail(tail, sizeof tail, lo, V, token, k);
    char piece[512]; int n = g_echo_tok ? tok_decode(g_echo_tok, &token, 1, piece, (int)sizeof piece) : 0;
    if (n < 0) n = 0;
    printf("ECHO %s %d %d%s\n", id, n, pos, tail);
    if (n > 0) fwrite(piece, 1, (size_t)n, stdout);
    fputc('\n', stdout); fflush(stdout);
}


/* ============================ GPU decode path ============================
 * See olmoe_cuda.cu. The DLL is optional: without it (or without a GPU, or with
 * LMA_GPU=0) nothing here runs and decode stays on the CPU. Any GPU error in
 * gpu_decode() trips the breaker (gpu_on = 0) and that same token is redone on the
 * CPU, whose KV mirror is kept current after every GPU token, so a session never
 * dies because the GPU path failed. */
typedef struct {
    int   (*init)(int, int, int, int, int, int, int, int, int, int, float, float, int);
    int   (*set_layer)(int, const void *, const void *, const void *, const void *, const float *, const float *, const float *, const float *, const float *);
    int   (*set_head)(const void *, const float *);
    void  (*mem_info)(size_t *, size_t *);
    int   (*alloc_slots)(int);
    int   (*upload_expert)(int, int, int, const int8_t *, const float *);
    int   (*unmap_expert)(int, int);
    void *(*stage_ptr)(void);
    int   (*kv_upload)(int, const float *, const float *, int, int, int);
    int   (*begin_token)(const float *, int);
    int   (*layer_attn)(int);
    int   (*layer_route)(int);
    int   (*route_fetch)(int *, float *, int *);
    int   (*layer_experts)(const int *);
    int   (*finish)(float *, float *);
    void  (*shutdown)(void);
    void  (*prof_enable)(int);                 /* optional */
    int   (*prof_get)(double *, long *);       /* optional */
    int   (*pf_begin)(const float *, int, int);            /* optional: batched prefill / verification */
    int   (*pf_attn)(int);
    int   (*pf_route)(int, int *);
    int   (*pf_experts)(int);
    int   (*pf_finish)(int, int, float *, float **);
    int   (*bench_bw)(double *, double *);                 /* optional */
    void  (*set_hf)(int, float);                           /* optional: High-Fidelity dynamic expert sparsity */
    int   (*alloc_slots4)(int);                            /* optional: int4 expert pool (High-Fidelity storage) */
    int   (*upload_expert4)(int, int, int, const uint8_t *, const uint16_t *);
    void  (*set_layer_fmt)(int, int);
    int   (*astage_init)(int);                             /* optional: async upload pipeline (pinned ring + copy stream) */
    void *(*astage_ptr)(int);
    int   (*astage_wait)(int);
    int   (*upload_async)(int, int, int, int, int);
    int   (*unmap_async)(int, int);
    int   (*upload_barrier)(void);
    int   (*upload_ptr)(int, int, int, int, const void *, const void *);   /* optional: upload from mapped host memory */
    int   (*host_register)(void *, size_t);
} OcApi;
static OcApi g_oc;
static int g_gpu_verify = 0;
static double g_gpu_baseline_tok_s = 0;   /* LMA_CPU_BASELINE_TOK_S: measured CPU decode speed (0 = unknown) */
static int g_spec_k = 0;              /* LMA_SPEC: max draft tokens per verification (0 = off) */
static int g_spec_ngram = 2;          /* LMA_SPEC_NGRAM: minimum suffix match length */
static int g_gpu_prefill = 1;      /* LMA_GPU_PREFILL=0 keeps prompts on the CPU */
#define GPU_PF_MAX 128             /* must match PF_MAX in olmoe_cuda.cu */

static int oc_load_dll(void) {
#ifdef _WIN32
    char path[1024];
    const char *ov = getenv("LMA_OLMOE_CUDA_DLL");
    if (ov && *ov) snprintf(path, sizeof path, "%s", ov);
    else { DWORD n = GetModuleFileNameA(NULL, path, sizeof path);
           while (n > 0 && path[n-1] != '\\' && path[n-1] != '/') n--;
           path[n] = 0; strncat(path, "lma_olmoe_cuda.dll", sizeof path - strlen(path) - 1); }
    HMODULE h = LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!h) return 0;
    #define OCR(f) do { FARPROC p_ = GetProcAddress(h, "oc_" #f); if (!p_) return 0; memcpy(&g_oc.f, &p_, sizeof p_); } while (0)
#else
    const char *ov = getenv("LMA_OLMOE_CUDA_DLL");
    void *h = dlopen(ov && *ov ? ov : "./liblma_olmoe_cuda.so", RTLD_NOW);
    if (!h) return 0;
    #define OCR(f) do { void *p_ = dlsym(h, "oc_" #f); if (!p_) return 0; memcpy(&g_oc.f, &p_, sizeof p_); } while (0)
#endif
    OCR(init); OCR(set_layer); OCR(set_head); OCR(mem_info); OCR(alloc_slots); OCR(upload_expert);
    OCR(unmap_expert); OCR(stage_ptr); OCR(kv_upload); OCR(begin_token); OCR(layer_attn); OCR(layer_route);
    OCR(route_fetch); OCR(layer_experts); OCR(finish); OCR(shutdown);
    #undef OCR
    /* optional symbols: an older DLL without them still works */
#ifdef _WIN32
    { FARPROC a = GetProcAddress(h, "oc_prof_enable"), b = GetProcAddress(h, "oc_prof_get");
      if (a && b) { memcpy(&g_oc.prof_enable, &a, sizeof a); memcpy(&g_oc.prof_get, &b, sizeof b); } }
    { FARPROC p[5]; const char *nm[5] = {"oc_pf_begin", "oc_pf_attn", "oc_pf_route", "oc_pf_experts", "oc_pf_finish"};
      int all = 1; for (int i = 0; i < 5; i++) { p[i] = GetProcAddress(h, nm[i]); if (!p[i]) all = 0; }
      if (all) { memcpy(&g_oc.pf_begin, &p[0], sizeof p[0]); memcpy(&g_oc.pf_attn, &p[1], sizeof p[1]);
                 memcpy(&g_oc.pf_route, &p[2], sizeof p[2]); memcpy(&g_oc.pf_experts, &p[3], sizeof p[3]);
                 memcpy(&g_oc.pf_finish, &p[4], sizeof p[4]); }
      FARPROC bb = GetProcAddress(h, "oc_bench_bw"); if (bb) memcpy(&g_oc.bench_bw, &bb, sizeof bb);
      FARPROC hf = GetProcAddress(h, "oc_set_hf"); if (hf) memcpy(&g_oc.set_hf, &hf, sizeof hf);
      FARPROC a4 = GetProcAddress(h, "oc_alloc_slots4"), u4 = GetProcAddress(h, "oc_upload_expert4"), lf = GetProcAddress(h, "oc_set_layer_fmt");
      if (a4 && u4 && lf) { memcpy(&g_oc.alloc_slots4, &a4, sizeof a4); memcpy(&g_oc.upload_expert4, &u4, sizeof u4); memcpy(&g_oc.set_layer_fmt, &lf, sizeof lf); }
      FARPROC as[6]; const char *an[6] = {"oc_astage_init", "oc_astage_ptr", "oc_astage_wait", "oc_upload_async", "oc_unmap_async", "oc_upload_barrier"};
      int aall = 1; for (int i = 0; i < 6; i++) { as[i] = GetProcAddress(h, an[i]); if (!as[i]) aall = 0; }
      if (aall) { memcpy(&g_oc.astage_init, &as[0], sizeof as[0]); memcpy(&g_oc.astage_ptr, &as[1], sizeof as[1]); memcpy(&g_oc.astage_wait, &as[2], sizeof as[2]);
                  memcpy(&g_oc.upload_async, &as[3], sizeof as[3]); memcpy(&g_oc.unmap_async, &as[4], sizeof as[4]); memcpy(&g_oc.upload_barrier, &as[5], sizeof as[5]); }
      FARPROC up = GetProcAddress(h, "oc_upload_ptr"), hr = GetProcAddress(h, "oc_host_register");
      if (up) memcpy(&g_oc.upload_ptr, &up, sizeof up); if (hr) memcpy(&g_oc.host_register, &hr, sizeof hr); }
#else
    { void *a = dlsym(h, "oc_prof_enable"), *b = dlsym(h, "oc_prof_get");
      if (a && b) { memcpy(&g_oc.prof_enable, &a, sizeof a); memcpy(&g_oc.prof_get, &b, sizeof b); } }
    { void *p[5]; const char *nm[5] = {"oc_pf_begin", "oc_pf_attn", "oc_pf_route", "oc_pf_experts", "oc_pf_finish"};
      int all = 1; for (int i = 0; i < 5; i++) { p[i] = dlsym(h, nm[i]); if (!p[i]) all = 0; }
      if (all) { memcpy(&g_oc.pf_begin, &p[0], sizeof p[0]); memcpy(&g_oc.pf_attn, &p[1], sizeof p[1]);
                 memcpy(&g_oc.pf_route, &p[2], sizeof p[2]); memcpy(&g_oc.pf_experts, &p[3], sizeof p[3]);
                 memcpy(&g_oc.pf_finish, &p[4], sizeof p[4]); }
      void *bb = dlsym(h, "oc_bench_bw"); if (bb) memcpy(&g_oc.bench_bw, &bb, sizeof bb);
      void *hf = dlsym(h, "oc_set_hf"); if (hf) memcpy(&g_oc.set_hf, &hf, sizeof hf);
      void *a4 = dlsym(h, "oc_alloc_slots4"), *u4 = dlsym(h, "oc_upload_expert4"), *lf = dlsym(h, "oc_set_layer_fmt");
      if (a4 && u4 && lf) { memcpy(&g_oc.alloc_slots4, &a4, sizeof a4); memcpy(&g_oc.upload_expert4, &u4, sizeof u4); memcpy(&g_oc.set_layer_fmt, &lf, sizeof lf); }
      void *as[6]; const char *an[6] = {"oc_astage_init", "oc_astage_ptr", "oc_astage_wait", "oc_upload_async", "oc_unmap_async", "oc_upload_barrier"};
      int aall = 1; for (int i = 0; i < 6; i++) { as[i] = dlsym(h, an[i]); if (!as[i]) aall = 0; }
      if (aall) { memcpy(&g_oc.astage_init, &as[0], sizeof as[0]); memcpy(&g_oc.astage_ptr, &as[1], sizeof as[1]); memcpy(&g_oc.astage_wait, &as[2], sizeof as[2]);
                  memcpy(&g_oc.upload_async, &as[3], sizeof as[3]); memcpy(&g_oc.unmap_async, &as[4], sizeof as[4]); memcpy(&g_oc.upload_barrier, &as[5], sizeof as[5]); }
      void *up = dlsym(h, "oc_upload_ptr"), *hr = dlsym(h, "oc_host_register");
      if (up) memcpy(&g_oc.upload_ptr, &up, sizeof up); if (hr) memcpy(&g_oc.host_register, &hr, sizeof hr); }
#endif
    return 1;
}

/* software float -> IEEE half (round to nearest even); used only for the int4 group scales */
static uint16_t f32_to_f16_bits(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u; int32_t e = (int32_t)((x >> 23) & 0xFF) - 127 + 15; uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0));
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        mant |= 0x800000u; int sh = 14 - e; uint32_t h = mant >> sh, rem = mant & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)e << 10) | (mant >> 13), rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;
    return (uint16_t)(sign | h);
}
static float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16; int e = (h >> 10) & 31; uint32_t mant = h & 0x3FFu, x;
    if (e == 0) { if (!mant) x = sign; else { e = 1; while (!(mant & 0x400u)) { mant <<= 1; e--; } mant &= 0x3FFu; x = sign | ((uint32_t)(e + 112) << 23) | (mant << 13); } }
    else if (e == 31) x = sign | 0x7F800000u | (mant << 13);
    else x = sign | ((uint32_t)(e + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &x, 4); return f;
}

static int g_i4_clip = 1;   /* LMA_HF_INT4_CLIP=0 disables the MSE clip search (plain amax/7 scale) */

/* High-Fidelity storage: re-quantise one expert from its int8 rows (+ per-row scale) to symmetric int4, one FP16 scale
 * per 32 weights. Layout is the one olmoe_cuda.cu documents (nibbles gate|up|down, scales gate|up|down). */
static void quant_expert_i4(const int8_t *w, const float *rs, int I, int D, uint8_t *w4, uint16_t *s4) {
    const int8_t *mat[3] = { w, w + (int64_t)I * D, w + (int64_t)2 * I * D };
    int rows[3] = { I, I, D }, len[3] = { D, D, I }, sbase[3] = { 0, I, 2 * I };
    int64_t nib_off[3] = { 0, (int64_t)I * D / 2, (int64_t)I * D }, sc_off[3] = { 0, (int64_t)I * (D / 32), (int64_t)2 * I * (D / 32) };
    for (int t = 0; t < 3; t++) {
        #pragma omp parallel for
        for (int r = 0; r < rows[t]; r++) {
            const int8_t *q = mat[t] + (int64_t)r * len[t]; float scale = rs[sbase[t] + r];
            uint8_t *dst = w4 + nib_off[t] + (int64_t)r * (len[t] / 2); uint16_t *sd = s4 + sc_off[t] + (int64_t)r * (len[t] / 32);
            for (int g = 0; g < len[t] / 32; g++) {
                float amax = 0.f;
                for (int i = 0; i < 32; i++) { float a = fabsf((float)q[g * 32 + i] * scale); if (a > amax) amax = a; }
                /* scale search: the amax/7 scale wastes range on outliers; pick the clip (1.0 .. 0.65 of amax) with the
                 * smallest squared reconstruction error of this group (MSE-optimal clipping, as in AWQ/GPTQ-style RTN+) */
                uint16_t hb = 0; float sf = 1.f, best_err = 1e30f;
                for (int ci = 0; ci < (g_i4_clip ? 8 : 1); ci++) {
                    float sc = amax > 0.f ? amax * (1.f - 0.05f * ci) / 7.f : 1.f;
                    uint16_t hh = f32_to_f16_bits(sc); float sq = f16_bits_to_f32(hh); if (sq <= 0.f) { hh = f32_to_f16_bits(1.f); sq = 1.f; }
                    float err = 0.f;
                    for (int i = 0; i < 32; i++) {
                        float wv = (float)q[g * 32 + i] * scale; int qq = (int)lrintf(wv / sq); if (qq < -7) qq = -7; if (qq > 7) qq = 7;
                        float d = wv - (float)qq * sq; err += d * d;
                    }
                    if (err < best_err) { best_err = err; hb = hh; sf = sq; }
                }
                sd[g] = hb;
                for (int i = 0; i < 16; i++) {
                    int a = (int)lrintf((float)q[g * 32 + 2 * i] * scale / sf), b = (int)lrintf((float)q[g * 32 + 2 * i + 1] * scale / sf);
                    if (a < -7) a = -7; if (a > 7) a = 7; if (b < -7) b = -7; if (b > 7) b = 7;
                    dst[g * 16 + i] = (uint8_t)((a + 8) | ((b + 8) << 4));
                }
            }
        }
    }
}

/* read one expert's merged int8 blob + scales into the DLL's pinned staging buffer and copy it to `slot` of the
 * layer's pool (int8 as stored, or re-quantised to int4 when the layer's storage format is int4) */
static int gpu_load_expert(Model *m, int layer, int eid, int slot) {
    Cfg *c = &m->c;
    int64_t ng = (int64_t)c->inter * c->hidden, want_w = 3 * ng;
    int8_t *stage = (int8_t *)g_oc.stage_ptr();
    if (m->gpu_fmt[layer] == 1 && m->s4_open) {   /* compiled overlay: int4 blob + FP16 scales straight from disk, no re-quantisation */
        char n4[256], s4nm[256]; int64_t w4b = 3 * ng / 2, s4n = 3 * ng / 32;
        snprintf(n4, sizeof n4, "model.layers.%d.mlp.experts.%d.w4", layer, eid);
        snprintf(s4nm, sizeof s4nm, "model.layers.%d.mlp.experts.%d.s4", layer, eid);
        st_tensor *t4 = st_find(&m->S4, n4), *u4 = st_find(&m->S4, s4nm);
        if (t4 && u4 && t4->nbytes == w4b && u4->nbytes == 2 * s4n) {
            st_read_raw(&m->S4, n4, stage, g_expert_drop); st_read_raw(&m->S4, s4nm, stage + w4b, g_expert_drop);
            __atomic_fetch_add(&m->disk_bytes, (uint64_t)(w4b + 2 * s4n), __ATOMIC_RELAXED);
            return g_oc.upload_expert4(layer, eid, slot, (const uint8_t *)stage, (const uint16_t *)(stage + w4b)) > 0;
        }
    }
    char nm[256], qsnm[256];
    snprintf(nm, sizeof nm, "model.layers.%d.mlp.experts.%d.merged_weight", layer, eid);
    snprintf(qsnm, sizeof qsnm, "model.layers.%d.mlp.experts.%d.qs", layer, eid);
    st_tensor *tw = st_find(&m->S, nm), *ts = st_find(&m->S, qsnm);
    if (!tw || tw->nbytes != want_w || !ts || ts->numel != (int64_t)(2 * c->inter + c->hidden)) return 0;
    st_read_raw(&m->S, nm, stage, g_expert_drop);
    st_read_f32(&m->S, qsnm, (float *)(stage + want_w), g_expert_drop);
    __atomic_fetch_add(&m->disk_bytes, (uint64_t)(want_w + ts->numel * 4), __ATOMIC_RELAXED);
    if (m->gpu_fmt[layer] == 1) {
        static uint8_t *w4 = NULL; static uint16_t *s4 = NULL;
        if (!w4) { w4 = malloc((size_t)(3 * ng / 2)); s4 = malloc((size_t)(3 * ng / 32) * 2); if (!w4 || !s4) return 0; }
        quant_expert_i4(stage, (const float *)(stage + want_w), c->inter, c->hidden, w4, s4);
        return g_oc.upload_expert4(layer, eid, slot, w4, s4) > 0;
    }
    return g_oc.upload_expert(layer, eid, slot, stage, (const float *)(stage + want_w)) > 0;
}

/* ---- async expert upload: parallel readers -> pinned staging ring -> copy stream (overlaps PCIe with reads) ---- */
#define GPU_NB 8
static double g_t_read = 0, g_t_upl = 0, g_t_sync = 0, g_t_expk = 0;   /* miss-path wall seconds: reads, upload issue+barrier, router sync, layer_experts issue */
static int g_astage_n = 0;                       /* 0 = pipeline unavailable / disabled (LMA_GPU_ASYNC=0): synchronous loads */
static int8_t *g_tmp8[GPU_NB]; static float *g_tmpq[GPU_NB];

/* Fill `dst` (a pinned staging buffer) with expert (layer, eid) in the layout of the layer's pool. Thread-safe: preads only. */
static int gpu_stage_expert(Model *m, int layer, int eid, int bi, uint8_t *dst) {
    Cfg *c = &m->c;
    int64_t ng = (int64_t)c->inter * c->hidden, want_w = 3 * ng, w4b = 3 * ng / 2, s4n = 3 * ng / 32;
    char nm[256], qsnm[256];
    if (m->gpu_fmt[layer] == 1 && m->s4_open) {
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.experts.%d.w4", layer, eid);
        snprintf(qsnm, sizeof qsnm, "model.layers.%d.mlp.experts.%d.s4", layer, eid);
        st_tensor *t4 = st_find(&m->S4, nm), *u4 = st_find(&m->S4, qsnm);
        if (t4 && u4 && t4->nbytes == w4b && u4->nbytes == 2 * s4n) {
            st_read_raw(&m->S4, nm, dst, g_expert_drop); st_read_raw(&m->S4, qsnm, dst + w4b, g_expert_drop);
            __atomic_fetch_add(&m->disk_bytes, (uint64_t)(w4b + 2 * s4n), __ATOMIC_RELAXED);
            return 1;
        }
    }
    snprintf(nm, sizeof nm, "model.layers.%d.mlp.experts.%d.merged_weight", layer, eid);
    snprintf(qsnm, sizeof qsnm, "model.layers.%d.mlp.experts.%d.qs", layer, eid);
    st_tensor *tw = st_find(&m->S, nm), *ts = st_find(&m->S, qsnm);
    if (!tw || tw->nbytes != want_w || !ts || ts->numel != (int64_t)(2 * c->inter + c->hidden)) return 0;
    __atomic_fetch_add(&m->disk_bytes, (uint64_t)(want_w + ts->numel * 4), __ATOMIC_RELAXED);
    if (m->gpu_fmt[layer] == 0) {
        st_read_raw(&m->S, nm, dst, g_expert_drop); st_read_f32(&m->S, qsnm, (float *)(dst + want_w), g_expert_drop);
        return 1;
    }
    if (!g_tmp8[bi]) { g_tmp8[bi] = malloc((size_t)want_w); g_tmpq[bi] = malloc((size_t)(2 * c->inter + c->hidden) * sizeof(float)); if (!g_tmp8[bi] || !g_tmpq[bi]) return 0; }
    st_read_raw(&m->S, nm, g_tmp8[bi], g_expert_drop); st_read_f32(&m->S, qsnm, g_tmpq[bi], g_expert_drop);
    quant_expert_i4(g_tmp8[bi], g_tmpq[bi], c->inter, c->hidden, dst, (uint16_t *)(dst + w4b));
    return 1;
}

/* ---- memory-mapped expert reads: page cache -> PCIe with no engine-side copy ----
 * Measured on this machine (page-cache-hot shard): pread into pinned staging + upload 3.7 GB/s effective, upload straight
 * from a mapped view 5.7 GB/s, from a registered (page-locked) view 8.2 GB/s (= the PCIe ceiling). Cold data is fetched
 * with PrefetchVirtualMemory / madvise(WILLNEED) for the whole layer's misses first, so the NVMe sees a deep queue of large
 * reads instead of page faults one 4 KB page at a time. */
static int g_map_on = 1;                          /* LMA_GPU_MAP=0 keeps the pread + staging path */
static const uint8_t *st_map_base(shards *S, int fidx) {
    static const uint8_t *maps[512];
    if (fidx < 0 || fidx >= 512) return NULL;
    if (maps[fidx]) return maps[fidx];
#ifdef _WIN32
    HANDLE fh = (HANDLE)_get_osfhandle(S->fds[fidx]);
    if (fh == INVALID_HANDLE_VALUE) return NULL;
    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mh) return NULL;
    maps[fidx] = (const uint8_t *)MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
#else
    void *p = mmap(NULL, (size_t)S->sizes[fidx], PROT_READ, MAP_SHARED, S->fds[fidx], 0);
    maps[fidx] = p == MAP_FAILED ? NULL : (const uint8_t *)p;
#endif
    return maps[fidx];
}
static const uint8_t *gpu_tensor_ptr(shards *S, const char *name, int64_t want_bytes) {
    st_tensor *t = st_find(S, name);
    if (!t || t->nbytes != want_bytes) return NULL;
    int fi = st_fidx(S, t->fd);
    if (fi < 0 || t->off < 0 || t->off + t->nbytes > S->sizes[fi]) return NULL;
    const uint8_t *b = st_map_base(S, fi);
    return b ? b + t->off : NULL;
}
static void gpu_prefetch_ranges(const uint8_t **p, const int64_t *n, int cnt) {
#ifdef _WIN32
    typedef struct { PVOID a; SIZE_T n; } RangeEntry;
    typedef BOOL (WINAPI *PrefetchFn)(HANDLE, ULONG_PTR, RangeEntry *, ULONG);
    static PrefetchFn fn = NULL; static int tried = 0;
    if (!tried) { tried = 1; FARPROC f = GetProcAddress(GetModuleHandleA("kernel32.dll"), "PrefetchVirtualMemory"); if (f) memcpy(&fn, &f, sizeof f); }
    if (!fn || cnt < 1) return;
    RangeEntry r[64]; int k = cnt < 64 ? cnt : 64;
    for (int i = 0; i < k; i++) { r[i].a = (PVOID)p[i]; r[i].n = (SIZE_T)n[i]; }
    fn(GetCurrentProcess(), (ULONG_PTR)k, r, 0);
#else
    for (int i = 0; i < cnt; i++) {
        uintptr_t a = (uintptr_t)p[i] & ~(uintptr_t)4095;
        madvise((void *)a, (size_t)(n[i] + ((uintptr_t)p[i] - a)), MADV_WILLNEED);
    }
#endif
}

/* Upload `n` experts of `layer` from mapped shards. Returns 1 ok, 0 = not applicable/failed (caller uses the pread path). */
static int gpu_upload_batch_map(Model *m, int layer, const int *eids, const int *slots, int n) {
    Cfg *c = &m->c; int fmt = m->gpu_fmt[layer];
    int64_t ng = (int64_t)c->inter * c->hidden, want_w = 3 * ng, w4b = 3 * ng / 2, s4n = 3 * ng / 32;
    if (!g_oc.upload_ptr || n < 1 || n > 64) return 0;
    const uint8_t *wp[64], *sp[64]; int64_t wn = fmt ? w4b : want_w, sn = fmt ? 2 * s4n : (int64_t)(2 * c->inter + c->hidden) * 4;
    const uint8_t *pf[128]; int64_t pn[128]; int npf = 0;
    for (int j = 0; j < n; j++) {
        char a[256], b[256];
        if (fmt == 1) {
            if (!m->s4_open) return 0;
            snprintf(a, sizeof a, "model.layers.%d.mlp.experts.%d.w4", layer, eids[j]);
            snprintf(b, sizeof b, "model.layers.%d.mlp.experts.%d.s4", layer, eids[j]);
            wp[j] = gpu_tensor_ptr(&m->S4, a, wn); sp[j] = gpu_tensor_ptr(&m->S4, b, sn);
        } else {
            snprintf(a, sizeof a, "model.layers.%d.mlp.experts.%d.merged_weight", layer, eids[j]);
            snprintf(b, sizeof b, "model.layers.%d.mlp.experts.%d.qs", layer, eids[j]);
            st_tensor *tq = st_find(&m->S, b);
            if (!tq || tq->dtype != 2) return 0;                    /* scales must already be raw F32 on disk */
            wp[j] = gpu_tensor_ptr(&m->S, a, wn); sp[j] = gpu_tensor_ptr(&m->S, b, sn);
        }
        if (!wp[j] || !sp[j]) return 0;
        if (npf + 2 <= 128) { pf[npf] = wp[j]; pn[npf++] = wn; pf[npf] = sp[j]; pn[npf++] = sn; }
    }
    double tq = now_s();
    gpu_prefetch_ranges(pf, pn, npf);
    for (int j = 0; j < n; j++) {
        if (g_oc.upload_ptr(layer, eids[j], slots[j], fmt, wp[j], sp[j]) <= 0) return 0;
        __atomic_fetch_add(&m->disk_bytes, (uint64_t)(wn + sn), __ATOMIC_RELAXED);
    }
    g_t_upl += now_s() - tq;
    return g_oc.upload_barrier() > 0;
}

/* Make `n` experts of `layer` resident in the given slots (their table entries were already reserved on the host). */
static int gpu_upload_batch(Model *m, int layer, const int *eids, const int *slots, int n) {
    int fmt = m->gpu_fmt[layer];
    if (g_map_on && gpu_upload_batch_map(m, layer, eids, slots, n)) return 1;
    if (g_astage_n < 1) {
        for (int j = 0; j < n; j++) if (!gpu_load_expert(m, layer, eids[j], slots[j])) return 0;
        return 1;
    }
    for (int base = 0; base < n; base += g_astage_n) {
        int cnt = n - base < g_astage_n ? n - base : g_astage_n, ok = 1;
        double tq = now_s();
        for (int j = 0; j < cnt; j++) if (g_oc.astage_wait(j) <= 0) return 0;      /* buffer j free again (its last upload finished) */
        #pragma omp parallel for num_threads(cnt) schedule(static, 1)
        for (int j = 0; j < cnt; j++)
            if (!gpu_stage_expert(m, layer, eids[base + j], j, (uint8_t *)g_oc.astage_ptr(j))) { __atomic_store_n(&ok, 0, __ATOMIC_RELAXED); }
        if (!ok) return 0;
        g_t_read += now_s() - tq; tq = now_s();
        for (int j = 0; j < cnt; j++) if (g_oc.upload_async(j, layer, eids[base + j], slots[base + j], fmt) <= 0) return 0;
        g_t_upl += now_s() - tq;
    }
    return g_oc.upload_barrier() > 0;
}

/* LMA_HF_INT4: "all" or a list like "0,3,8-11" of layers whose experts are stored as int4 (lossy, off by default) */
static void gpu_parse_int4_layers(Model *m) {
    const char *e = getenv("LMA_HF_INT4");
    memset(m->gpu_fmt, 0, sizeof m->gpu_fmt);
    if (!e || !*e || !strcmp(e, "none")) return;
    if (!strcmp(e, "all")) { for (int l = 0; l < m->c.n_layers && l < 128; l++) m->gpu_fmt[l] = 1; return; }
    const char *p = e;
    while (*p) {
        char *end; long a = strtol(p, &end, 10); if (end == p) break; long b = a;
        if (*end == '-') { p = end + 1; b = strtol(p, &end, 10); }
        for (long l = a; l <= b && l < m->c.n_layers && l < 128; l++) if (l >= 0) m->gpu_fmt[l] = 1;
        p = end; while (*p == ',' || *p == ' ') p++;
    }
}

static void gpu_try_init(Model *m) {
    const char *en = getenv("LMA_GPU");
    if (en && atoi(en) == 0) return;
    Cfg *c = &m->c;
    if (!m->L || !m->lm_head_h || !m->L[0].q_h || c->head_dim != 128 || c->n_experts > 1024 || c->n_layers > 128) return;  /* needs FP16-resident dense */
    if (!oc_load_dll()) { if (en) fprintf(stderr, "[GPU] lma_olmoe_cuda.dll not found; staying on CPU\n"); return; }
    int max_t = getenv("LMA_GPU_CTX") ? atoi(getenv("LMA_GPU_CTX")) : 1024;
    if (max_t < 64) max_t = 64;
    if (max_t > 4096) max_t = 4096;
    double t0 = now_s();
    gpu_parse_int4_layers(m);
    if (getenv("LMA_HF_INT4_CLIP")) g_i4_clip = atoi(getenv("LMA_HF_INT4_CLIP"));
    { const char *ov = getenv("LMA_EXPERT_OVERLAY");   /* compiled int4 experts (see qcompile); missing tensors fall back to on-the-fly */
      if (ov && *ov) { st_init(&m->S4, ov); m->s4_open = 1; fprintf(stderr, "[GPU] int4 expert overlay: %s\n", ov); } }
    int n4l = 0; for (int l = 0; l < c->n_layers; l++) n4l += m->gpu_fmt[l];
    if (n4l && !(g_oc.alloc_slots4 && g_oc.upload_expert4 && g_oc.set_layer_fmt)) { fprintf(stderr, "[GPU] DLL lacks int4 support; staying on CPU\n"); return; }
    if (g_oc.init(0, c->n_layers, c->hidden, c->n_heads, c->head_dim, c->inter, c->n_experts, c->topk,
                  c->vocab, max_t, c->eps, c->theta, c->norm_topk) <= 0) {
        fprintf(stderr, "[GPU] init failed; staying on CPU\n"); return; }
    for (int l = 0; l < c->n_layers; l++) {
        Layer *L = &m->L[l];
        if (g_oc.set_layer(l, L->q_h, L->k_h, L->v_h, L->o_h, L->in_ln, L->post_ln, L->qn, L->kn, L->gate) <= 0) {
            fprintf(stderr, "[GPU] weight upload failed (layer %d); staying on CPU\n", l); g_oc.shutdown(); return; }
        if (g_oc.set_layer_fmt) g_oc.set_layer_fmt(l, m->gpu_fmt[l]);
    }
    if (g_oc.set_head(m->lm_head_h, m->final_norm) <= 0) { g_oc.shutdown(); return; }
    size_t fb = 0, tb = 0; g_oc.mem_info(&fb, &tb);
    size_t reserve = (size_t)(getenv("LMA_GPU_RESERVE_MB") ? atoi(getenv("LMA_GPU_RESERVE_MB")) : 160) << 20;
    size_t per8 = (size_t)3 * c->inter * c->hidden + ((size_t)2 * c->inter + c->hidden) * 4;
    size_t per4 = (size_t)3 * c->inter * c->hidden / 2 + (size_t)3 * c->inter * c->hidden / 32 * 2;
    int want = c->n_layers * c->n_experts, need[2] = { (c->n_layers - n4l) * c->n_experts, n4l * c->n_experts };
    size_t budget = fb > reserve ? fb - reserve : 0;
    int cap_env = getenv("LMA_GPU_SLOTS") ? atoi(getenv("LMA_GPU_SLOTS")) : want;
    if (cap_env < want) {   /* LMA_GPU_SLOTS caps the total number of resident experts (used to force partial residency) */
        int t = need[0] + need[1];
        need[0] = (int)((double)need[0] * cap_env / t); need[1] = cap_env - need[0] < need[1] ? cap_env - need[0] : need[1];
    }
    int np[2] = { 0, 0 };
    if (need[1] > 0) { long fit = (long)(budget / per4); np[1] = g_oc.alloc_slots4(need[1] < fit ? need[1] : (int)fit); budget -= (size_t)np[1] * per4; }
    if (need[0] > 0) { long fit = (long)(budget / per8); int n = need[0] < fit ? need[0] : (int)fit; if (n < 1) n = 1; np[0] = g_oc.alloc_slots(n); }
    if (np[0] + np[1] < c->topk || (need[0] && np[0] < c->topk) || (need[1] && np[1] < c->topk)) {
        fprintf(stderr, "[GPU] no VRAM for expert slots; staying on CPU\n"); g_oc.shutdown(); return; }
    m->gpu_np[0] = np[0]; m->gpu_np[1] = np[1]; m->gpu_n_slots = np[0] + np[1];
    m->gpu_slot_of = malloc((size_t)want * sizeof(int));
    for (int f = 0; f < 2; f++) { m->gpu_owner[f] = malloc((size_t)(np[f] + 1) * sizeof(int)); m->gpu_use[f] = calloc((size_t)np[f] + 1, sizeof(uint64_t));
                                  for (int i = 0; i < np[f]; i++) m->gpu_owner[f][i] = -1; }
    for (int i = 0; i < want; i++) m->gpu_slot_of[i] = -1;
    int loaded = 0, nl[2] = { 0, 0 };   /* layer-major preload; LRU refines it at run time when it does not all fit */
    for (int l = 0; l < c->n_layers; l++) {
        int f = m->gpu_fmt[l];
        for (int e = 0; e < c->n_experts && nl[f] < np[f]; e++) {
            int sl = nl[f];
            if (!gpu_load_expert(m, l, e, sl)) { fprintf(stderr, "[GPU] expert load failed\n"); g_oc.shutdown(); return; }
            m->gpu_slot_of[l * c->n_experts + e] = sl; m->gpu_owner[f][sl] = l * c->n_experts + e; nl[f]++; loaded++;
        }
    }
    m->gpu_all_resident = (loaded == want);
    if (getenv("LMA_GPU_MAP")) g_map_on = atoi(getenv("LMA_GPU_MAP"));
    if (!m->gpu_all_resident && g_oc.astage_init && !(getenv("LMA_GPU_ASYNC") && atoi(getenv("LMA_GPU_ASYNC")) == 0)) {
        int nb = getenv("LMA_GPU_ASYNC_BUFS") ? atoi(getenv("LMA_GPU_ASYNC_BUFS")) : 8;
        if (nb > GPU_NB) nb = GPU_NB;
        g_astage_n = g_oc.astage_init(nb) > 0 ? nb : 0;
    }
    m->gpu_max_t = max_t; m->gpu_kv_len = 0;
    m->gpu_kv_stage = malloc((size_t)c->n_layers * 2 * c->hidden * sizeof(float));
    m->gpu_on = 1;
    g_gpu_verify = getenv("LMA_GPU_VERIFY") ? atoi(getenv("LMA_GPU_VERIFY")) : 0;
    g_gpu_prefill = getenv("LMA_GPU_PREFILL") ? atoi(getenv("LMA_GPU_PREFILL")) : 1;
    g_spec_k = getenv("LMA_SPEC") ? atoi(getenv("LMA_SPEC")) : 0;
    if (getenv("LMA_SPEC_NGRAM")) g_spec_ngram = atoi(getenv("LMA_SPEC_NGRAM"));
    if (g_spec_ngram < 1) g_spec_ngram = 1;
    g_gpu_baseline_tok_s = getenv("LMA_CPU_BASELINE_TOK_S") ? atof(getenv("LMA_CPU_BASELINE_TOK_S")) : 0.0;
    if (g_oc.bench_bw) g_oc.bench_bw(&m->gpu_peak_d2d_gbs, &m->gpu_peak_h2d_gbs);
    /* High-Fidelity dynamic expert sparsity: LOSSY, never enabled by default (Exact Mode) */
    { int hk = getenv("LMA_HF_KEEP") ? atoi(getenv("LMA_HF_KEEP")) : 0; float hs = getenv("LMA_HF_SKIP") ? (float)atof(getenv("LMA_HF_SKIP")) : 0.f;
      if ((hk > 0 || hs > 0.f) && g_oc.set_hf) { g_oc.set_hf(hk, hs); fprintf(stderr, "[GPU] High-Fidelity sparsity ON: keep<=%d skip_rel=%.3f (lossy)\n", hk, hs); } }
    /* weight bytes one decode token reads: q/k/v/o FP16 + router FP32 per layer, K routed experts per layer (int8 or int4), lm_head FP16 */
    m->gpu_bytes_tok = (double)c->n_layers * (4.0 * c->hidden * c->hidden * 2 + (double)c->n_experts * c->hidden * 4) +
                       (double)c->topk * ((double)(c->n_layers - n4l) * per8 + (double)n4l * per4) + (double)c->vocab * c->hidden * 2;
    if (g_oc.prof_enable && getenv("LMA_GPU_PROF") && atoi(getenv("LMA_GPU_PROF"))) g_oc.prof_enable(1);
    fprintf(stderr, "[GPU] ready in %.1fs: %d/%d experts resident in VRAM (%.2f GB%s), ctx %d, %s%s\n",
            now_s() - t0, loaded, want, ((double)nl[0] * per8 + (double)nl[1] * per4) / 1e9, n4l ? ", int4 layers lossy" : "", max_t,
            m->gpu_all_resident ? "all resident (no host sync per layer)" : "partial residency (LRU + on-demand upload)",
            g_gpu_verify ? ", VERIFY vs CPU" : "");
    if (n4l) fprintf(stderr, "[GPU] High-Fidelity int4 experts ON for %d/%d layers (group-32 FP16 scales, lossy)\n", n4l, c->n_layers);
}

/* pick a slot of pool `pool`: a free one, else the least recently used that is not in `keep` */
static int gpu_pick_slot(Model *m, int pool, const int *keep, int nkeep) {
    int best = -1; uint64_t bu = ~0ULL;
    for (int i = 0; i < m->gpu_np[pool]; i++) {
        if (m->gpu_owner[pool][i] < 0) return i;
        int kept = 0;
        for (int j = 0; j < nkeep; j++) if (keep[j] == i) { kept = 1; break; }
        if (kept) continue;
        if (m->gpu_use[pool][i] < bu) { bu = m->gpu_use[pool][i]; best = i; }
    }
    return best;
}

/* ---- Atomic Compiler backend: expert overlay shards ----
 * QCOMPILE=<lo>-<hi> QC_OUT=<file> (with SNAP): re-quantise the int8 experts of layers lo..hi to the int4 group-32 format
 * and write them as ONE safetensors overlay shard (tensors `model.layers.L.mlp.experts.E.w4` U8 and `.s4` F16). The
 * runtime (LMA_EXPERT_OVERLAY) then reads 0.5625 B/weight from disk instead of re-quantising int8 at load time. The
 * quantiser is quant_expert_i4 -- the very function the on-the-fly fallback uses -- so both routes give identical weights.
 * The Python driver (lmatomic/compiler.py) orchestrates shards, checksums and resume; this function never writes
 * anywhere but QC_OUT and never touches the source model. */
static int qcompile(const char *snap, const char *spec, const char *outp) {
    Cfg c; load_cfg(&c, snap);
    shards S; st_init(&S, snap);
    int lo = 0, hi = c.n_layers - 1;
    if (sscanf(spec, "%d-%d", &lo, &hi) < 1) return 2;
    if (strchr(spec, '-') == NULL) hi = lo;
    if (lo < 0 || hi >= c.n_layers || lo > hi) { fprintf(stderr, "QCOMPILE: bad layer range %s\n", spec); return 2; }
    if (getenv("LMA_HF_INT4_CLIP")) g_i4_clip = atoi(getenv("LMA_HF_INT4_CLIP"));
    int64_t ng = (int64_t)c.inter * c.hidden, want_w = 3 * ng, w4b = 3 * ng / 2, s4n = 3 * ng / 32;
    if (c.hidden % 32 || c.inter % 32) { fprintf(stderr, "QCOMPILE: hidden/inter must be multiples of 32\n"); return 2; }
    int nl = hi - lo + 1, ne = c.n_experts;
    size_t hcap = (size_t)nl * ne * 480 + 512, hp = 0;
    char *hdr = malloc(hcap);
    hp += (size_t)snprintf(hdr + hp, hcap - hp, "{\"__metadata__\":{\"lma_overlay\":\"int4-g32-fp16\"}");
    int64_t off = 0;
    for (int l = lo; l <= hi; l++)
        for (int e = 0; e < ne; e++) {
            hp += (size_t)snprintf(hdr + hp, hcap - hp, ",\"model.layers.%d.mlp.experts.%d.w4\":{\"dtype\":\"U8\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}",
                                   l, e, (long long)w4b, (long long)off, (long long)(off + w4b));
            off += w4b;
            hp += (size_t)snprintf(hdr + hp, hcap - hp, ",\"model.layers.%d.mlp.experts.%d.s4\":{\"dtype\":\"F16\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}",
                                   l, e, (long long)s4n, (long long)off, (long long)(off + 2 * s4n));
            off += 2 * s4n;
        }
    hp += (size_t)snprintf(hdr + hp, hcap - hp, "}");
    while (hp % 8) hdr[hp++] = ' ';
    FILE *f = fopen(outp, "wb");
    if (!f) { perror(outp); return 1; }
    uint64_t hl = (uint64_t)hp;
    fwrite(&hl, 8, 1, f); fwrite(hdr, 1, hp, f);
    int8_t *w = malloc((size_t)want_w), *w_chk = NULL; (void)w_chk;
    float *qs = malloc((size_t)(2 * c.inter + c.hidden) * sizeof(float));
    uint8_t *w4 = malloc((size_t)w4b); uint16_t *s4 = malloc((size_t)s4n * 2);
    for (int l = lo; l <= hi; l++) {
        for (int e = 0; e < ne; e++) {
            char nm[256], qn[256];
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.experts.%d.merged_weight", l, e);
            snprintf(qn, sizeof qn, "model.layers.%d.mlp.experts.%d.qs", l, e);
            st_tensor *tw = st_find(&S, nm), *ts = st_find(&S, qn);
            if (!tw || tw->nbytes != want_w || !ts || ts->numel != (int64_t)(2 * c.inter + c.hidden)) {
                fprintf(stderr, "QCOMPILE: layer %d expert %d missing or mis-sized\n", l, e); fclose(f); return 1; }
            st_read_raw(&S, nm, w, 0); st_read_f32(&S, qn, qs, 0);
            quant_expert_i4(w, qs, c.inter, c.hidden, w4, s4);
            if (fwrite(w4, 1, (size_t)w4b, f) != (size_t)w4b || fwrite(s4, 2, (size_t)s4n, f) != (size_t)s4n) { perror("write"); fclose(f); return 1; }
        }
        fprintf(stderr, "[QCOMPILE] layer %d done\n", l);
    }
    if (fclose(f) != 0) { perror("close"); return 1; }
    free(w); free(qs); free(w4); free(s4); free(hdr);
    printf("QCOMPILE ok %s bytes=%lld\n", outp, (long long)(8 + hp + off));
    return 0;
}

/* one decode token on the GPU. Returns 1 on success, 0 on failure (caller falls back to the CPU). */
static int gpu_decode(Model *m, int tok, int pos, float *logit_out, int mirror_kv) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk, L = c->n_layers, H = c->n_heads, hd = c->head_dim;
    double t0 = now_s();
    if (pos >= m->gpu_max_t || !m->K || K > 16) return 0;
    if (m->gpu_kv_len < pos) {                       /* rows the CPU produced (prefill / fallback) */
        for (int l = 0; l < L; l++)
            if (g_oc.kv_upload(l, m->K[l], m->V[l], m->max_t, m->gpu_kv_len, pos) <= 0) return 0;
        m->gpu_kv_len = pos;
    }
    if (g_oc.begin_token(m->embed + (int64_t)tok * D, pos) <= 0) return 0;
    for (int l = 0; l < L; l++) {
        if (g_oc.layer_attn(l) <= 0 || g_oc.layer_route(l) <= 0) return 0;
        if (m->gpu_all_resident) {
            if (g_oc.layer_experts(NULL) <= 0) return 0;
            continue;
        }
        int ids[16], miss = 0, slots[16], kept[16], nk = 0, me[16], ms[16], nmiss = 0; float w[16];
        double tsy = now_s();
        if (g_oc.route_fetch(ids, w, &miss) <= 0) return 0;
        g_t_sync += now_s() - tsy;
        m->gpu_fetch_syncs++;
        for (int k = 0; k < K; k++) {
            if (ids[k] < 0) { slots[k] = -2; continue; }      /* dropped by High-Fidelity sparsity */
            int idx = l * E + ids[k], sl = m->gpu_slot_of[idx], pool = m->gpu_fmt[l];
            if (sl < 0) {
                sl = gpu_pick_slot(m, pool, kept, nk);
                if (sl < 0) return 0;
                int old = m->gpu_owner[pool][sl];
                if (old >= 0) { if (g_astage_n) g_oc.unmap_async(old / E, old % E); else g_oc.unmap_expert(old / E, old % E); m->gpu_slot_of[old] = -1; }
                me[nmiss] = ids[k]; ms[nmiss] = sl; nmiss++;
                m->gpu_slot_of[idx] = sl; m->gpu_owner[pool][sl] = idx; m->gpu_miss++;
            }
            m->gpu_use[pool][sl] = ++m->gpu_clock; slots[k] = sl; kept[nk++] = sl;
        }
        if (nmiss && !gpu_upload_batch(m, l, me, ms, nmiss)) return 0;
        if (g_oc.layer_experts(slots) <= 0) return 0;
    }
    if (g_oc.finish(logit_out, m->gpu_kv_stage) <= 0) return 0;
    if (mirror_kv) {
        for (int l = 0; l < L; l++)
            for (int h = 0; h < H; h++) {
                memcpy(m->K[l] + ((int64_t)h * m->max_t + pos) * hd, m->gpu_kv_stage + (int64_t)l * 2 * D + h * hd, hd * sizeof(float));
                memcpy(m->V[l] + ((int64_t)h * m->max_t + pos) * hd, m->gpu_kv_stage + (int64_t)l * 2 * D + D + h * hd, hd * sizeof(float));
            }
    }
    m->gpu_kv_len = pos + 1;
    m->gpu_tokens++; m->gpu_s += now_s() - t0;
    return 1;
}

/* Circuit-breaker health check after each GPU decode token. Returns 0 = healthy, 1 = this token is bad (redo on the
 * CPU and disable), 2 = token is fine but the optimisation must be switched off from now on. */
static int gpu_health(Model *m, const float *lg, double dt, uint64_t miss_before) {
    Cfg *c = &m->c;
    for (int i = 0; i < c->vocab; i++)
        if (!isfinite(lg[i])) { snprintf(m->gpu_trip_reason, sizeof m->gpu_trip_reason, "non-finite logits (correctness)"); return 1; }
    m->gpu_win_n++; m->gpu_win_s += dt; m->gpu_win_miss += m->gpu_miss - miss_before;
    if (m->gpu_win_n < 32) return 0;
    double tps = m->gpu_win_n / m->gpu_win_s, mpt = (double)m->gpu_win_miss / m->gpu_win_n;
    m->gpu_win_n = 0; m->gpu_win_s = 0; m->gpu_win_miss = 0;
    if (g_gpu_baseline_tok_s > 0 && tps < g_gpu_baseline_tok_s) {
        snprintf(m->gpu_trip_reason, sizeof m->gpu_trip_reason, "%s: GPU %.1f tok/s < CPU baseline %.1f tok/s (%.1f expert misses/token)",
                 mpt > 0.5 * c->topk * c->n_layers ? "cache thrash" : "slower than baseline", tps, g_gpu_baseline_tok_s, mpt);
        return 2;
    }
    return 0;
}

/* S prompt/verification tokens on the GPU in chunks of GPU_PF_MAX (batched "rows" kernels: each weight row is read
 * once per 8 tokens). Writes the logits of the LAST token to logit_out (may be NULL) and, if mirror_kv, the K/V rows
 * to the CPU cache so a later CPU fallback / prefix reuse sees them. nlog (<= 8, <= S) = how many trailing rows' logits to
 * return, [nlog][vocab] (speculative verification needs several). Returns 1 on success, 0 on failure. */
static int gpu_prefill(Model *m, const int *ids, int S, int pos_base, float *logit_out, int nlog, int mirror_kv) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, L = c->n_layers, H = c->n_heads, hd = c->head_dim;
    double t0 = now_s();
    if (!g_oc.pf_begin || S < 1 || pos_base + S > m->gpu_max_t || !m->K) return 0;
    if (m->gpu_kv_len < pos_base) {                  /* rows the CPU produced earlier (prefix reuse / fallback) */
        for (int l = 0; l < L; l++)
            if (g_oc.kv_upload(l, m->K[l], m->V[l], m->max_t, m->gpu_kv_len, pos_base) <= 0) return 0;
        m->gpu_kv_len = pos_base;
    }
    float *xr = falloc((int64_t)GPU_PF_MAX * D);
    int *kept = malloc((size_t)E * sizeof(int)), *used = malloc((size_t)E * sizeof(int)), *pme = malloc((size_t)E * sizeof(int)), *pms = malloc((size_t)E * sizeof(int));
    int ok = 1;
    for (int c0 = 0; c0 < S && ok; c0 += GPU_PF_MAX) {
        int cs = S - c0 < GPU_PF_MAX ? S - c0 : GPU_PF_MAX, pos = pos_base + c0, last = (c0 + cs == S);
        for (int s = 0; s < cs; s++) memcpy(xr + (int64_t)s * D, m->embed + (int64_t)ids[c0 + s] * D, D * sizeof(float));
        if (g_oc.pf_begin(xr, cs, pos) <= 0) { ok = 0; break; }
        for (int l = 0; l < L && ok; l++) {
            if (g_oc.pf_attn(l) <= 0) { ok = 0; break; }
            if (m->gpu_all_resident) {
                if (g_oc.pf_route(l, NULL) <= 0 || g_oc.pf_experts(l) <= 0) ok = 0;
                continue;
            }
            if (g_oc.pf_route(l, used) <= 0) { ok = 0; break; }
            m->gpu_fetch_syncs++;
            int nk = 0, npm = 0;
            for (int e = 0; e < E && ok; e++) {
                if (!used[e]) continue;
                int idx = l * E + e, sl = m->gpu_slot_of[idx], pool = m->gpu_fmt[l];
                if (sl < 0) {
                    sl = gpu_pick_slot(m, pool, kept, nk);
                    if (sl < 0) { ok = 0; break; }
                    int old = m->gpu_owner[pool][sl];
                    if (old >= 0) { if (g_astage_n) g_oc.unmap_async(old / E, old % E); else g_oc.unmap_expert(old / E, old % E); m->gpu_slot_of[old] = -1; }
                    pme[npm] = e; pms[npm] = sl; npm++;
                    m->gpu_slot_of[idx] = sl; m->gpu_owner[pool][sl] = idx; m->gpu_miss++;
                }
                m->gpu_use[pool][sl] = ++m->gpu_clock; kept[nk++] = sl;
            }
            if (ok && npm && !gpu_upload_batch(m, l, pme, pms, npm)) ok = 0;
            if (ok && g_oc.pf_experts(l) <= 0) ok = 0;
        }
        if (!ok) break;
        float *kv = NULL;
        if (last && nlog > cs) { ok = 0; break; }
        if (g_oc.pf_finish(last ? cs - nlog : 0, last ? nlog : 0, logit_out, &kv) <= 0 || !kv) { ok = 0; break; }
        if (mirror_kv) {
            for (int l = 0; l < L; l++) {
                const float *kl = kv + (int64_t)l * 2 * GPU_PF_MAX * D, *vl = kl + (int64_t)cs * D;
                for (int s = 0; s < cs; s++)
                    for (int h = 0; h < H; h++) {
                        memcpy(m->K[l] + ((int64_t)h * m->max_t + pos + s) * hd, kl + (int64_t)s * D + h * hd, hd * sizeof(float));
                        memcpy(m->V[l] + ((int64_t)h * m->max_t + pos + s) * hd, vl + (int64_t)s * D + h * hd, hd * sizeof(float));
                    }
            }
        }
        m->gpu_kv_len = pos + cs;
    }
    free(xr); free(kept); free(used); free(pme); free(pms);
    if (ok) { m->gpu_pf_rows += (uint64_t)S; m->gpu_pf_calls++; m->gpu_pf_s += now_s() - t0; }
    return ok;
}

static float *step(Model *m, const int *ids, int S, int pos_base) {
    Cfg *c = &m->c; int D = c->hidden;
    if (m->gpu_on && S > 1 && !g_gpu_verify && g_gpu_prefill && g_oc.pf_begin && !(g_echo_k > 0 && g_echo_id)) {
        float *lg = falloc(c->vocab);
        if (gpu_prefill(m, ids, S, pos_base, lg, 1, 1)) {
            m->token_count += S; m->freq_token_count += S;
            m->kv_len = pos_base + S;
            kv_prefix_record(&m->kvp, ids, pos_base, S);
            g_prof_forwards += 1;
            return lg;
        }
        free(lg);
        m->gpu_on = 0;   /* circuit breaker */
        fprintf(stderr, "[GPU] prefill failed at position %d: falling back to the CPU for the rest of the session\n", pos_base);
    }
    if (m->gpu_on && S == 1 && !g_gpu_verify) {
        float *lg = falloc(c->vocab);
        double s0 = m->gpu_s; uint64_t miss0 = m->gpu_miss;
        int ok = gpu_decode(m, ids[0], pos_base, lg, 1), hv = 0;
        if (ok) { hv = gpu_health(m, lg, m->gpu_s - s0, miss0); if (hv == 1) ok = 0; }
        if (ok) {
            m->token_count += 1; m->freq_token_count += 1;
            m->kv_len = pos_base + 1;
            kv_prefix_record(&m->kvp, ids, pos_base, 1);
            g_prof_forwards += 1;
            if (hv == 2) {   /* circuit breaker (performance): keep this token, run the rest of the session on the CPU */
                m->gpu_on = 0;
                fprintf(stderr, "[GPU] circuit breaker: %s; continuing on the CPU\n", m->gpu_trip_reason);
            }
            return lg;
        }
        free(lg);
        m->gpu_on = 0;   /* circuit breaker: this session continues on the CPU */
        fprintf(stderr, "[GPU] decode failed at position %d%s%s: falling back to the CPU for the rest of the session\n", pos_base,
                m->gpu_trip_reason[0] ? ": " : "", m->gpu_trip_reason);
    }
    if (m->gpu_on && pos_base < m->gpu_kv_len) m->gpu_kv_len = pos_base;   /* CPU rewrites these rows */
    if (g_pilot && m->token_count > 0) {
        /* Flush stale prefetch requests: clear is_queued so pilot_realload
         * will skip any entries still sitting in pilot_q for the previous
         * token.  We deliberately do NOT move pilot_w backwards; that would
         * break the ring-buffer invariant (pilot_r could exceed pilot_w if
         * the worker consumed an entry concurrently).  The worker will drain
         * the stale slots harmlessly because pilot_realload already exits
         * early when the expert is already cached or is_queued is clear. */
        pthread_mutex_lock(&g_pilot_mx);
        memset(m->is_queued, 0, (size_t)c->n_layers * c->n_experts);
        pthread_mutex_unlock(&g_pilot_mx);
    }
    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++) memcpy(x + (int64_t)s*D, m->embed + (int64_t)ids[s]*D, D*sizeof(float));
    layers_forward_range(m, x, S, pos_base, 0, c->n_layers, 1);
    /* count actual tokens processed (S>1 during prefill) */
    m->token_count += S; m->freq_token_count += S;
    if (!m->hot_pinned && m->hot_n > 0 && m->freq_token_count >= m->warmup_tokens)
        pin_hot_experts(m);
    /* LM ATOMIC Session-Adaptive Working Set: after the first (lifetime-freq)
     * pinning, keep re-deriving the hot set from the decayed window so it can
     * track a session whose topic drifts, instead of freezing at warmup. */
    if (m->hot_pinned && m->hot_reeval_tok > 0 && S == 1 &&
        m->token_count % m->hot_reeval_tok == 0)
        pin_hot_experts_adaptive(m);
    m->kv_len = pos_base + S;
    /* Recorded HERE, where the tokens entered the cache: fed[0..len-1] are
     * the ids those positions were built from, and that invariant is the
     * whole safety argument for reusing them next turn. */
    kv_prefix_record(&m->kvp, ids, pos_base, S);
    /* Lettura del prefill: un passaggio di lm_head per posizione, pagato solo
     * da chi ha chiesto il canale. La posizione p predice il token p+1; il
     * primo token fresco e predetto dalla fotografia. Cosi ogni token di
     * un'opzione ha il suo logprob, anche fuori dai primi k. */
    if (g_echo_k > 0 && g_echo_id && S > 0) {
        float *erow = falloc(D), *elog = falloc(c->vocab);
        if (g_pin_use_logit && g_pin_logit)
            olmoe_echo(g_echo_id, pos_base, ids[0], g_pin_logit, c->vocab, g_echo_k);
        for (int p = 0; p + 1 < S; p++) {
            rmsnorm_row(erow, x + (int64_t)p*D, m->final_norm, D, c->eps);
            mm_dense(elog, erow, m->lm_head, m->lm_head_h, 1, D, c->vocab);
            olmoe_echo(g_echo_id, pos_base + p + 1, ids[p+1], elog, c->vocab, g_echo_k);
        }
        free(erow); free(elog);
    }
    float *last = falloc(D);
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    double t_head = now_s();
    mm_dense(logit, last, m->lm_head, m->lm_head_h, 1, D, c->vocab);
    g_prof_head_s += now_s() - t_head;
    g_prof_forwards += 1;   /* forward passes, not positions: a prefill of S rows is one */
    free(x); free(last);
    if (m->gpu_on && g_gpu_verify && S > 1 && g_gpu_prefill && g_oc.pf_begin) {
        float *lg = falloc(c->vocab);
        if (gpu_prefill(m, ids, S, pos_base, lg, 1, 0)) {
            float mx = 0.f; int ac = 0, ag = 0;
            for (int i = 0; i < c->vocab; i++) {
                float d = fabsf(lg[i] - logit[i]); if (d > mx) mx = d;
                if (lg[i] > lg[ag]) ag = i;
                if (logit[i] > logit[ac]) ac = i;
            }
            if (mx > m->gpu_pf_max_dlogit) m->gpu_pf_max_dlogit = mx;
            m->gpu_pf_verify_n++; if (ag == ac) m->gpu_pf_verify_agree++;
        } else { m->gpu_on = 0; fprintf(stderr, "[GPU] verify prefill failed; GPU disabled\n"); }
        free(lg);
    }
    if (m->gpu_on && g_gpu_verify && S == 1) {
        /* CPU is the reference (its logits are returned); the GPU runs the same token on its own KV */
        float *lg = falloc(c->vocab);
        if (gpu_decode(m, ids[0], pos_base, lg, 0)) {
            float mx = 0.f; int ac = 0, ag = 0;
            for (int i = 0; i < c->vocab; i++) {
                float d = fabsf(lg[i] - logit[i]); if (d > mx) mx = d;
                if (lg[i] > lg[ag]) ag = i;
                if (logit[i] > logit[ac]) ac = i;
            }
            if (mx > m->gpu_max_dlogit) m->gpu_max_dlogit = mx;
            m->gpu_verify_n++; if (ag == ac) m->gpu_verify_agree++;
        } else { m->gpu_on = 0; fprintf(stderr, "[GPU] verify decode failed; GPU disabled\n"); }
        free(lg);
    }
    return logit;
}

static void pilot_realload(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer];
    Cfg *c = &m->c;

    pthread_mutex_lock(&g_pilot_mx);
    /* Early-exit if entry was flushed (is_queued cleared) while waiting. */
    if (!m->is_queued[layer * c->n_experts + eid]) {
        pthread_mutex_unlock(&g_pilot_mx);
        return;
    }
    if (slot_indexed(m, layer, eid) || slot_in_flight(m, layer, eid)) {
        m->is_queued[layer * c->n_experts + eid] = 0;
        pthread_mutex_unlock(&g_pilot_mx);
        return;
    }
    Slot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
        slot_ensure_allocated(m, s);
    } else {
        /* Multi-tier scheduler eviction (SCHED_COST, default on) — same
         * scorer as expert_get()'s demand-miss path (evict_keep_score). */
        int lru = -1; uint64_t lru_sc = 0;
        for (int i = 0; i < lc->n; i++) {
            if (lc->slots[i].pinned || lc->slots[i].eid < 0) continue;
            uint64_t sc = evict_keep_score(m, layer, lc->slots[i].eid, lc->slots[i].used);
            if (lru < 0 || sc < lru_sc) { lru = i; lru_sc = sc; }
        }
        if (lru < 0) {
            m->is_queued[layer * c->n_experts + eid] = 0;
            pthread_mutex_unlock(&g_pilot_mx);
            return; /* all pinned/in-flight, skip */
        }

        /* LFRU eviction guard: don't displace a warm resident expert with a speculation */
        if (g_pilot_evict_guard && m->freq && m->freq[layer] && m->last_access &&
            lc->slots[lru].eid >= 0) {
            int vid = lc->slots[lru].eid;
            uint64_t vs = lfru_score(m->freq[layer][vid], m->last_access[layer * c->n_experts + vid], m->clock);
            uint64_t cs = lfru_score(m->freq[layer][eid], m->last_access[layer * c->n_experts + eid], m->clock);
            if (cs <= vs + (vs >> 2) + (4u << 8)) {
                m->is_queued[layer * c->n_experts + eid] = 0;
                pthread_mutex_unlock(&g_pilot_mx);
                return; /* drop speculation */
            }
        }

        s = &lc->slots[lru]; s->pinned = 0;
    }
    cache_reserve(m, layer, s, eid); s->used = ++m->clock;
    pthread_mutex_unlock(&g_pilot_mx);

    load_expert_merged(m, layer, eid, s);
    __atomic_fetch_add(&m->pf_issued, 1, __ATOMIC_RELAXED);

    pthread_mutex_lock(&g_pilot_mx);
    cache_publish(m, layer, s, eid);
    s->pinned = m->is_pinned[layer * c->n_experts + eid];
    s->used = ++m->clock;
    if (m->last_access) m->last_access[layer * c->n_experts + eid] = m->clock;
    if (m->pf_flag) m->pf_flag[layer * c->n_experts + eid] = 1;
    m->is_queued[layer * c->n_experts + eid] = 0;
    pthread_cond_broadcast(&g_pilot_cv);
    pthread_mutex_unlock(&g_pilot_mx);
}

/* Multi-consumer-safe dequeue: the claim (read pilot_q[r], advance pilot_r) is
 * a short mutex section shared by every worker; only pilot_realload's actual
 * pread runs unlocked and concurrently across them. A lock-free ring would
 * need a completion bitmap to let out-of-order finishers advance a watermark
 * safely — not worth the risk here since the claim itself is microseconds
 * next to a multi-hundred-microsecond NVMe read. */
static int pilot_dequeue(int *layer, int *eid) {
    pthread_mutex_lock(&g_pilot_mx);
    /* pilot_r is exclusively mutex-owned from here on (multiple consumers);
     * pilot_w is still written by the single producer via atomic RELEASE
     * stores outside the lock, so read it the same way it always was. */
    unsigned r = pilot_r, w = __atomic_load_n(&pilot_w, __ATOMIC_ACQUIRE);
    if (r == w) { pthread_mutex_unlock(&g_pilot_mx); return 0; }
    *layer = pilot_q[r & 4095].l;
    *eid = pilot_q[r & 4095].e;
    pilot_r = r + 1;
    pthread_mutex_unlock(&g_pilot_mx);
    return 1;
}

static void *pilot_worker(void *arg) {
    (void)arg;
    while (1) {
        int layer, eid;
        if (!pilot_dequeue(&layer, &eid)) { sleep_ms(1); continue; }
        pilot_realload(pilot_m, layer, eid);
    }
    return NULL;
}

static void pilot_prefetch(Model *m, int lnext, const float *x, int S) {
    if (lnext < 0 || lnext >= m->c.n_layers) return;
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts;
    ensure_pilot_worker_started(m);
    float *logits = falloc((int64_t)S * E);
    Layer *l = &m->L[lnext];

    // PREDICTION IMPROVEMENT B: Apply RMSNorm to x using destination layer's post_ln
    // This scales inputs to the distribution expected by l->gate.
    float *nrm_x = falloc((int64_t)S * D);
    for (int s = 0; s < S; s++) {
        rmsnorm_row(nrm_x + (int64_t)s * D, x + (int64_t)s * D, l->post_ln, D, c->eps);
    }

    matmul(logits, nrm_x, l->gate, S, D, E);
    free(nrm_x);

    for (int s = 0; s < S; s++) {
        float *pr = logits + (int64_t)s * E;

        // PREDICTION IMPROVEMENT A: Apply routing momentum (EMA of gate logits)
        float *blended = pr;
        float *ema = m->momentum_logits + (int64_t)lnext * E;
        if (m->pilot_smooth > 0.f) {
            blended = falloc(E);
            int is_zero = 1;
            for (int e = 0; e < E; e++) { if (ema[e] != 0.f) { is_zero = 0; break; } }
            if (is_zero) {
                for (int e = 0; e < E; e++) {
                    ema[e] = pr[e];
                    blended[e] = pr[e];
                }
            } else {
                for (int e = 0; e < E; e++) {
                    blended[e] = (1.f - m->pilot_smooth) * pr[e] + m->pilot_smooth * ema[e];
                    ema[e] = blended[e]; // update EMA
                }
            }
        }

        int cand = 0;
        int idx[128];

        float max_logit = -1e30f;
        for (int e = 0; e < E; e++) { if (blended[e] > max_logit) max_logit = blended[e]; }
        float *exps = falloc(E);
        float sum_exps = 0.f;
        for (int e = 0; e < E; e++) {
            exps[e] = expf(blended[e] - max_logit);
            sum_exps += exps[e];
        }

        /* LM ATOMIC Predictive Expert Engine: blend the gate-logit softmax
         * score above with the learned co-activation score conditioned on
         * whichever REAL (not speculative) selection is closest behind this
         * call — gap=2 covers both the depth-1 call (source lnext-2==i-1)
         * and the depth-2 call (source lnext-2==i); gap=3 covers the depth-3
         * call (source lnext-3==i). Falls back to the original gate-only
         * score, byte-for-byte, when no matching real selection is in the
         * window yet (early layers/tokens) or PREDICT_BETA=0. */
        float *combined = falloc(E);
        float combined_total = sum_exps;
        int have_coact = 0;
        float beta = m->predict_beta;
        if (beta > 0.f && (m->coact2 || m->coact3)) {
            for (int gap = 2; gap <= 3 && !have_coact; gap++) {
                uint32_t *tbl = (gap == 2) ? m->coact2 : m->coact3;
                if (!tbl) continue;
                for (int slot = 0; slot < 3; slot++) {
                    if (!m->win_valid[slot] || lnext - m->win_layer[slot] != gap) continue;
                    int src = m->win_layer[slot];
                    float *cscore = falloc(E);
                    double tot = 0.0;
                    for (int a = 0; a < m->win_k[slot]; a++) {
                        int ea = m->win_idx[slot][a];
                        if (ea < 0 || ea >= E) continue;
                        uint32_t *row = tbl + (int64_t)src * E * E + (int64_t)ea * E;
                        for (int e = 0; e < E; e++) { cscore[e] += (float)row[e]; tot += row[e]; }
                    }
                    if (tot > 0.0) {
                        for (int e = 0; e < E; e++)
                            combined[e] = (1.f - beta) * (exps[e] / sum_exps) + beta * (float)(cscore[e] / tot);
                        combined_total = 1.f;
                        have_coact = 1;
                    }
                    free(cscore);
                    break;
                }
            }
        }
        if (!have_coact) for (int e = 0; e < E; e++) combined[e] = exps[e];
        free(exps);

        float cum_sum = 0.f;
        int min_cand = c->topk;
        int max_cand = c->topk * g_wide;
        if (max_cand < min_cand) max_cand = min_cand;
        if (max_cand > 128) max_cand = 128; /* idx[] buffer bound */
        if (max_cand > E) max_cand = E;

        for (int kk = 0; kk < max_cand; kk++) {
            float bv;
            int best = lma_router_best(combined, NULL, NULL, E, idx, kk, &bv);
            if (best < 0) break;
            idx[kk] = best;
            cum_sum += bv;
            cand++;
            if (cum_sum >= m->pilot_conf_limit * combined_total && cand >= min_cand) {
                break;
            }
        }
        free(combined);

        if (blended != pr) free(blended);

        /* IMPROVEMENT 5: sort candidates by eid for sequential SSD read locality */
        for (int a = 0; a < cand-1; a++)
            for (int b = a+1; b < cand; b++)
                if (idx[b] >= 0 && (idx[a] < 0 || idx[a] > idx[b])) { int t = idx[a]; idx[a] = idx[b]; idx[b] = t; }

        for (int kk = 0; kk < cand; kk++) {
            int eid = idx[kk];
            if (eid < 0) continue;
            int found = 0;
            pthread_mutex_lock(&g_pilot_mx);
            found = slot_indexed(m, lnext, eid) != NULL;
            pthread_mutex_unlock(&g_pilot_mx);
            if (!found) {
                int gidx = lnext * E + eid;
                pthread_mutex_lock(&g_pilot_mx);
                int already_queued = m->is_queued[gidx];
                if (!already_queued) {
                    m->is_queued[gidx] = 1;
                }
                pthread_mutex_unlock(&g_pilot_mx);

                if (!already_queued) {
                    unsigned w2 = __atomic_load_n(&pilot_w, __ATOMIC_RELAXED);
                    unsigned r2 = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
                    if (w2 - r2 < 4096) {
                        pilot_q[w2 & 4095].l = lnext;
                        pilot_q[w2 & 4095].e = eid;
                        __atomic_store_n(&pilot_w, w2 + 1, __ATOMIC_RELEASE);
                    } else {
                        pthread_mutex_lock(&g_pilot_mx);
                        m->is_queued[gidx] = 0;
                        pthread_mutex_unlock(&g_pilot_mx);
                    }
                }
            }
        }
    }
    free(logits);
}


/* generazione greedy. prompt[np] -> riempie out[np+n_new] */
static void generate(Model *m, const int *prompt, int np, int n_new, int *out) {
    Cfg *c = &m->c;
    m->max_t = np + n_new;
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        m->K[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
    }
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    float *logit = step(m, prompt, np, 0);          /* PREFILL */
    int len = np;
    for (int s = 0; s < n_new; s++) {
        int best = 0; float bv = logit[0];
        for (int i = 1; i < c->vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        free(logit);
        out[len++] = best;
        if (s == n_new - 1) break;
        int one = best;
        logit = step(m, &one, 1, len - 1);          /* DECODE */
    }
}

/* teacher-forced NLL of full_ids[np..nfull): feed the REFERENCE token at each step
 * (never the argmax), accumulate -log softmax(logits)[next_ref]. A loss meter for
 * throughput experiments: same engine path as decode, so hit rate/speed stay
 * comparable, but quality is measured as perplexity instead of exact-match.
 * Cross-checked vs HF transformers bf16 on identical token ids: engine (int8
 * experts) 12.11 ppl vs reference 12.25 (#108). Enabled by PPL=1. */
static int tf_nll(Model *m, const int *full, int nfull, int np, double *nll_out) {
    Cfg *c = &m->c;
    m->max_t = nfull;
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        m->K[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
    }
    double nll = 0; int scored = 0;
    /* QDUMP=<path>: Quality Governor probe. int32 vocab, int32 n, then n x (int32 target, float32 log-softmax[vocab]). */
    FILE *qd = getenv("QDUMP") && *getenv("QDUMP") ? fopen(getenv("QDUMP"), "wb") : NULL;
    if (qd) { int32_t hdr[2] = { c->vocab, 0 }; fwrite(hdr, sizeof hdr, 1, qd); }
    float *logit = step(m, full, np, 0);              /* prefill on the prompt */
    for (int i = np; i < nfull; i++) {
        /* log softmax(logit)[full[i]] without materializing the softmax */
        float mx = logit[0]; for (int v = 1; v < c->vocab; v++) if (logit[v] > mx) mx = logit[v];
        double Z = 0; for (int v = 0; v < c->vocab; v++) Z += exp((double)logit[v] - mx);
        if (qd) {
            int32_t tg = full[i]; float lz = (float)(mx + log(Z));
            fwrite(&tg, sizeof tg, 1, qd);
            for (int v = 0; v < c->vocab; v++) { float lp = logit[v] - lz; fwrite(&lp, sizeof lp, 1, qd); }
        }
        nll += -((double)logit[full[i]] - mx - log(Z));
        scored++;
        free(logit); logit = NULL;
        if (i == nfull - 1) break;
        logit = step(m, &full[i], 1, i);              /* teacher forcing */
    }
    if (logit) free(logit);
    if (qd) { int32_t n = scored; fseek(qd, 4, SEEK_SET); fwrite(&n, sizeof n, 1, qd); fclose(qd); }
    *nll_out = nll / scored;
    return scored;
}

/* ---------- interactive chat mode (CHAT=1) ---------- */
/* OLMoE-Instruct's real template (tokenizer_config.json's chat_template):
 *   {{bos_token}}<|user|>\n{msg}\n<|assistant|>\n{reply}{eos_token}\n<|user|>\n...
 * bos_token == eos_token == "|||IP_ADDRESS|||" (a genuine OLMoE tokenizer quirk,
 * a PII-scrubbing artifact repurposed as the BOS/EOS marker — not a bug here).
 * It's an added special token, so tok_encode() tokenizes the literal string as
 * one atomic id, same as any other added token. <|user|>/<|assistant|> are NOT
 * special tokens in this tokenizer — just plain text the model was trained to
 * treat as turn markers.
 * Turn 1 gets bos_token with no separator before "<|user|>"; every later turn
 * gets eos_token+"\n" first (closing the previous assistant turn we never
 * explicitly appended to history, matching the is_stop-skips-append design
 * below) before its own "<|user|>...". */
static int fmt_user_turn(char *out, int cap, const char *msg, int first_turn) {
    int n = first_turn
        ? snprintf(out, cap, "|||IP_ADDRESS|||<|user|>\n%s\n<|assistant|>\n", msg)
        : snprintf(out, cap, "|||IP_ADDRESS|||\n<|user|>\n%s\n<|assistant|>\n", msg);
    return (n < 0 || n >= cap) ? -1 : n;
}

/* KV cache allocated ONCE for ctx_cap and never reallocated — hist_len only
 * grows (or resets to 0 on /reset), so every turn after the first reuses the
 * previous turns' cached keys/values: real multi-turn context, not a fresh
 * generate() call per message. ctx_cap capped at 4096: attention()'s per-head
 * score buffer (sc[4096]) is fixed-size, any position >= 4096 would overflow it. */
static void run_chat(Model *m, Tok *T, int ctx_cap) {
    Cfg *c = &m->c;
    m->max_t = ctx_cap;
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        m->K[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
    }

    int tok_eos = tok_id_of(T, "|||IP_ADDRESS|||");
    stops_arm_tok(c, tok_eos, T);

    int max_new = getenv("MAX_NEW") ? atoi(getenv("MAX_NEW")) : 512;
    if (max_new < 1) max_new = 1;

    int *hist = malloc((size_t)ctx_cap * sizeof(int));
    int hist_len = 0;
    int first_turn = 1;

    char *line = malloc(8192);
    char *turn = malloc(8192 + 64);
    int  *newids = malloc(8192 * sizeof(int));
    int  *gen = malloc((size_t)max_new * sizeof(int));
    char *outbuf = malloc(65536);

    fprintf(stderr, "olmoe chat — SNAP=%s, ctx=%d, TEMP=%.2f, NUCLEUS=%.2f\n"
                     "  type a message and press enter; /reset clears context; Ctrl-D exits\n",
            getenv("SNAP"), ctx_cap, g_temp, g_nuc);

    for (;;) {
        printf("\n> "); fflush(stdout);
        if (!fgets(line, 8192, stdin)) break;
        size_t L = strlen(line);
        while (L > 0 && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = 0;
        if (L == 0) continue;
        if (!strcmp(line, "/reset")) { hist_len = 0; first_turn = 1; fprintf(stderr, "[chat] context reset\n"); continue; }

        int tn = fmt_user_turn(turn, 8192 + 64, line, first_turn);
        if (tn < 0) { fprintf(stderr, "[chat] message too long, skipped\n"); continue; }
        first_turn = 0;
        int nn = tok_encode(T, turn, tn, newids, 8192);

        if (hist_len + nn + max_new > ctx_cap) {
            fprintf(stderr, "[chat] context window full (%d/%d tokens) — /reset to start over\n",
                    hist_len + nn, ctx_cap);
            continue;
        }

        float *logit = step(m, newids, nn, hist_len);
        hist_len += nn;

        /* Every token counted in hist_len must have gone through step() exactly
         * once (that's what writes its KV-cache slot) — otherwise a later turn's
         * attention would read an uninitialized slot. So even on the turn's LAST
         * token we still call step() once to populate its KV before breaking;
         * only its returned logit (which nothing will consume) is discarded. */
        int ngen = 0;
        for (int s = 0; s < max_new; s++) {
            int nt = pick_tok(logit, c->vocab, -1);
            free(logit); logit = NULL;
            if (is_stop(nt)) break;
            hist[hist_len] = nt; gen[ngen++] = nt; hist_len++;
            int room_left = (s < max_new - 1) && (hist_len < ctx_cap);
            logit = step(m, &nt, 1, hist_len - 1);
            if (!room_left) {
                if (hist_len >= ctx_cap) fprintf(stderr, "\n[chat] context window full mid-reply — /reset to start over\n");
                free(logit); logit = NULL;
                break;
            }
        }

        int outn = tok_decode(T, gen, ngen, outbuf, 65535);
        outbuf[outn] = 0;
        printf("%s\n", outbuf);
        fflush(stdout);
    }
    free(line); free(turn); free(newids); free(gen); free(outbuf); free(hist);
}

/* ---------- serve mode: openai_server.py engine protocol ----------
 * stdin:  SUBMIT <id> <slot> <len> <max_tokens> <temp> <top_p>\n<payload>\n
 *         CANCEL <id>\n
 * stdout: READY sentinel once loaded, then per request a stream of
 *         DATA <id> <size>\n<bytes>\n frames and a final
 *         DONE <id> STAT <tok> <tps> <hit%> <rss> <prompt_tok> <len_limited>\n
 * Byte-identical to glm52.c's serve protocol (inkling.c documents it in
 * full above its own SUBMIT handling) so the shared openai_server.py gateway
 * drives olmoe unchanged.
 *
 * v1 scope, same as Inkling's first serve mode: one request in flight, full
 * re-prefill every turn, no cross-request KV reuse. The payload arrives
 * already rendered by openai_server.py's render_chat_olmoe (bos/eos turn
 * markers and all) -- this engine tokenizes it as-is, the same way run_chat()
 * feeds fmt_user_turn()'s output to tok_encode() above. Because nothing here
 * persists state across requests, a fresh prefill at pos_base=0 is enough to
 * start clean: attention() only ever reads positions [0, kv_len), so the
 * previous request's leftover K/V contents past the new prompt's length are
 * never touched, the same invariant CHAT mode's /reset already relies on
 * (it clears hist_len, not the K/V buffers themselves). */

typedef struct { char id[64]; int max_tok; float temp, top_p; char *payload; int plen;
                 int logprobs, pin; } SReq;   /* SUBMIT logprobs=k / pin=1 */
#define SRV_QMAX 16
static SReq g_q[SRV_QMAX]; static int g_qn = 0;
static const LmaServeWireProfile olmoe_wire = {
    .max_header_bytes = 511,
    .max_payload_bytes = 1u << 22,
    .max_tokens = 1 << 20,
    .require_exact_lf = 1,
    .require_finite_sampling = 0,
};

/* read one control line (+ payload for SUBMIT). cur_id: request in flight;
 * returns 1 if that request was cancelled, 0 otherwise, -1 on input EOF. */
static int serve_read_cmd(FILE *in, FILE *out, const char *cur_id) {
    LmaServeCommand command;
    LmaServeReadResult result = lma_serve_read_command(in, &olmoe_wire, &command);
    if (result == LMA_SERVE_READ_EOF || result == LMA_SERVE_READ_BAD_FRAME) return -1;
    if (result == LMA_SERVE_READ_NOMEM) {
        lma_serve_write_error(out, command.id, "out of memory");
        return -1;
    }
    if (result == LMA_SERVE_READ_BAD_REQUEST &&
        command.kind == LMA_SERVE_COMMAND_SUBMIT) {
        lma_serve_write_error(out, command.id, "bad submit header");
        return -1;
    }
    if (result != LMA_SERVE_READ_OK) return 0;
    if (command.kind == LMA_SERVE_COMMAND_CANCEL) {
        int cancelled = cur_id && !strcmp(command.id, cur_id);
        lma_serve_command_dispose(&command);
        return cancelled;
    }
    if (command.kind == LMA_SERVE_COMMAND_SUBMIT) {
        if (g_qn < SRV_QMAX) {
            SReq *q = &g_q[g_qn++];
            snprintf(q->id, sizeof(q->id), "%s", command.id);
            q->max_tok = command.max_tokens;
            q->logprobs = command.logprobs;
            q->pin = command.pin;
            q->temp = command.temperature;
            q->top_p = command.top_p;
            q->payload = (char *)lma_serve_command_take_payload(&command);
            q->plen = (int)command.payload_bytes;
        } else {
            lma_serve_write_error(out, command.id, "queue full");
        }
    }
    lma_serve_command_dispose(&command);
    return 0;
}

/* HITS rows cols hex: which experts this turn routed, one bit each, every
 * layer (all are MoE here, same rows and columns as EMAP), packed 8 per hex
 * pair. Same line glm52.c emits; the Brain tab lights up from it. */
static void serve_hits(Model *m) {
    Cfg *c = &m->c; int E = c->n_experts, rows = c->n_layers;
    if (!m->ehit) ehit_mark(m, -1, -1);
    int nb = (rows * E + 7) / 8;
    uint8_t *bm = calloc((size_t)nb, 1); int bit = 0;
    for (int i = 0; i < rows; i++)
        for (int e = 0; e < E; e++, bit++)
            if (m->ehit[i][e]) { bm[bit >> 3] |= (uint8_t)(1 << (bit & 7)); m->ehit[i][e] = 0; }
    char *hex = malloc((size_t)nb * 2 + 1); int w = 0;
    for (int b = 0; b < nb; b++) { hex[w++] = "0123456789abcdef"[bm[b] >> 4]; hex[w++] = "0123456789abcdef"[bm[b] & 15]; }
    hex[w] = 0;
    printf("HITS %d %d %s\n", rows, E, hex);
    fflush(stdout); free(hex); free(bm);
}

/* LMA_KV_PREFIX=0: never reuse a previous turn's cache. The escape hatch,
 * and the B arm of the A/B that shows reuse changes nothing but the time. */
static int kv_prefix_off(void){ const char *e = getenv("LMA_KV_PREFIX"); return e && *e == '0'; }

/* ---- Speculative decoding (greedy only): prompt-lookup n-gram drafts + batched GPU verification ----
 * The draft is free (a suffix n-gram match in the prompt+generated history); the k+1 tokens [current, d1..dk] are
 * verified in ONE batched forward (weights read once per 8 tokens). Acceptance is exact: d_j is kept only when it
 * equals argmax of the previous row, and the batched kernels accumulate in the same order as the decode kernels, so the
 * emitted tokens are those plain greedy decoding would produce. Everything is measured end to end; a spec window that
 * is slower per token than plain GPU decode switches speculation off for the session (circuit breaker). */
static uint64_t g_spec_calls, g_spec_drafted, g_spec_accepted, g_spec_nodraft, g_spec_plain;
static double g_spec_s, g_spec_plain_s;   /* wall seconds: verify calls / plain decode steps taken while spec was armed */
static int g_spec_off = 0;
static double g_spec_win_s; int g_spec_win_calls, g_spec_win_tok;

static int spec_draft(const int *hist, int hist_len, int kmax, int *out) {
    for (int n = 4; n >= g_spec_ngram; n--) {
        if (hist_len <= n) continue;
        for (int j = hist_len - n - 1; j >= 0; j--) {
            int eq = 1;
            for (int t = 0; t < n; t++) if (hist[j + t] != hist[hist_len - n + t]) { eq = 0; break; }
            if (!eq) continue;
            int k = 0;
            while (k < kmax && j + n + k < hist_len - 0 && j + n + k < hist_len) { out[k] = hist[j + n + k]; k++; }
            if (k > 0) return k;
        }
    }
    return 0;
}

/* One token step with speculation. `hist[0..hist_len)` holds prompt + generated tokens, the last being `nt` (position
 * hist_len-1, not yet run). Returns the logits after `nt` (like step()); accepted draft rows are left in the rows buffer (count in nq) for the
 * caller to consume one per following token. NULL = caller must run step(). */
static float *spec_step(Model *m, const int *hist, int hist_len, int ctx_cap, float *rows, int *nq) {
    Cfg *c = &m->c; int V = c->vocab;
    *nq = 0;
    if (g_spec_off || !m->gpu_on || !g_oc.pf_begin || g_spec_k < 1) return NULL;
    int pos = hist_len - 1;
    int kmax = g_spec_k;
    if (kmax > 7) kmax = 7;
    if (pos + 1 + kmax > m->gpu_max_t - 1) kmax = m->gpu_max_t - 1 - pos - 1;
    if (pos + 1 + kmax > ctx_cap - 1) kmax = ctx_cap - 1 - pos - 1;
    if (kmax < 1) return NULL;
    int T[8], k = spec_draft(hist, hist_len, kmax, T + 1);
    if (k < 1) { g_spec_nodraft++; return NULL; }
    T[0] = hist[hist_len - 1];
    double t0 = now_s();
    if (!gpu_prefill(m, T, k + 1, pos, rows, k + 1, 1)) {
        m->gpu_on = 0; fprintf(stderr, "[GPU] speculative verification failed at position %d: falling back to the CPU\n", pos);
        return NULL;
    }
    for (int r = 0; r <= k; r++)
        for (int i = 0; i < V; i++)
            if (!isfinite(rows[(size_t)r * V + i])) {
                m->gpu_on = 0; snprintf(m->gpu_trip_reason, sizeof m->gpu_trip_reason, "non-finite logits in speculative verification");
                fprintf(stderr, "[GPU] %s: falling back to the CPU\n", m->gpu_trip_reason);
                return NULL;
            }
    int a = 0;
    while (a < k) {                                  /* accept d_{a+1} iff it is the argmax of row a */
        const float *rw = rows + (size_t)a * V; int am = 0;
        for (int i = 1; i < V; i++) if (rw[i] > rw[am]) am = i;
        if (am != T[a + 1]) break;
        a++;
    }
    m->gpu_kv_len = pos + a + 1;
    m->token_count += a + 1; m->freq_token_count += a + 1;
    m->kv_len = pos + a + 1;
    kv_prefix_record(&m->kvp, T, pos, a + 1);
    g_prof_forwards += 1;
    double dt = now_s() - t0;
    g_spec_calls++; g_spec_drafted += (uint64_t)k; g_spec_accepted += (uint64_t)a; g_spec_s += dt;
    g_spec_win_s += dt; g_spec_win_calls++; g_spec_win_tok += a + 1;
    if (g_spec_win_calls >= 16) {                    /* breaker: per-token cost must beat plain GPU decode */
        double per_tok = g_spec_win_s / g_spec_win_tok;
        double plain = g_spec_plain ? g_spec_plain_s / (double)g_spec_plain : 0.0;
        if (plain > 0 && per_tok > plain) {
            g_spec_off = 1;
            fprintf(stderr, "[SPEC] circuit breaker: %.2f ms/token with speculation > %.2f ms/token plain decode; speculation off\n",
                    per_tok * 1e3, plain * 1e3);
        }
        g_spec_win_s = 0; g_spec_win_calls = 0; g_spec_win_tok = 0;
    }
    *nq = a;
    float *lg = falloc(V);
    memcpy(lg, rows, (size_t)V * sizeof(float));
    return lg;
}

/* ---- Bottleneck diagnostics (counter-based: measured counters vs measured ceilings, nothing modelled) ---- */
static double weight_bytes_tok(const Cfg *c) {   /* weight bytes one decode token reads (FP16 dense + router, K int8 experts, lm_head) */
    return (double)c->n_layers * (4.0 * c->hidden * c->hidden * 2 + (double)c->n_experts * c->hidden * 4 +
           (double)c->topk * (3.0 * c->inter * c->hidden + (2.0 * c->inter + c->hidden) * 4)) + (double)c->vocab * c->hidden * 2;
}
static double ram_ceiling_gbs(void) {   /* multi-threaded memcpy traffic (read+write bytes), measured once */
    static double v = 0;
    if (v > 0) return v;
    size_t n = (size_t)128 << 20; char *a = malloc(n), *b = malloc(n);
    if (!a || !b) { free(a); free(b); return 0; }
    memset(a, 1, n); memset(b, 2, n);
    int nt = 8; size_t chunk = n / nt; double best = 0;
    for (int r = 0; r < 4; r++) {
        double t = now_s();
        #pragma omp parallel for num_threads(nt)
        for (int k = 0; k < nt; k++) memcpy(b + (size_t)k * chunk, a + (size_t)k * chunk, chunk);
        double bw = 2.0 * n / (now_s() - t) / 1e9; if (bw > best) best = bw;
    }
    free(a); free(b);
    return v = best;
}
static void emit_diag(Model *m, int gen, double decode_s, uint64_t g_tok, double g_s, uint64_t g_miss, uint64_t disk_b) {
    if (gen < 2 || decode_s <= 0) return;
    const double target = 30.0;
    double tps = (gen - 1) / decode_s, bt = weight_bytes_tok(&m->c);
    double pe = 3.0 * m->c.inter * m->c.hidden + (2.0 * m->c.inter + m->c.hidden) * 4;
    int gpu = g_tok > 0 && g_s > 0;
    double gtps = gpu ? g_tok / g_s : tps, meas = bt * gtps / 1e9, req = bt * target / 1e9;
    double ceil_ = gpu ? m->gpu_peak_d2d_gbs : ram_ceiling_gbs();
    double mpt = gpu ? (double)g_miss / g_tok : 0.0, util = ceil_ > 0 ? meas / ceil_ : 0.0;
    const char *cls;
    if (gpu && decode_s > 0 && g_s < 0.75 * decode_s * ((double)g_tok / (gen - 1 > 0 ? gen - 1 : 1)))
        cls = "host-side work outside the GPU step (sampling, protocol, sync)";
    else if (gpu && mpt > 0.5) { cls = "expert upload (disk/RAM -> VRAM)"; meas = mpt * pe * gtps / 1e9; ceil_ = m->gpu_peak_h2d_gbs; req = mpt * pe * target / 1e9; util = ceil_ > 0 ? meas / ceil_ : 0; }
    else if (util > 0.5) cls = gpu ? "VRAM bandwidth" : "RAM bandwidth";
    else if (!gpu && (double)disk_b / gen > 1e6) cls = "expert disk I/O";
    else cls = gpu ? "GPU kernel latency (not bandwidth-bound)" : "CPU compute / latency (not bandwidth-bound)";
    fprintf(stderr, "[LMA_DIAG] path=%s decode_tok_s=%.1f target=%.0f %s bottleneck=\"%s\" measured_gbs=%.1f ceiling_gbs=%.1f "
                    "required_for_target_gbs=%.1f util=%.0f%% weight_mb_per_token=%.0f misses_per_token=%.2f disk_mb_per_token=%.3f trip=\"%s\"\n",
            gpu ? "gpu" : "cpu", tps, target, tps >= target ? "MET" : "BELOW", cls, meas, ceil_, req, util * 100.0, bt / 1e6, mpt,
            (double)disk_b / gen / 1e6, m->gpu_trip_reason);
}

static int serve_one(Model *m, Tok *T, SReq *q, int ctx_cap) {
    Cfg *c = &m->c;
    int cap = q->plen + 16;
    int *ids = malloc((size_t)cap * sizeof(int));
    int np = tok_encode(T, q->payload, q->plen, ids, cap);
    if (np <= 0) { lma_serve_write_error(stdout, q->id, "empty prompt"); free(ids); return 0; }
    int budget = lma_serve_budget(np, q->max_tok, ctx_cap, q->logprobs > 0);
    if (budget < 0) {
        char message[128];
        /* The frame the gateway turns into a 400 context_length_exceeded
         * (#506, #1381). Free text here reached the client as a 500. */
        snprintf(message, sizeof(message),
                 "CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d",
                 np, q->max_tok, ctx_cap);
        lma_serve_write_error(stdout, q->id, message); free(ids); return 0;
    }
    if (budget < q->max_tok) {
        fprintf(stderr, "[serve] max_tokens %d clamped to %d (context %d - prompt %d); "
                        "raise CTX for longer answers\n",
                q->max_tok, budget, ctx_cap, np);
        q->max_tok = budget;
    }
    g_temp = q->temp; g_nuc = q->top_p;
    /* A chat client resends the whole transcript every turn. If this prompt
     * begins with the ids the cache was built from, those keys and values ARE
     * the state at those positions -- attention is causal, so a row depends on
     * its prefix and nothing later. Prefill only the tail. Reuse is all or
     * nothing: nothing here can rewind a cache. LMA_KV_PREFIX=0 turns it off,
     * LMA_PREFIX_LOG=1 reports the decision and its reason, because "it did
     * not get faster" is otherwise indistinguishable from "it is not wired up".
     *
     * On a miss the record must be CLEARED and not merely overwritten: a
     * shorter prompt writes fewer positions than the last one recorded, and
     * kv_prefix_record only ever grows the length, so the stale tail would
     * claim coverage the cache no longer has. */
    int reuse = kv_prefix_off() ? 0 : kv_prefix_reuse(&m->kvp, ids, np);
    /* La fotografia si prova sempre, non solo quando il riuso vivo fallisce:
     * altrimenti la prima opzione trova ancora lo stato del prompt, passa dal
     * riuso normale e il suo primo token resta senza predittore. */
    g_pin_use_logit = 0; g_pin_logit = NULL;
    {
        /* Il piu profondo degli scatti che sia un prefisso di questo prompt.
         * Con due livelli (istruzioni, istruzioni+domanda) e il secondo a
         * decidere; se e' morto si ripiega sul primo invece di rifare tutto. */
        int s = lma_pin_best(&g_pins, ids, np);
        while (s >= 0) {
            LmaPin *k = &g_pins.slot[s];
            if (kv_prefix_holds(&m->kvp, k->ids, k->len)) {
                kv_prefix_clear(&m->kvp);
                kv_prefix_record(&m->kvp, k->ids, 0, k->len);
                m->kv_len = k->len;
                reuse = k->len;
                g_pin_logit = k->logit; g_pin_use_logit = k->logit != NULL;
                lma_pin_touch(&g_pins, s);
                break;
            }
            k->len = 0;                    /* le righe non ci sono piu: scatto orfano */
            s = lma_pin_best(&g_pins, ids, np);
        }
    }
    g_echo_k = q->logprobs; g_echo_id = q->id; g_echo_tok = T;
    if (!reuse) kv_prefix_clear(&m->kvp);
    if (getenv("LMA_PREFIX_LOG")) {
        if (reuse)
            fprintf(stderr, "[PREFIX] reusing %d of %d prompt tokens (%.0f%%)\n",
                    reuse, np, 100.0 * reuse / np);
        else
            fprintf(stderr, "[PREFIX] no reuse: held=%d cap=%d prompt=%d\n",
                    m->kvp.len, m->kvp.cap, np);
        fflush(stderr);
    }
    double t0 = now_s();
    uint64_t h0 = m->hits, m0 = m->miss;
    uint64_t disk0 = __atomic_load_n(&m->disk_ns, __ATOMIC_RELAXED);
    uint64_t bytes0 = __atomic_load_n(&m->disk_bytes, __ATOMIC_RELAXED);
    uint64_t pfi0 = m->pf_issued, pfu0 = m->pf_used;
    double attn0 = g_prof_attn_s, moe0 = g_prof_moe_s, head0 = g_prof_head_s;
    long long fwd0 = g_prof_forwards;
    /* `reuse` is the ABSOLUTE position of the first fresh token: attention
     * and the KV rows are position-indexed, so this has to be the real
     * offset, not a count of what is left to do. */
    uint64_t gtok0 = m->gpu_tokens, gmiss0 = m->gpu_miss; double gs0 = m->gpu_s;
    float *logit = step(m, ids + reuse, np - reuse, reuse);
    double t_pf = now_s() - t0;
    int hist_len = np, gen = 0, limited = 1, cancelled = 0;
    char buf[512];
    /* speculative decoding: greedy, no logprob channel, no pin snapshot */
    int use_spec = g_spec_k > 0 && !g_spec_off && m->gpu_on && g_oc.pf_begin && q->temp <= 0 && q->logprobs <= 0 && !q->pin;
    int *shist = NULL, sq_n = 0, sq_i = 0; float *srows = NULL;
    if (use_spec) {
        shist = malloc((size_t)(np + q->max_tok + 16) * sizeof(int));
        srows = malloc((size_t)8 * c->vocab * sizeof(float));
        if (!shist || !srows) { free(shist); free(srows); shist = NULL; srows = NULL; use_spec = 0; }
        else memcpy(shist, ids, (size_t)np * sizeof(int));
    }
    if (q->pin && logit) {
        lma_pin_pool_init(&g_pins, c->vocab);
        if (lma_pin_store(&g_pins, ids, np, logit)) {
            fprintf(stderr, "[PIN] scatto a %d token\n", np); fflush(stderr);
        }
    }
    g_echo_k = 0; g_echo_id = NULL;   /* la lettura riguarda il prefill, non la decodifica */
    for (int s = 0; s < q->max_tok && !cancelled; s++) {
        int nt = pick_tok(logit, c->vocab, -1);
        char lptail[1024]; lptail[0] = 0;
        if (q->logprobs > 0) lma_logprob_tail(lptail, sizeof lptail, logit, c->vocab, nt, q->logprobs);
        free(logit); logit = NULL;
        if (is_stop(nt)) { limited = 0; break; }
        int nb = tok_decode(T, &nt, 1, buf, sizeof(buf)-1);
        if (q->logprobs > 0) lma_serve_write_data_lp(stdout, q->id, buf, (size_t)nb, lptail);
        else lma_serve_write_data(stdout, q->id, buf, (size_t)nb);
        gen++; hist_len++;
        if (shist) shist[hist_len - 1] = nt;
        while (lma_stdin_readable()) {
            int r = serve_read_cmd(stdin, stdout, q->id);
            if (r < 0) { free(ids); return -1; }
            if (r > 0) { cancelled = 1; limited = 0; }
        }
        /* Unlike run_chat(), we do not step() the final token just to populate
         * its KV slot: nothing in serve mode reads past this request's own
         * reply, so a discarded logit here costs nothing. */
        if (cancelled || s == q->max_tok - 1 || hist_len >= ctx_cap) break;
        if (use_spec && sq_i < sq_n) {               /* row already computed by the last batched verification */
            logit = falloc(c->vocab);
            memcpy(logit, srows + (size_t)(sq_i + 1) * c->vocab, (size_t)c->vocab * sizeof(float));
            sq_i++;
            continue;
        }
        if (use_spec) {
            sq_n = 0; sq_i = 0;
            logit = spec_step(m, shist, hist_len, ctx_cap, srows, &sq_n);
            if (logit) continue;
            double ts = now_s();
            logit = step(m, &nt, 1, hist_len - 1);
            if (!g_spec_off && m->gpu_on) { g_spec_plain++; g_spec_plain_s += now_s() - ts; }
            continue;
        }
        logit = step(m, &nt, 1, hist_len - 1);
    }
    free(logit); free(shist); free(srows);
    double dt = now_s() - t0;
    double tot = (double)(m->hits - h0 + m->miss - m0);
    LmaServeDone done = {
        .completion_tokens = gen,
        .tokens_per_second = dt > 0 ? gen/dt : 0.0,
        .cache_hit_percent = tot ? 100.0*(m->hits-h0)/tot : 0.0,
        .rss_gb = rss_gb(),
        .prompt_tokens = np,
        .length_limited = limited,
    };
    lma_serve_write_done(stdout, q->id, &done);
    /* PROF: per-turn phase timings for the dashboard, field order the protocol's:
     * disk, wait, matmul, attention, lm_head, forwards. disk is what the expert
     * loads took (#1449 first half); matmul is the MoE block time minus that
     * disk time, clamped at zero (the PILOT worker's reads are counted in disk
     * but happen off the block's clock); attention and lm_head are measured
     * around their calls; forwards counts positions through step(). wait stays
     * zero: this engine has no separate wait phase. */
    double disk_s = (double)(__atomic_load_n(&m->disk_ns, __ATOMIC_RELAXED) - disk0) / 1e9;
    double matmul_s = (g_prof_moe_s - moe0) - disk_s; if (matmul_s < 0.0) matmul_s = 0.0;
    if (getenv("LMA_BYTES_LOG") && m->gpu_n_slots) {
        fprintf(stderr, "[LMA_GPU] on=%d tokens=%llu expert_misses=%llu fetch_syncs=%llu gpu_s=%.3f "
                        "verify_n=%llu verify_argmax_agree=%llu verify_max_dlogit=%.5f\n",
                m->gpu_on, (unsigned long long)m->gpu_tokens, (unsigned long long)m->gpu_miss,
                (unsigned long long)m->gpu_fetch_syncs, m->gpu_s, (unsigned long long)m->gpu_verify_n,
                (unsigned long long)m->gpu_verify_agree, m->gpu_max_dlogit);
        fprintf(stderr, "[LMA_GPU_PF] calls=%llu rows=%llu gpu_s=%.3f verify_n=%llu verify_argmax_agree=%llu verify_max_dlogit=%.5f\n",
                (unsigned long long)m->gpu_pf_calls, (unsigned long long)m->gpu_pf_rows, m->gpu_pf_s,
                (unsigned long long)m->gpu_pf_verify_n, (unsigned long long)m->gpu_pf_verify_agree, m->gpu_pf_max_dlogit);
        if (g_oc.prof_get) {
            double ms[8]; long tk = 0;
            if (g_oc.prof_get(ms, &tk) > 0 && tk > 0)
                fprintf(stderr, "[LMA_GPU_PROF] per token over %ld tokens (ms): dense_gemv=%.3f attn=%.3f route=%.3f "
                                "expert_gateup=%.3f expert_down=%.3f head=%.3f\n", tk, ms[0]/tk, ms[1]/tk, ms[2]/tk, ms[3]/tk, ms[4]/tk, ms[5]/tk);
        }
    }
    if (g_spec_k > 0 && getenv("LMA_BYTES_LOG"))
        fprintf(stderr, "[LMA_SPEC] k=%d ngram=%d off=%d verify_calls=%llu drafted=%llu accepted=%llu accept_rate=%.3f "
                        "tokens_per_verify=%.3f nodraft_steps=%llu plain_steps=%llu verify_s=%.3f plain_s=%.3f\n",
                g_spec_k, g_spec_ngram, g_spec_off, (unsigned long long)g_spec_calls, (unsigned long long)g_spec_drafted,
                (unsigned long long)g_spec_accepted, g_spec_drafted ? (double)g_spec_accepted / g_spec_drafted : 0.0,
                g_spec_calls ? 1.0 + (double)g_spec_accepted / g_spec_calls : 0.0, (unsigned long long)g_spec_nodraft,
                (unsigned long long)g_spec_plain, g_spec_s, g_spec_plain_s);
    emit_diag(m, gen, dt - t_pf, m->gpu_tokens - gtok0, m->gpu_s - gs0, m->gpu_miss - gmiss0,
              __atomic_load_n(&m->disk_bytes, __ATOMIC_RELAXED) - bytes0);
    if (getenv("LMA_BYTES_LOG") && gen > 0) {
        uint64_t db = __atomic_load_n(&m->disk_bytes, __ATOMIC_RELAXED) - bytes0;
        uint64_t pfi = m->pf_issued - pfi0, pfu = m->pf_used - pfu0;
        fprintf(stderr, "[LMA_STAT] req=%s tokens=%d disk_bytes=%llu gb_per_token=%.6f "
                        "pf_issued=%llu pf_used=%llu pf_precision=%.4f tok_s=%.4f\n",
                q->id, gen, (unsigned long long)db, (double)db / gen / 1e9,
                (unsigned long long)pfi, (unsigned long long)pfu,
                pfi ? (double)pfu / pfi : 0.0, gen / dt);
    }
    /* microsecond resolution: see qwen36.c, same reason (a sub-millisecond tiny turn read as unmeasured) */
    printf("PROF %.6f %d %d %.6f 0.0 %.6f %.6f %.6f %lld\n", dt, np, gen, disk_s, matmul_s,
           g_prof_attn_s - attn0, g_prof_head_s - head0, g_prof_forwards - fwd0);
    fflush(stdout);
    serve_hits(m);
    free(ids);
    return 0;
}

/* dashboard HWINFO/TIERS/EMAP: same lines the other serve-capable engines
 * emit for the web dashboard's hardware panel and Brain page. olmoe.c is
 * CPU-only (no CUDA/Metal backend), so the GPU fields are always empty, and
 * every layer is a MoE layer (no dense/sparse split like GLM-5.2), so the
 * tier scan below runs over all n_layers unconditionally. */
static void serve_hwinfo(Model *m) {
    (void)m;
    char cpu[256] = ""; int cores = 0; double rt = 0, ra = 0;
    FILE *ci = fopen("/proc/cpuinfo", "r");
    if (ci) { char ln[256];
        while (fgets(ln, sizeof(ln), ci)) if (!strncmp(ln, "model name", 10)) {
            char *p = strchr(ln, ':'); if (p) { p++; while (*p == ' ') p++;
            int n = (int)strlen(p); if (n > 0 && p[n-1] == '\n') p[--n] = 0;
            snprintf(cpu, sizeof(cpu), "%s", p); } break; }
        fclose(ci); }
#ifdef _SC_NPROCESSORS_ONLN
    cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    FILE *mi = fopen("/proc/meminfo", "r");
    if (mi) { char ln[256]; double v = 0;
        while (fgets(ln, sizeof(ln), mi)) {
            if (sscanf(ln, "MemTotal: %lf", &v) == 1) rt = v/1e6;
            if (sscanf(ln, "MemAvailable: %lf", &v) == 1) ra = v/1e6;
        } fclose(mi); }
    if (ra <= 0.0) ra = compat_mem_available_gb();   /* macOS, Windows: no /proc (#1500) */
#ifdef _WIN32
    if (rt <= 0.0) { double a2 = 0.0; compat_meminfo(&rt, &a2); }
#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    if (rt <= 0.0) { long pg = sysconf(_SC_PHYS_PAGES), ps = sysconf(_SC_PAGESIZE); if (pg > 0 && ps > 0) rt = (double)pg * ps / 1e9; }
#endif
    printf("HWINFO %d %.1f %.1f 0 0.0 %s|\n", cores, rt, ra, cpu[0] ? cpu : "unknown");
    fflush(stdout);
}

static void serve_tiers_emap(Model *m) {
    Cfg *c = &m->c; int E = c->n_experts;
    int filled = 0;
    for (int i = 0; i < c->n_layers; i++) filled += m->cache[i].n;
    int64_t I = c->inter, D = c->hidden;
    /* per-expert resident bytes: int8 gate/up/down + one f32 scale per row */
    int64_t slotb = 3*I*D + (2*I+D)*4;
    printf("TIERS 0 %d %d 0.00 %.2f\n", filled, c->n_layers*E - filled, filled*(double)slotb/1e9);
    /* EMAP: 1 byte/expert hex — tier(2b: 0=disk 1=RAM)<<6 | heat(6b: log2 usage) */
    char *hex = malloc((size_t)c->n_layers*E*2 + 1); int w = 0;
    for (int i = 0; i < c->n_layers; i++) {
        LCache *lc = &m->cache[i];
        for (int e = 0; e < E; e++) {
            int tier = 0;
            for (int z = 0; z < lc->n; z++) if (lc->slots[z].eid == e) { tier = 1; break; }
            uint32_t u = m->freq[i] ? m->freq[i][e] : 0;
            int heat = 0; while (u) { heat++; u >>= 1; } if (heat > 63) heat = 63;
            int b = (tier << 6) | heat;
            hex[w++] = "0123456789abcdef"[b >> 4];
            hex[w++] = "0123456789abcdef"[b & 15];
        }
    }
    hex[w] = 0;
    printf("EMAP %d %d %s\n", c->n_layers, E, hex);
    fflush(stdout); free(hex);
}

static void serve_loop(Model *m, Tok *T, int ctx_cap) {
    lma_serve_stdio_init();
    int tok_eos = tok_id_of(T, "|||IP_ADDRESS|||");
    stops_arm_tok(&m->c, tok_eos, T);
    lma_serve_write_ready(stdout, rss_gb());
    serve_hwinfo(m);
    serve_tiers_emap(m);
    for (;;) {
        while (!g_qn) if (serve_read_cmd(stdin, stdout, NULL) < 0) return;
        SReq q = g_q[0];
        memmove(g_q, g_q + 1, (size_t)(--g_qn) * sizeof(SReq));
        int fatal = serve_one(m, T, &q, ctx_cap);
        free(q.payload);
        if (fatal < 0) return;
        /* Resend the grid after EVERY turn, not only after READY: at boot the
         * expert cache is empty by definition, and that cold snapshot stayed
         * the only one the dashboard ever saw -- all grey, RAM 0, everything
         * on disk, forever. HITS was already per turn, which is why the white
         * "routed now" flash worked while the residency colour never moved.
         * inkling.c, kimi_k3.c, qwen38.c, deepseek_v41.c and glm52.c
         * already do this. */
        serve_tiers_emap(m);
    }
}

/* ---------- lettura ref.json ---------- */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

#ifndef OLMOE_NO_MAIN
int main(int argc, char **argv) {
    lma_omp_tune_threads("olmoe");   /* squadra sui core fisici, niente spin-wait: vedi omp_tune.h */
    const char *snap = getenv("SNAP");
    if (!snap) { lma_print_launcher_help("OLMoE"); return 1; }
    g_pilot = getenv("PILOT") ? atoi(getenv("PILOT")) : 0;
    g_wide  = getenv("WIDE")  ? atoi(getenv("WIDE"))  : 1;
    g_pilot_evict_guard = getenv("PILOT_EVICT_GUARD") ? atoi(getenv("PILOT_EVICT_GUARD")) : 1;
    g_expert_drop = getenv("EXPERT_DROP") ? atoi(getenv("EXPERT_DROP")) : 0;
    g_fused3     = getenv("FUSED3") ? atoi(getenv("FUSED3")) : 0;
    g_dense_f16 = getenv("LMA_DENSE_F16") ? atoi(getenv("LMA_DENSE_F16")) : 1;
    g_giant_stick = getenv("LMA_GIANT_STICK") ? (float)atof(getenv("LMA_GIANT_STICK")) : 0.0f;
    if (g_giant_stick < 0.f) g_giant_stick = 0.f; if (g_giant_stick > 1.f) g_giant_stick = 1.f;
    if (g_wide < 1) g_wide = 1;
    if (g_wide > 4) g_wide = 4;
    int hot_n  = getenv("HOT")   ? atoi(getenv("HOT"))   : 0;
    int cap    = argc > 1 ? lma_arg_int(argv[1], "cache/layer") : 16;
    int bits   = argc > 2 ? lma_arg_int(argv[2], "expert bits") : 8;
    if (bits < 2 || bits > 8) {
        fprintf(stderr, "quant_bits must be 2..8 (got %d)\n", bits);
        return 1;
    }

    /* QENCODE=<file>: tokenizer probe for the Quality Governor. The file holds text segments separated by 0x01
     * bytes; prints one line of space-separated token ids per segment (the engine's own tokenizer, so the ids are
     * exactly what serve mode would feed) and exits. */
    if (getenv("QENCODE") && *getenv("QENCODE")) {
        Tok T; char tp[2048]; snprintf(tp, sizeof tp, "%s/tokenizer.json", snap); tok_load(&T, tp);
        FILE *qf = fopen(getenv("QENCODE"), "rb"); if (!qf) { perror("QENCODE"); return 1; }
        fseek(qf, 0, SEEK_END); long qn = ftell(qf); fseek(qf, 0, SEEK_SET);
        char *qb = malloc((size_t)qn + 1); if (fread(qb, 1, (size_t)qn, qf) != (size_t)qn) { fclose(qf); return 1; } qb[qn] = 0; fclose(qf);
        int *qi = malloc(65536 * sizeof(int)); long a = 0;
        for (long i = 0; i <= qn; i++) {
            if (i == qn || qb[i] == 1) {
                int n = tok_encode(&T, qb + a, (int)(i - a), qi, 65536);
                for (int k = 0; k < n; k++) printf(k ? " %d" : "%d", qi[k]);
                printf("\n"); a = i + 1;
            }
        }
        return 0;
    }

    if (getenv("QCOMPILE") && *getenv("QCOMPILE") && getenv("QC_OUT"))
        return qcompile(snap, getenv("QCOMPILE"), getenv("QC_OUT"));

    /* SERVE=1: openai_server.py drives the engine over stdin/stdout (READY
     * handshake, SUBMIT/CANCEL, DATA/DONE/PROF frames, HWINFO/TIERS/EMAP for
     * the dashboard) — same protocol as glm52.c/inkling.c/kimi_k3.c. v1:
     * one request at a time, full re-prefill every turn (see serve_one()). */
    if (getenv("SERVE") && getenv("SERVE")[0] == '1') {
        int ctx_cap = getenv("CTX") ? atoi(getenv("CTX")) : 4096;
        if (ctx_cap < 1 || ctx_cap > 4096) {   /* attention()'s sc[4096] score buffer hard-caps this */
            fprintf(stderr, "CTX must be 1..4096 (got %d)\n", ctx_cap);
            return 1;
        }
    /* static, not a stack local: the PILOT prefetch worker is detached and
     * loops forever, and it keeps this address in the global pilot_m. A stack
     * Model dies when main returns while that thread is still dereferencing
     * it -- ASan: stack-use-after-return, READ of size 8, in a worker thread,
     * with the run's tokens already correct (#1262). Static storage outlives
     * every thread, so the pointer the worker holds stays valid. */
        static Model m; model_init(&m, snap, cap, bits); gpu_try_init(&m);
        m.max_t = ctx_cap;
        m.K = calloc(m.c.n_layers, sizeof(float*)); m.V = calloc(m.c.n_layers, sizeof(float*));
        for (int i = 0; i < m.c.n_layers; i++) {
            m.K[i] = falloc((int64_t)m.c.n_heads * m.max_t * m.c.head_dim);
            m.V[i] = falloc((int64_t)m.c.n_heads * m.max_t * m.c.head_dim);
        }
        /* Serve mode only: the record is sized with the cache it describes, and
         * the cache here is allocated once for ctx_cap and never reallocated,
         * so a recorded position stays valid for the life of the process. The
         * CLI paths (generate/run_chat/PPL) leave it unallocated, which makes
         * every kv_prefix call there a no-op. */
        kv_prefix_alloc(&m.kvp, m.max_t);
        Tok T;
        char tokpath[2048]; snprintf(tokpath, sizeof(tokpath), "%s/tokenizer.json", snap);
        tok_load(&T, tokpath);
        lma_rt_term_arm();   /* SIGTERM must reach the save below (#1629) */
        serve_loop(&m, &T, ctx_cap);
        { const char *up = getenv("LMA_USAGE");
          if (up && *up) rt_save(up, 0); }
        return 0;
    }

    if (getenv("CHAT")) {   /* interactive mode: bypasses the ref.json harness entirely */
        /* #509 convention, same as the GLM engine: LMA_TEMP is the primary channel;
         * TEMP stays a legacy alias ONLY when it is fully numeric — on Windows and under
         * ROCm stacks %TEMP% names a directory, and atof("C:\...") == 0.0 would silently
         * force greedy decoding for every olmoe chat on those hosts. */
        if (getenv("LMA_TEMP")) g_temp = (float)atof(getenv("LMA_TEMP"));
        else if (getenv("TEMP") && *getenv("TEMP")) {
            char *tend; double tv = strtod(getenv("TEMP"), &tend);
            if (tend != getenv("TEMP") && *tend == '\0') g_temp = (float)tv;
        }
        g_nuc  = getenv("NUCLEUS") ? (float)atof(getenv("NUCLEUS")) : g_nuc;
        int ctx_cap = getenv("CTX") ? atoi(getenv("CTX")) : 4096;
        if (ctx_cap < 1 || ctx_cap > 4096) {   /* attention()'s sc[4096] score buffer hard-caps this */
            fprintf(stderr, "CTX must be 1..4096 (got %d)\n", ctx_cap);
            return 1;
        }
    /* static, not a stack local: the PILOT prefetch worker is detached and
     * loops forever, and it keeps this address in the global pilot_m. A stack
     * Model dies when main returns while that thread is still dereferencing
     * it -- ASan: stack-use-after-return, READ of size 8, in a worker thread,
     * with the run's tokens already correct (#1262). Static storage outlives
     * every thread, so the pointer the worker holds stays valid. */
        static Model m; model_init(&m, snap, cap, bits); gpu_try_init(&m);
        printf("resident weights loaded in %.1fs | RSS after load: %.2f GB\n", m.dense_load_s, rss_gb());
        Tok T;
        char tokpath[2048]; snprintf(tokpath, sizeof(tokpath), "%s/tokenizer.json", snap);
        tok_load(&T, tokpath);
        run_chat(&m, &T, ctx_cap);
        { const char *up = getenv("LMA_USAGE");
          if (up && *up) rt_save(up, 0); }
        return 0;
    }

    const char *refpath = argc > 3 ? argv[3] : "ref.json";

    float smooth = getenv("SMOOTH") ? (float)atof(getenv("SMOOTH")) : 0.3f;
    float conf   = getenv("CONF_LIMIT") ? (float)atof(getenv("CONF_LIMIT")) : 0.92f;

    printf("== Streaming C engine v2.2 | cache=%d/layer bits=%d pilot=%d wide=%d guard=%d hot=%d smooth=%.2f conf=%.2f fused3=%d ==\n",
           cap, bits, g_pilot, g_wide, g_pilot_evict_guard, hot_n, smooth, conf, g_fused3);

    FILE *f = fopen(refpath, "rb"); if (!f) { perror(refpath); return 1; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if (fread(buf,1,n,f)!=(size_t)n) {} buf[n]=0; fclose(f);
    char *arena=NULL; jval *ref = json_parse(buf, &arena);
    int np, nfull; int *prompt = read_int_array(ref,"prompt_ids",&np); int *full = read_int_array(ref,"full_ids",&nfull);
    int n_new = nfull - np;

    /* static, not a stack local: the PILOT prefetch worker is detached and
     * loops forever, and it keeps this address in the global pilot_m. A stack
     * Model dies when main returns while that thread is still dereferencing
     * it -- ASan: stack-use-after-return, READ of size 8, in a worker thread,
     * with the run's tokens already correct (#1262). Static storage outlives
     * every thread, so the pointer the worker holds stays valid. */
    static Model m; model_init(&m, snap, cap, bits); gpu_try_init(&m);
    printf("resident weights loaded in %.1fs | RSS after load: %.2f GB\n", m.dense_load_s, rss_gb());

    if (getenv("PPL") && atoi(getenv("PPL")) == 1) {   /* loss-meter mode: teacher-forced NLL */
        double nll; double t = now_s();
        int scored = tf_nll(&m, full, nfull, np, &nll);
        double dt = now_s() - t;
        double tot = m.hits + m.miss;
        printf("TF-NLL: %.4f nats/token over %d tokens  |  ppl = %.2f\n", nll, scored, exp(nll));
        printf("Expert cache hit rate: %.1f%%  (hit=%llu miss=%llu)\n", tot?100.0*m.hits/tot:0.0,
               (unsigned long long)m.hits, (unsigned long long)m.miss);
        printf("Speed: %.2f tok/s (%.1fs for %d tokens) | PEAK RSS: %.2f GB\n", scored/dt, dt, scored, rss_gb());
        if (m.gpu_n_slots) {   /* GPU tiering counters for the giant emulator: what crossed disk/PCIe per token */
            double pe = 3.0 * m.c.inter * m.c.hidden + (2.0 * m.c.inter + m.c.hidden) * 4;
            uint64_t db = __atomic_load_n(&m.disk_bytes, __ATOMIC_RELAXED);
            printf("[GPU_STATS] on=%d slots=%d/%d tokens=%llu expert_misses=%llu miss_per_token=%.2f fetch_syncs=%llu "
                   "disk_gb_per_token=%.4f gpu_s=%.3f gpu_pf_s=%.3f expert_mb_int8=%.2f t_read=%.3f t_upl=%.3f t_sync=%.3f\n",
                   m.gpu_on, m.gpu_n_slots, m.c.n_layers * m.c.n_experts, (unsigned long long)m.gpu_tokens,
                   (unsigned long long)m.gpu_miss, m.gpu_tokens ? (double)m.gpu_miss / m.gpu_tokens : 0.0,
                   (unsigned long long)m.gpu_fetch_syncs, scored ? (double)db / scored / 1e9 : 0.0, m.gpu_s, m.gpu_pf_s, pe / 1e6, g_t_read, g_t_upl, g_t_sync);
        }
        free(buf); free(arena);
        return 0;      /* PPL is a measurement run: no rt_save on purpose, so a loss
                        * sweep cannot fold its own tokens into the persisted ranking */
    }

    int *out = malloc((np + n_new) * sizeof(int));
    double t = now_s();
    generate(&m, prompt, np, n_new, out);
    double dt = now_s() - t;

    int match = 0;
    printf("\nReference: ");  for (int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nC engine : ");  for (int i=np;i<nfull;i++) { printf("%d ", out[i]); if (out[i]==full[i]) match++; }
    printf("\nMatching tokens: %d/%d\n", match, n_new);
    double tot = m.hits + m.miss;
    printf("\nPEAK RSS: %.2f GB\n", rss_gb());
    printf("Expert cache hit rate: %.1f%%  (hit=%llu miss=%llu)\n", tot?100.0*m.hits/tot:0.0,
           (unsigned long long)m.hits, (unsigned long long)m.miss);


    // Persistent Hot Pinning: save dynamic pinning if newly created
    if (m.hot_pinned) {
        char pinpath[512];
        snprintf(pinpath, sizeof(pinpath), "%s/hot_pinned.bin", snap);
        FILE *pinf_chk = fopen(pinpath, "rb");
        if (!pinf_chk) {
            FILE *pinf_save = fopen(pinpath, "wb");
            if (pinf_save) {
                size_t expected_size = (size_t)m.c.n_layers * m.c.n_experts;
                fwrite(m.is_pinned, 1, expected_size, pinf_save);
                fclose(pinf_save);
                printf("[HOT] Saved persistent pinning to %s\n", pinpath);
            }
        } else {
            fclose(pinf_chk);
        }
    }

    { const char *up = getenv("LMA_USAGE");
      if (up && *up) rt_save(up, 0); }              /* same bytes as every other engine */
    printf("Speed: %.2f tok/s (%.1fs for %d tokens)\n", n_new/dt, dt, n_new);
    /* One line, every engine, one format: `lmatomic tune` sweeps scheduling knobs and
     * needs tokens-and-elapsed to compare candidates. Before this only lmatomic
     * emitted a parseable throughput line (REPLAY decode), so the tuner was
     * GLM-only and bannered the right model while launching the wrong engine
     * (#898). Printed to stdout, which is what autotune captures.
     * Tokens and seconds, not tok/s: the ratio is derived by the caller at full
     * precision (#852 -- two decimals of tok/s is one significant digit at the
     * rates this engine runs at). */
    printf("TUNE decode: %d tokens in %.3fs\n", n_new, dt);
    if (getenv("LMA_BYTES_LOG")) {
        uint64_t db = __atomic_load_n(&m.disk_bytes, __ATOMIC_RELAXED);
        double gbtok = n_new > 0 ? (double)db / n_new / 1e9 : 0.0;
        fprintf(stderr, "[LMA_STAT] disk_bytes=%llu gb_per_token=%.6f pf_issued=%llu pf_used=%llu "
                        "pf_precision=%.4f miss=%llu hit=%llu\n",
                (unsigned long long)db, gbtok,
                (unsigned long long)m.pf_issued, (unsigned long long)m.pf_used,
                m.pf_issued ? (double)m.pf_used / m.pf_issued : 0.0,
                (unsigned long long)m.miss, (unsigned long long)m.hits);
    }
    free(buf); free(arena);
    return 0;
}
#endif /* OLMOE_NO_MAIN */

#ifdef LMA_SEGMENT_ADAPTER
/* ---------- engine-owned Segment adapter ------------------------------ */

typedef struct {
    Model model;
    uint32_t layer_begin, layer_end, context_tokens;
    pthread_mutex_t run_lock;
} OlmoeSegmentEngine;

typedef struct {
    OlmoeSegmentEngine *engine;
    float **K, **V;
    uint32_t context_tokens, position;
} OlmoeSegmentSession;

static void olmoe_segment_model_destroy(OlmoeSegmentEngine *engine) {
    if (!engine) return;
    Model *model = &engine->model;
    for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
         layer++) {
        Layer *weights = &model->L[layer];
        free(weights->in_ln); free(weights->post_ln);
        free(weights->q); free(weights->k); free(weights->v); free(weights->o);
        free(weights->qn); free(weights->kn); free(weights->gate);
        LCache *cache = &model->cache[layer];
        for (int slot = 0; slot < cache->n; slot++) {
            free(cache->slots[slot].g);
            free(cache->slots[slot].gs);
        }
        free(cache->slot_by_expert);
        free(cache->slots);
    }
    free(model->last_access); free(model->is_queued); free(model->is_pinned);
    free(model->freq);
    free(model->momentum_logits);
    free(model->cache); free(model->L);
    st_destroy(&model->S);
}

static int olmoe_segment_engine_open(
    void **engine_impl, LmaSegmentCapabilities *capabilities,
    const LmaSegmentEngineOptions *options, char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options)
        return lma_segment_adapter_error(error, error_size,
                                           "invalid OLMoE Segment open");
    *engine_impl = NULL;
    if (options->backend_mask &&
        (options->backend_mask & ~LMA_SEGMENT_CAP_CPU))
        return lma_segment_adapter_error(error, error_size,
                                           "OLMoE Segment supports CPU only");
    if (options->context_tokens > 4096)
        return lma_segment_adapter_error(error, error_size,
                                           "OLMoE Segment context exceeds 4096");

    Cfg config;
    memset(&config, 0, sizeof(config));
    load_cfg(&config, options->model_dir);
    if (options->layer_end > (uint32_t)config.n_layers)
        return lma_segment_adapter_error(error, error_size,
                                           "OLMoE Segment range exceeds model");
    int range_layers = (int)(options->layer_end - options->layer_begin);
    int cap = 16;
    if (options->memory_limit_bytes) {
        uint64_t weights = (uint64_t)config.hidden * config.inter * 3u;
        uint64_t scales = (uint64_t)(config.inter * 2 + config.hidden) *
                          sizeof(float);
        uint64_t per_slot = weights + scales;
        uint64_t slots = per_slot && range_layers > 0
            ? options->memory_limit_bytes / per_slot / (uint64_t)range_layers
            : 0;
        cap = slots > (uint64_t)config.n_experts ? config.n_experts : (int)slots;
        if (cap < 1) cap = 1;
    }

    OlmoeSegmentEngine *engine = calloc(1, sizeof(*engine));
    if (!engine)
        return lma_segment_adapter_error(error, error_size,
                                           "out of memory opening OLMoE Segment");
    engine->layer_begin = options->layer_begin;
    engine->layer_end = options->layer_end;
    engine->context_tokens = options->context_tokens;
    if (pthread_mutex_init(&engine->run_lock, NULL)) {
        free(engine);
        return lma_segment_adapter_error(error, error_size,
                                           "cannot initialize OLMoE Segment lock");
    }
    model_init_range(&engine->model, options->model_dir, cap, 8,
                     (int)options->layer_begin, (int)options->layer_end, 0, 0);

    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = LMA_SEGMENT_ABI_VERSION;
    capabilities->flags = LMA_SEGMENT_CAP_SNAPSHOT |
                          LMA_SEGMENT_CAP_RANGE_NATIVE |
                          LMA_SEGMENT_CAP_MULTI_SESSION |
                          LMA_SEGMENT_CAP_CPU;
    lma_segment_capability_string(capabilities->engine_id,
                                   sizeof(capabilities->engine_id), "olmoe");
    lma_segment_capability_string(capabilities->state_schema,
                                   sizeof(capabilities->state_schema),
                                   "olmoe/kv-f32-v1");
    lma_segment_capability_string(capabilities->numeric_class,
                                   sizeof(capabilities->numeric_class),
                                   "olmoe/f32-int8/cpu-v1");
    capabilities->state_dtype = LMA_SEGMENT_DTYPE_F32;
    capabilities->state_width = (uint32_t)config.hidden;
    capabilities->max_batch_rows = 128;
    capabilities->max_context_tokens = 4096;
    capabilities->num_layers = (uint32_t)config.n_layers;
    *engine_impl = engine;
    return 0;
}

static void olmoe_segment_engine_destroy(void *engine_impl) {
    OlmoeSegmentEngine *engine = (OlmoeSegmentEngine *)engine_impl;
    if (!engine) return;
    olmoe_segment_model_destroy(engine);
    pthread_mutex_destroy(&engine->run_lock);
    free(engine);
}

static int olmoe_segment_session_create(
    void *engine_impl, void **session_impl,
    const LmaSegmentSessionOptions *options, char *error, size_t error_size) {
    OlmoeSegmentEngine *engine = (OlmoeSegmentEngine *)engine_impl;
    if (!engine || !session_impl || !options)
        return lma_segment_adapter_error(error, error_size,
                                           "invalid OLMoE Segment session");
    *session_impl = NULL;
    OlmoeSegmentSession *session = calloc(1, sizeof(*session));
    if (!session)
        return lma_segment_adapter_error(error, error_size,
                                           "out of memory creating OLMoE session");
    session->engine = engine;
    session->context_tokens = options->context_tokens;
    int layers = engine->model.c.n_layers;
    session->K = calloc((size_t)layers, sizeof(*session->K));
    session->V = calloc((size_t)layers, sizeof(*session->V));
    if (!session->K || !session->V) goto oom;
    size_t cells;
    if (lma_segment_size_mul((size_t)engine->model.c.n_heads,
                              options->context_tokens, &cells) ||
        lma_segment_size_mul(cells, (size_t)engine->model.c.head_dim,
                              &cells)) goto oom;
    for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
         layer++) {
        session->K[layer] = calloc(cells, sizeof(float));
        session->V[layer] = calloc(cells, sizeof(float));
        if (!session->K[layer] || !session->V[layer]) goto oom;
    }
    *session_impl = session;
    return 0;

oom:
    if (session->K && session->V)
        for (uint32_t layer = engine->layer_begin; layer < engine->layer_end;
             layer++) {
            free(session->K[layer]); free(session->V[layer]);
        }
    free(session->K); free(session->V); free(session);
    return lma_segment_adapter_error(error, error_size,
                                       "out of memory allocating OLMoE KV");
}

static void olmoe_segment_session_destroy(void *session_impl) {
    OlmoeSegmentSession *session = (OlmoeSegmentSession *)session_impl;
    if (!session) return;
    for (uint32_t layer = session->engine->layer_begin;
         layer < session->engine->layer_end; layer++) {
        free(session->K[layer]); free(session->V[layer]);
    }
    free(session->K); free(session->V); free(session);
}

static int olmoe_segment_session_run(void *session_impl,
                                     const LmaSegmentRunRequest *request,
                                     char *error, size_t error_size) {
    OlmoeSegmentSession *session = (OlmoeSegmentSession *)session_impl;
    if (!session || !request || request->position != session->position)
        return lma_segment_adapter_error(
            error, error_size, "OLMoE Segment requires contiguous positions");
    if (request->should_cancel &&
        request->should_cancel(request->cancel_user_data))
        return lma_segment_adapter_error(error, error_size,
                                           "OLMoE Segment run cancelled");
    OlmoeSegmentEngine *engine = session->engine;
    if (request->output != request->input)
        memcpy(request->output, request->input, request->input_bytes);
    pthread_mutex_lock(&engine->run_lock);
    Model *model = &engine->model;
    model->K = session->K; model->V = session->V;
    model->max_t = (int)session->context_tokens;
    model->kv_len = (int)session->position;
    layers_forward_range(model, (float *)request->output, (int)request->rows,
                         (int)request->position, (int)engine->layer_begin,
                         (int)engine->layer_end, 0);
    model->K = NULL; model->V = NULL; model->max_t = 0; model->kv_len = 0;
    pthread_mutex_unlock(&engine->run_lock);
    session->position += request->rows;
    return 0;
}

static int olmoe_segment_payload_size(const OlmoeSegmentSession *session,
                                      uint32_t position, size_t *bytes) {
    size_t cells = (size_t)(session->engine->layer_end -
                            session->engine->layer_begin);
    if (lma_segment_size_mul(cells, 2, &cells) ||
        lma_segment_size_mul(cells,
                              (size_t)session->engine->model.c.n_heads,
                              &cells) ||
        lma_segment_size_mul(cells, position, &cells) ||
        lma_segment_size_mul(cells,
                              (size_t)session->engine->model.c.head_dim,
                              &cells) ||
        lma_segment_size_mul(cells, sizeof(float), bytes)) return -1;
    return 0;
}

static uint64_t olmoe_segment_state_hash(const OlmoeSegmentSession *session) {
    uint64_t hash = LMA_SEGMENT_HASH_INIT;
    size_t row_bytes = (size_t)session->position *
                       session->engine->model.c.head_dim * sizeof(float);
    int heads = session->engine->model.c.n_heads;
    size_t stride = (size_t)session->context_tokens *
                    session->engine->model.c.head_dim;
    for (uint32_t layer = session->engine->layer_begin;
         layer < session->engine->layer_end; layer++)
        for (int kv = 0; kv < 2; kv++) {
            float *state = kv ? session->V[layer] : session->K[layer];
            for (int head = 0; head < heads; head++)
                hash = lma_segment_hash_update(hash, state + head * stride,
                                                row_bytes);
        }
    return hash;
}

static int olmoe_segment_session_snapshot(
    void *session_impl, LmaSegmentWriteFn write_fn, void *write_user_data,
    char *error, size_t error_size) {
    OlmoeSegmentSession *session = (OlmoeSegmentSession *)session_impl;
    size_t payload_bytes;
    if (!session || olmoe_segment_payload_size(session, session->position,
                                               &payload_bytes))
        return lma_segment_adapter_error(error, error_size,
                                           "OLMoE snapshot size overflow");
    LmaSegmentSnapshotHeader header;
    lma_segment_snapshot_header_init(
        &header, "olmoe", session->engine->layer_begin,
        session->engine->layer_end, session->context_tokens, session->position,
        payload_bytes, olmoe_segment_state_hash(session));
    if (lma_segment_stream_write(write_fn, write_user_data, &header,
                                  sizeof(header), error, error_size)) return -1;
    size_t row_bytes = (size_t)session->position *
                       session->engine->model.c.head_dim * sizeof(float);
    int heads = session->engine->model.c.n_heads;
    size_t stride = (size_t)session->context_tokens *
                    session->engine->model.c.head_dim;
    for (uint32_t layer = session->engine->layer_begin;
         layer < session->engine->layer_end; layer++)
        for (int kv = 0; kv < 2; kv++) {
            float *state = kv ? session->V[layer] : session->K[layer];
            for (int head = 0; head < heads; head++)
                if (lma_segment_stream_write(
                        write_fn, write_user_data, state + head * stride,
                        row_bytes, error, error_size)) return -1;
        }
    return 0;
}

static int olmoe_segment_session_restore(
    void *session_impl, LmaSegmentReadFn read_fn, void *read_user_data,
    char *error, size_t error_size) {
    OlmoeSegmentSession *session = (OlmoeSegmentSession *)session_impl;
    LmaSegmentSnapshotHeader header;
    if (!session || lma_segment_stream_read(read_fn, read_user_data, &header,
                                             sizeof(header), error, error_size))
        return -1;
    size_t payload_bytes;
    if (olmoe_segment_payload_size(session, header.position, &payload_bytes) ||
        lma_segment_snapshot_header_valid(
            &header, "olmoe", session->engine->layer_begin,
            session->engine->layer_end, session->context_tokens, payload_bytes,
            error, error_size)) return -1;
    unsigned char *payload = payload_bytes ? malloc(payload_bytes) : NULL;
    if (payload_bytes && !payload)
        return lma_segment_adapter_error(error, error_size,
                                           "out of memory restoring OLMoE KV");
    if (lma_segment_stream_read(read_fn, read_user_data, payload, payload_bytes,
                                 error, error_size)) {
        free(payload);
        return -1;
    }
    if (lma_segment_hash_update(LMA_SEGMENT_HASH_INIT, payload,
                                 payload_bytes) != header.payload_hash) {
        free(payload);
        return lma_segment_adapter_error(error, error_size,
                                           "OLMoE snapshot checksum mismatch");
    }
    size_t row_bytes = (size_t)header.position *
                       session->engine->model.c.head_dim * sizeof(float);
    int heads = session->engine->model.c.n_heads;
    size_t stride = (size_t)session->context_tokens *
                    session->engine->model.c.head_dim;
    unsigned char *cursor = payload;
    for (uint32_t layer = session->engine->layer_begin;
         layer < session->engine->layer_end; layer++)
        for (int kv = 0; kv < 2; kv++) {
            float *state = kv ? session->V[layer] : session->K[layer];
            for (int head = 0; head < heads; head++) {
                memcpy(state + head * stride, cursor, row_bytes);
                cursor += row_bytes;
            }
        }
    session->position = header.position;
    free(payload);
    return 0;
}

static const LmaSegmentAdapter olmoe_segment_adapter = {
    sizeof(LmaSegmentAdapter), LMA_SEGMENT_ABI_VERSION, "olmoe",
    olmoe_segment_engine_open, olmoe_segment_engine_destroy,
    olmoe_segment_session_create, olmoe_segment_session_destroy,
    olmoe_segment_session_run, olmoe_segment_session_snapshot,
    olmoe_segment_session_restore, {0}
};

int lma_olmoe_segment_adapter_register(void) {
    return lma_segment_adapter_register(&olmoe_segment_adapter);
}
#endif /* LMA_SEGMENT_ADAPTER */

#ifdef LMA_EDGE_ADAPTER
/* ---------- engine-owned model Edge adapter --------------------------- */

typedef struct {
    Model model;
    Tok tokenizer;
} OlmoeEdgeEngine;

static void olmoe_edge_engine_destroy(void *engine_impl) {
    OlmoeEdgeEngine *engine = (OlmoeEdgeEngine *)engine_impl;
    if (!engine) return;
    free(engine->model.embed);
    free(engine->model.lm_head);
    free(engine->model.final_norm);
    st_destroy(&engine->model.S);
    tok_free(&engine->tokenizer);
    free(engine);
}

static int olmoe_edge_engine_open(
    void **engine_impl, LmaEdgeCapabilities *capabilities,
    const LmaEdgeEngineOptions *options, char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options)
        return lma_edge_adapter_error(error, error_size,
                                       "invalid OLMoE Edge open");
    *engine_impl = NULL;
    if (options->backend_mask &&
        (options->backend_mask & ~LMA_EDGE_CAP_CPU))
        return lma_edge_adapter_error(error, error_size,
                                       "OLMoE Edge supports CPU only");
    OlmoeEdgeEngine *engine = calloc(1, sizeof(*engine));
    if (!engine)
        return lma_edge_adapter_error(error, error_size,
                                       "out of memory opening OLMoE Edge");
    load_cfg(&engine->model.c, options->model_dir);
    st_init(&engine->model.S, options->model_dir);
    engine->model.embed = load_t(&engine->model, "model.embed_tokens.weight");
    engine->model.lm_head = load_t(&engine->model, "lm_head.weight");
    engine->model.final_norm = load_t(&engine->model, "model.norm.weight");
    char tokenizer_path[4096];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json",
             options->model_dir);
    tok_load(&engine->tokenizer, tokenizer_path);

    Cfg *config = &engine->model.c;
    uint64_t cells = (uint64_t)config->vocab * config->hidden;
    uint64_t resident = (2u * cells + (uint64_t)config->hidden) * sizeof(float);
    if (options->memory_limit_bytes && resident > options->memory_limit_bytes) {
        olmoe_edge_engine_destroy(engine);
        return lma_edge_adapter_error(error, error_size,
                                       "OLMoE Edge exceeds memory limit");
    }
    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = LMA_EDGE_ABI_VERSION;
    capabilities->flags = LMA_EDGE_CAP_TOKENIZE |
                          LMA_EDGE_CAP_DETOKENIZE |
                          LMA_EDGE_CAP_GREEDY | LMA_EDGE_CAP_LOGITS |
                          LMA_EDGE_CAP_CPU;
    lma_edge_capability_string(capabilities->engine_id,
                                sizeof(capabilities->engine_id), "olmoe");
    lma_edge_capability_string(capabilities->state_schema,
                                sizeof(capabilities->state_schema),
                                "olmoe/kv-f32-v1");
    lma_edge_capability_string(capabilities->numeric_class,
                                sizeof(capabilities->numeric_class),
                                "olmoe/f32-int8/cpu-v1");
    lma_edge_capability_string(capabilities->tokenizer_class,
                                sizeof(capabilities->tokenizer_class),
                                "olmoe/byte-bpe-v1");
    capabilities->state_dtype = LMA_EDGE_DTYPE_F32;
    capabilities->state_width = (uint32_t)config->hidden;
    capabilities->vocab_size = (uint32_t)config->vocab;
    capabilities->max_batch_rows = 128;
    capabilities->max_context_tokens = 4096;
    capabilities->num_layers = (uint32_t)config->n_layers;
    capabilities->bos_token_id = -1;
    capabilities->eos_token_id = -1;
    capabilities->resident_bytes = resident;
    *engine_impl = engine;
    return 0;
}

static int olmoe_edge_tokenize(
    void *engine_impl, const char *text, size_t text_bytes,
    int32_t *token_ids, size_t token_capacity, size_t *token_count,
    char *error, size_t error_size) {
    OlmoeEdgeEngine *engine = (OlmoeEdgeEngine *)engine_impl;
    return lma_edge_tok_tokenize(&engine->tokenizer, text, text_bytes,
                                  token_ids, token_capacity, token_count,
                                  error, error_size);
}

static int olmoe_edge_detokenize(
    void *engine_impl, const int32_t *token_ids, size_t token_count,
    char *text, size_t text_capacity, size_t *text_bytes,
    char *error, size_t error_size) {
    OlmoeEdgeEngine *engine = (OlmoeEdgeEngine *)engine_impl;
    return lma_edge_tok_detokenize(&engine->tokenizer, token_ids, token_count,
                                    text, text_capacity, text_bytes,
                                    error, error_size);
}

static int olmoe_edge_embed(void *engine_impl,
                            const LmaEdgeEmbedRequest *request,
                            char *error, size_t error_size) {
    OlmoeEdgeEngine *engine = (OlmoeEdgeEngine *)engine_impl;
    Cfg *config = &engine->model.c;
    float *output = (float *)request->output;
    for (uint32_t row = 0; row < request->rows; row++) {
        int token = request->token_ids[row];
        if (token < 0 || token >= config->vocab)
            return lma_edge_adapter_error(error, error_size,
                                           "OLMoE token ID is out of range");
        memcpy(output + (size_t)row * config->hidden,
               engine->model.embed + (size_t)token * config->hidden,
               (size_t)config->hidden * sizeof(float));
    }
    return 0;
}

static int olmoe_edge_select(void *engine_impl,
                             const LmaEdgeSelectRequest *request,
                             char *error, size_t error_size) {
    OlmoeEdgeEngine *engine = (OlmoeEdgeEngine *)engine_impl;
    Cfg *config = &engine->model.c;
    float *normalized = falloc(config->hidden);
    float *logits = falloc(config->vocab);
    const float *input = (const float *)request->input;
    for (uint32_t row = 0; row < request->rows; row++) {
        if (request->should_cancel &&
            request->should_cancel(request->cancel_user_data)) {
            free(logits); free(normalized);
            return lma_edge_adapter_error(error, error_size,
                                           "OLMoE Edge selection cancelled");
        }
        rmsnorm_row(normalized, input + (size_t)row * config->hidden,
                    engine->model.final_norm, config->hidden, config->eps);
        matmul(logits, normalized, engine->model.lm_head,
               1, config->hidden, config->vocab);
        if (lma_edge_argmax(logits, (uint32_t)config->vocab,
                            &request->token_ids[row],
                            request->scores ? &request->scores[row] : NULL)) {
            free(logits); free(normalized);
            return lma_edge_adapter_error(error, error_size,
                                           "OLMoE Edge head failed");
        }
    }
    free(logits); free(normalized);
    return 0;
}

static int olmoe_edge_logits(void *engine_impl,
                             const LmaEdgeLogitsRequest *request,
                             char *error, size_t error_size) {
    OlmoeEdgeEngine *engine = (OlmoeEdgeEngine *)engine_impl;
    Cfg *config = &engine->model.c;
    float *normalized = falloc(config->hidden);
    const float *input = (const float *)request->input;
    for (uint32_t row = 0; row < request->rows; row++) {
        if (request->should_cancel &&
            request->should_cancel(request->cancel_user_data)) {
            free(normalized);
            return lma_edge_adapter_error(error, error_size,
                                           "OLMoE Edge logits cancelled");
        }
        rmsnorm_row(normalized, input + (size_t)row * config->hidden,
                    engine->model.final_norm, config->hidden, config->eps);
        matmul(request->logits + (size_t)row * config->vocab,
               normalized, engine->model.lm_head,
               1, config->hidden, config->vocab);
    }
    free(normalized);
    return 0;
}

static const LmaEdgeAdapter olmoe_edge_adapter = {
    sizeof(LmaEdgeAdapter), LMA_EDGE_ABI_VERSION, "olmoe",
    olmoe_edge_engine_open, olmoe_edge_engine_destroy,
    olmoe_edge_tokenize, olmoe_edge_detokenize,
    olmoe_edge_embed, olmoe_edge_select, olmoe_edge_logits, {0}
};

int lma_olmoe_edge_adapter_register(void) {
    return lma_edge_adapter_register(&olmoe_edge_adapter);
}
#endif /* LMA_EDGE_ADAPTER */
