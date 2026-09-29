/* moe_router.h -- the one implementation of "pick the top-K experts".
 *
 * Every MoE engine in this tree used to carry its own copy of this loop:
 *
 *     int best = -1; float bv = -1e30f;
 *     for (e ...) if (!taken && score[e] > bv) { bv = score[e]; best = e; }
 *     idx[kk] = best;
 *
 * and each copy drifted on its own: which start value, how "taken" was tested,
 * and above all what happens when best stays -1. If every candidate score is
 * NaN, no comparison succeeds (NaN > bv is false for any bv), so best is -1 and
 * gets used as an index: score[-1] (heap read), usage[-1]++ (heap write) and,
 * because an expert id also picks a file offset, a pread at a negative offset.
 * A release build does not fault; it returns wrong numbers over corrupted
 * memory. NaN reaches the router from a corrupt or hostile expert tile, or from
 * an fp overflow at an eviction boundary (c/tests/test_logit_nan.c documents
 * both and hardened only the sampling side, downstream of this).
 *
 * The fix landed in glm52.c first, then in four sibling engines, and stayed
 * missing in glm53.c -- the recurring shape of defects in this tree, a fix that
 * reaches one engine and not its siblings. So it lives here, once.
 *
 * What is shared, and what is not:
 *   shared    candidate ranking (score + optional bias, optional keep mask),
 *             tie-breaking (lowest expert id wins), the NaN policy, distinctness.
 *   per engine  how the score is produced (sigmoid, softmax, sqrt-softplus),
 *             and what the weights are once experts are chosen (normalisation,
 *             route scale, top-p tail cut). Those are the architecture.
 *
 * Numerics: the ranking is bit-identical to the loops it replaced for finite
 * inputs -- same start value, same strict '>' so the first (lowest-id) maximum
 * wins, and the choice is the same float sum score[e]+bias[e] each engine
 * already formed. Scores at or below -1e30 are never picked, which is how a
 * caller masks a candidate out (a keep mask does the same without a sentinel).
 *
 * Two policies over one core:
 *   degrade = 1  every slot is a valid expert id even when the scores are not
 *                finite: the slot degrades to the lowest expert not yet taken,
 *                and one warning is printed. For engines whose downstream code
 *                indexes tables with the result and cannot represent "none".
 *   degrade = 0  a slot with no finite candidate is -1 and selection stops.
 *                For callers that already handle "none" (prefetch predictions,
 *                keep-masked windows) or that treat it as an error.
 */
#ifndef LMA_MOE_ROUTER_H
#define LMA_MOE_ROUTER_H

#include <stdint.h>
#include <stdio.h>

#define LMA_ROUTER_BITMAP_EXPERTS 4096   /* above this, "taken" is a prefix scan */

/* One pick: the best eligible candidate that is not in taken[0..ntaken), or -1
 * when none is finite. For incremental callers (prefetch predictions that stop
 * once enough probability mass is covered) that would waste work selecting all
 * K up front. Same ranking and tie-break as lma_router_topk. *val gets the
 * ranking value picked (may be NULL). */
static inline int lma_router_best(const float *score, const float *bias,
                                   const unsigned char *keep, int E,
                                   const int *taken, int ntaken, float *val)
{
    int best = -1; float bv = -1e30f;
    for (int e = 0; e < E; e++) {
        if (keep && !keep[e]) continue;
        int t = 0; for (int j = 0; j < ntaken; j++) if (taken[j] == e) { t = 1; break; }
        if (t) continue;
        float ch = bias ? score[e] + bias[e] : score[e];
        if (ch > bv) { bv = ch; best = e; }
    }
    if (val) *val = best >= 0 ? bv : 0.f;
    return best;
}

/* Replace every unfilled slot (idx[k] < 0) with the lowest expert id that is not
 * already present, and zero its val. Slots that are already filled are never
 * touched. This is the "degrade" step on its own, for callers that ran a
 * selection with "none" semantics (lma_router_topk(..., degrade = 0), or a
 * keep-masked window) and then hand idx[] to code that indexes tables with it.
 * One warning is printed the first time it fires. */
static inline void lma_router_fill(int *idx, float *val, int K, int E, int layer)
{
    for (int kk = 0; kk < K; kk++) {
        if (idx[kk] >= 0) continue;
        static int warned;
        if (!warned) {
            warned = 1;
            fprintf(stderr, "[router] non-finite logits at layer %d: selection degraded "
                            "(corrupt expert tile, or overflow at an eviction boundary)\n", layer);
        }
        int pick = -1;
        for (int e = 0; e < E && pick < 0; e++) {
            int t = 0;
            for (int j = 0; j < K; j++) if (idx[j] == e) { t = 1; break; }
            if (!t) pick = e;
        }
        if (pick < 0) pick = E > 0 ? kk % E : 0;   /* K > E: malformed config, stay in range */
        idx[kk] = pick;
        if (val) val[kk] = 0.f;
    }
}

/* Select up to K distinct experts, best first.
 *
 *   score  [E]  candidate scores
 *   bias   [E]  added to score for ranking only (may be NULL)
 *   keep   [E]  non-zero = eligible (may be NULL = all eligible)
 *   idx    [K]  out: expert ids; -1 where no candidate was found (degrade = 0)
 *   val    [K]  out: the ranking value picked (score+bias); may be NULL
 *   layer       for the warning text only
 *
 * Returns how many slots were filled by a real (finite, eligible) candidate.
 * With degrade = 1 the remaining slots are still filled, with in-range ids
 * (lma_router_fill); with degrade = 0 they are -1 and val is 0. */
static inline int lma_router_topk(const float *score, const float *bias,
                                   const unsigned char *keep, int E, int K,
                                   int *idx, float *val, int layer, int degrade)
{
    uint64_t bits[LMA_ROUTER_BITMAP_EXPERTS / 64];
    int use_bits = E <= LMA_ROUTER_BITMAP_EXPERTS;
    int picked = 0;
    if (use_bits) for (int w = 0; w < (E + 63) / 64; w++) bits[w] = 0;

    for (int kk = 0; kk < K; kk++) {
        int best = -1; float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            if (keep && !keep[e]) continue;
            if (use_bits) { if ((bits[e >> 6] >> (e & 63)) & 1u) continue; }
            else { int t = 0; for (int j = 0; j < kk; j++) if (idx[j] == e) { t = 1; break; } if (t) continue; }
            float ch = bias ? score[e] + bias[e] : score[e];
            if (ch > bv) { bv = ch; best = e; }
        }
        if (best < 0) {
            for (int j = kk; j < K; j++) { idx[j] = -1; if (val) val[j] = 0.f; }
            break;
        }
        picked++;
        idx[kk] = best;
        if (val) val[kk] = bv;
        if (use_bits) bits[best >> 6] |= (uint64_t)1 << (best & 63);
    }
    if (degrade && picked < K) lma_router_fill(idx, val, K, E, layer);
    return picked;
}

#endif /* LMA_MOE_ROUTER_H */
