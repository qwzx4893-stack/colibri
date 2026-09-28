/* moe_router.h: the shared top-K expert selection.
 *
 * Two things are pinned here.
 *
 *  1. Finite inputs rank EXACTLY as the per-engine loops this header replaced
 *     did (same choice float, strict '>', lowest id wins ties). The reference
 *     below is that loop, verbatim in shape, and is compared on random inputs,
 *     with and without bias, with and without a keep mask, across sizes that
 *     take the bitmap path and the prefix-scan path.
 *
 *  2. Non-finite scores never index out of range or duplicate a slot. This is
 *     the class of defect the header exists for: all-NaN scores leave the naive
 *     loop's best at -1, which is then used as an index (score[-1], usage[-1]++,
 *     a negative pread offset).
 *
 * No model file, no engine: the header is self-contained. */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../moe_router.h"

/* The loop every engine used to carry, in its most common shape. */
static void ref_topk(const float *score, const float *bias, const unsigned char *keep,
                     int E, int K, int *idx, int *n_real) {
    *n_real = 0;
    for (int kk = 0; kk < K; kk++) {
        int best = -1; float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            if (keep && !keep[e]) continue;
            int taken = 0; for (int j = 0; j < kk; j++) if (idx[j] == e) { taken = 1; break; }
            float ch = bias ? score[e] + bias[e] : score[e];
            if (!taken && ch > bv) { bv = ch; best = e; }
        }
        idx[kk] = best;
        if (best >= 0) (*n_real)++;
    }
}

static uint32_t rs = 12345u;
static float rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return (float)(rs >> 8) * (1.f / 16777216.f); }

static void check_matches_reference(int E, int K, int use_bias, int use_keep) {
    float *score = malloc((size_t)E * sizeof(float));
    float *bias  = malloc((size_t)E * sizeof(float));
    unsigned char *keep = malloc((size_t)E);
    int *a = malloc((size_t)K * sizeof(int)), *b = malloc((size_t)K * sizeof(int));
    float *v = malloc((size_t)K * sizeof(float));
    for (int rep = 0; rep < 60; rep++) {
        for (int e = 0; e < E; e++) {
            score[e] = rnd();
            bias[e] = (rnd() - 0.5f) * 0.4f;
            keep[e] = rep % 3 == 0 ? (rnd() > 0.3f) : 1;
        }
        if (rep % 7 == 0) for (int e = 0; e + 1 < E; e += 2) score[e + 1] = score[e];   /* ties */
        int nref;
        ref_topk(score, use_bias ? bias : NULL, use_keep ? keep : NULL, E, K, a, &nref);
        int got = coli_router_topk(score, use_bias ? bias : NULL, use_keep ? keep : NULL,
                                   E, K, b, v, 0, 0);
        assert(got == nref);
        for (int k = 0; k < K; k++) {
            assert(a[k] == b[k] && "ranking must equal the per-engine loop it replaced");
            if (b[k] >= 0) {
                float want = use_bias ? score[b[k]] + bias[b[k]] : score[b[k]];
                assert(v[k] == want && "returned value is the ranking float");
            }
        }
    }
    free(score); free(bias); free(keep); free(a); free(b); free(v);
}

static void check_in_range_distinct(const int *idx, int K, int E) {
    for (int k = 0; k < K; k++) {
        assert(idx[k] >= 0 && idx[k] < E && "degraded slot must be a valid expert");
        for (int j = 0; j < k; j++) assert(idx[j] != idx[k] && "no slot may repeat");
    }
}

int main(void) {
    /* --- 1. identical to the loops it replaced (finite inputs) --- */
    check_matches_reference(8, 4, 0, 0);
    check_matches_reference(64, 8, 1, 0);
    check_matches_reference(256, 8, 1, 1);
    check_matches_reference(896, 16, 1, 0);           /* Kimi K3 shape */
    check_matches_reference(5000, 6, 1, 1);           /* above the bitmap limit */

    /* ties go to the lowest expert id */
    { float s[6] = {1.f, 3.f, 3.f, 3.f, 2.f, 3.f}; int idx[3];
      coli_router_topk(s, NULL, NULL, 6, 3, idx, NULL, 0, 0);
      assert(idx[0] == 1 && idx[1] == 2 && idx[2] == 3); }

    /* bias changes the ranking but the caller keeps the raw score */
    { float s[4] = {0.9f, 0.5f, 0.4f, 0.1f}, bz[4] = {0.f, 0.f, 0.6f, 0.f}; int idx[2];
      coli_router_topk(s, bz, NULL, 4, 2, idx, NULL, 0, 0);
      assert(idx[0] == 2 && idx[1] == 0); }

    /* -inf masks a candidate out */
    { float s[4] = {-INFINITY, 0.2f, -INFINITY, 0.1f}; int idx[4];
      int n = coli_router_topk(s, NULL, NULL, 4, 4, idx, NULL, 0, 0);
      assert(n == 2 && idx[0] == 1 && idx[1] == 3 && idx[2] == -1 && idx[3] == -1); }

    /* --- 2. non-finite scores --- */
    { float s[8]; for (int e = 0; e < 8; e++) s[e] = NAN; int idx[4];
      /* the trigger is real: the naive scan selects nothing */
      int nref; ref_topk(s, NULL, NULL, 8, 4, idx, &nref);
      assert(nref == 0 && idx[0] == -1 && "all-NaN leaves the naive scan at -1");
      /* degrade = 0: honest "none" */
      int n = coli_router_topk(s, NULL, NULL, 8, 4, idx, NULL, 0, 0);
      assert(n == 0 && idx[0] == -1 && idx[3] == -1);
      /* degrade = 1: in range, distinct, deterministic (lowest ids) */
      n = coli_router_topk(s, NULL, NULL, 8, 4, idx, NULL, 0, 1);
      assert(n == 0);
      check_in_range_distinct(idx, 4, 8);
      for (int k = 0; k < 4; k++) assert(idx[k] == k); }

    /* one finite expert among NaNs: it is picked first and never duplicated by the fallback.
     * The old fallback (slot index kk) picked expert 3 twice here. */
    { float s[8]; for (int e = 0; e < 8; e++) s[e] = NAN; s[3] = 2.f; int idx[5];
      int n = coli_router_topk(s, NULL, NULL, 8, 5, idx, NULL, 0, 1);
      assert(n == 1 && idx[0] == 3);
      check_in_range_distinct(idx, 5, 8); }

    /* NaN entries are skipped, not chosen (a NaN-first scan that latches onto it would pick 0) */
    { float s[5] = {NAN, 0.1f, 0.7f, NAN, 0.4f}; int idx[3];
      int n = coli_router_topk(s, NULL, NULL, 5, 3, idx, NULL, 0, 0);
      assert(n == 3 && idx[0] == 2 && idx[1] == 4 && idx[2] == 1); }

    /* NaN bias poisons the choice, not the process */
    { float s[4] = {1.f, 2.f, 3.f, 4.f}, bz[4] = {0.f, NAN, 0.f, 0.f}; int idx[4];
      coli_router_topk(s, bz, NULL, 4, 4, idx, NULL, 0, 1);
      check_in_range_distinct(idx, 4, 4);
      assert(idx[0] == 3 && idx[1] == 2 && idx[2] == 0 && "finite candidates first"); }

    /* K larger than E (malformed config): still in range */
    { float s[3] = {0.3f, 0.2f, 0.1f}; int idx[7];
      coli_router_topk(s, NULL, NULL, 3, 7, idx, NULL, 0, 1);
      for (int k = 0; k < 7; k++) assert(idx[k] >= 0 && idx[k] < 3); }

    /* prefix-scan path (E above the bitmap limit) degrades the same way */
    { int E = 5000; float *s = malloc((size_t)E * sizeof(float)); int idx[6];
      for (int e = 0; e < E; e++) s[e] = NAN;
      coli_router_topk(s, NULL, NULL, E, 6, idx, NULL, 0, 1);
      check_in_range_distinct(idx, 6, E);
      for (int k = 0; k < 6; k++) assert(idx[k] == k);
      free(s); }

    /* keep mask with degrade = 0: nothing eligible is "none", not a masked expert */
    { float s[4] = {1.f, 2.f, 3.f, 4.f}; unsigned char keep[4] = {0, 0, 0, 0}; int idx[2];
      int n = coli_router_topk(s, NULL, keep, 4, 2, idx, NULL, 0, 0);
      assert(n == 0 && idx[0] == -1 && idx[1] == -1); }

    /* coli_router_best: incremental picking reproduces coli_router_topk */
    { float s[9] = {0.2f, 0.9f, NAN, 0.9f, 0.1f, 0.5f, 0.8f, NAN, 0.3f}; int a[5], b[5]; float v;
      coli_router_topk(s, NULL, NULL, 9, 5, a, NULL, 0, 0);
      for (int k = 0; k < 5; k++) { b[k] = coli_router_best(s, NULL, NULL, 9, b, k, &v); assert(b[k] == a[k]); }
      float t[3] = {NAN, NAN, NAN};
      assert(coli_router_best(t, NULL, NULL, 3, NULL, 0, &v) == -1 && v == 0.f); }

    /* coli_router_fill: only unfilled slots change, in range, distinct from filled ones */
    { int idx[5] = {4, -1, 0, -1, -1}; float v[5] = {.5f, .1f, .3f, .2f, .9f};
      coli_router_fill(idx, v, 5, 6, 0);
      assert(idx[0] == 4 && idx[2] == 0 && v[0] == .5f && v[2] == .3f);
      check_in_range_distinct(idx, 5, 6);
      assert(idx[1] == 1 && idx[3] == 2 && idx[4] == 3 && v[1] == 0.f && v[4] == 0.f); }

    printf("OK test_moe_router: shared selection matches the per-engine loops; non-finite scores stay in range\n");
    return 0;
}
