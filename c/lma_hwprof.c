/* lma_hwprof: measured CPU and memory numbers for the LM Atomic cost model.
 *
 * Prints one JSON object on stdout. Everything here is MEASURED on the machine it
 * runs on; nothing is a datasheet constant. The scheduler and the digital twin read
 * this file (through lmatomic/hwprof.py) instead of assuming bandwidths.
 *
 *   ram_read_gbs        streaming read, all threads / one thread (buffer >> LLC)
 *   ram_copy_gbs        memcpy, all threads
 *   cpu_expert_*        the engine's own int8-weight GEMV loop on an expert-sized
 *                       matrix set (gate, up, down), cache-resident and DRAM-bound,
 *                       one thread and all threads: weight GB/s consumed and the
 *                       time to run one expert for one token
 *
 * usage: lma_hwprof [--threads N] [--hidden D] [--inter I] [--ram-mb M]
 */
#define _GNU_SOURCE
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "compat.h"

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void *big_alloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "lma_hwprof: out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}

static void fill_random(uint8_t *p, size_t n) {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < n; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; p[i] = (uint8_t)(s >> 24); }
}

/* Read every cache line of the buffer: a sum keeps the loads alive. */
static double read_sum(const uint64_t *p, size_t words, volatile uint64_t *sink) {
    uint64_t acc = 0;
    #pragma omp simd reduction(+:acc)
    for (size_t i = 0; i < words; i++) acc += p[i];
    *sink = acc;
    return (double)acc;
}

static double ram_read_gbs(size_t bytes, int threads) {
    size_t words = bytes / 8;
    uint64_t *buf = big_alloc(bytes);
    fill_random((uint8_t *)buf, bytes);
    volatile uint64_t sink = 0;
    double best = 0;
    for (int rep = 0; rep < 4; rep++) {
        double t0 = now_s();
        #pragma omp parallel num_threads(threads)
        {
            int tid = omp_get_thread_num(), nt = omp_get_num_threads();
            size_t per = words / nt;
            volatile uint64_t s2 = 0;
            read_sum(buf + (size_t)tid * per, per, &s2);
        }
        double dt = now_s() - t0;
        double gbs = (double)bytes / dt / 1e9;
        if (gbs > best) best = gbs;
    }
    (void)sink; free(buf);
    return best;
}

static double ram_copy_gbs(size_t bytes, int threads) {
    uint8_t *a = big_alloc(bytes), *b = big_alloc(bytes);
    fill_random(a, bytes); memset(b, 0, bytes);
    double best = 0;
    for (int rep = 0; rep < 4; rep++) {
        double t0 = now_s();
        #pragma omp parallel num_threads(threads)
        {
            int tid = omp_get_thread_num(), nt = omp_get_num_threads();
            size_t per = bytes / nt;
            memcpy(b + (size_t)tid * per, a + (size_t)tid * per, per);
        }
        double dt = now_s() - t0;
        double gbs = (double)bytes * 2 / dt / 1e9;      /* read + write traffic */
        if (gbs > best) best = gbs;
    }
    free(a); free(b);
    return best;
}

/* The engine's default expert kernel (olmoe.c matmul_q, IDOT off): float accumulate
 * over int8 weights, per output row. Serial per call so the timing is comparable to
 * one item of expert-major prefill. */
static void gemv_i8(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        #pragma omp simd reduction(+:acc)
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}

typedef struct { int8_t *g, *u, *d; float *gs, *us, *ds; } Expert;

static Expert expert_new(int D, int I) {
    Expert e;
    size_t ng = (size_t)I * D;
    e.g = big_alloc(ng * 3); fill_random((uint8_t *)e.g, ng * 3);
    e.u = e.g + ng; e.d = e.g + 2 * ng;
    e.gs = big_alloc(sizeof(float) * (2 * (size_t)I + D));
    for (int i = 0; i < 2 * I + D; i++) e.gs[i] = 0.001f + 1e-6f * (float)(i % 97);
    e.us = e.gs + I; e.ds = e.gs + 2 * I;
    return e;
}

static void expert_run(const Expert *e, const float *x, float *g, float *u, float *hh, int D, int I) {
    gemv_i8(g, x, e->g, e->gs, D, I);
    gemv_i8(u, x, e->u, e->us, D, I);
    for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
    gemv_i8(hh, g, e->d, e->ds, I, D);
}

/* seconds per expert-for-one-token, one thread, weights resident in cache (same
 * expert repeated) or streamed from DRAM (rotating through `nexp` distinct experts). */
static double expert_seconds(int D, int I, int nexp, int reps) {
    Expert *ex = big_alloc(sizeof(Expert) * (size_t)nexp);
    for (int i = 0; i < nexp; i++) ex[i] = expert_new(D, I);
    float *x = big_alloc(sizeof(float) * (size_t)D), *g = big_alloc(sizeof(float) * (size_t)I),
          *u = big_alloc(sizeof(float) * (size_t)I), *hh = big_alloc(sizeof(float) * (size_t)D);
    for (int i = 0; i < D; i++) x[i] = 0.01f * (float)((i % 13) - 6);
    for (int i = 0; i < nexp; i++) expert_run(&ex[i], x, g, u, hh, D, I);      /* warm */
    double t0 = now_s();
    for (int r = 0; r < reps; r++) expert_run(&ex[r % nexp], x, g, u, hh, D, I);
    double dt = (now_s() - t0) / reps;
    for (int i = 0; i < nexp; i++) { free(ex[i].g); free(ex[i].gs); }
    free(ex); free(x); free(g); free(u); free(hh);
    return dt;
}

/* experts per second with `threads` threads each running its own experts (as in
 * expert-major prefill: independent items, serial kernel per item). */
static double expert_parallel_rate(int D, int I, int threads, int nexp, int reps) {
    Expert *ex = big_alloc(sizeof(Expert) * (size_t)nexp);
    for (int i = 0; i < nexp; i++) ex[i] = expert_new(D, I);
    double t0 = now_s();
    #pragma omp parallel num_threads(threads)
    {
        float *x = big_alloc(sizeof(float) * (size_t)D), *g = big_alloc(sizeof(float) * (size_t)I),
              *u = big_alloc(sizeof(float) * (size_t)I), *hh = big_alloc(sizeof(float) * (size_t)D);
        for (int i = 0; i < D; i++) x[i] = 0.01f * (float)((i % 13) - 6);
        int tid = omp_get_thread_num();
        for (int r = 0; r < reps; r++) expert_run(&ex[(tid + r) % nexp], x, g, u, hh, D, I);
        free(x); free(g); free(u); free(hh);
    }
    double dt = now_s() - t0;
    for (int i = 0; i < nexp; i++) { free(ex[i].g); free(ex[i].gs); }
    free(ex);
    return (double)threads * reps / dt;
}

int main(int argc, char **argv) {
    int threads = omp_get_max_threads(), D = 2048, I = 1024, ram_mb = 1024;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--threads")) threads = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--hidden")) D = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--inter")) I = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--ram-mb")) ram_mb = atoi(argv[i + 1]);
    }
    if (threads < 1) threads = 1;
    size_t ram_bytes = (size_t)ram_mb << 20;
    size_t expert_bytes = (size_t)3 * D * I;
    /* enough distinct experts that the rotating set exceeds a 30 MB LLC by a wide margin */
    int dram_n = (int)((256u << 20) / expert_bytes) + 2;

    double r1 = ram_read_gbs(ram_bytes, 1), rn = ram_read_gbs(ram_bytes, threads);
    double cn = ram_copy_gbs(ram_bytes / 2, threads);
    double t_hot = expert_seconds(D, I, 1, 200);
    double t_dram = expert_seconds(D, I, dram_n, 100);
    double rate_n = expert_parallel_rate(D, I, threads, dram_n, 40);

    printf("{\n");
    printf("  \"tool\": \"lma_hwprof\", \"version\": 1,\n");
    printf("  \"threads\": %d, \"hidden\": %d, \"inter\": %d, \"expert_int8_bytes\": %zu,\n", threads, D, I, expert_bytes);
    printf("  \"ram_read_gbs_1thread\": %.2f, \"ram_read_gbs_all\": %.2f, \"ram_copy_gbs_all\": %.2f,\n", r1, rn, cn);
    printf("  \"cpu_expert_ms_cache_resident_1thread\": %.4f,\n", t_hot * 1e3);
    printf("  \"cpu_expert_ms_dram_streamed_1thread\": %.4f,\n", t_dram * 1e3);
    printf("  \"cpu_expert_weight_gbs_1thread_resident\": %.2f,\n", (double)expert_bytes / t_hot / 1e9);
    printf("  \"cpu_expert_weight_gbs_1thread_dram\": %.2f,\n", (double)expert_bytes / t_dram / 1e9);
    printf("  \"cpu_experts_per_s_all_threads\": %.1f,\n", rate_n);
    printf("  \"cpu_expert_weight_gbs_all_threads\": %.2f\n", rate_n * (double)expert_bytes / 1e9);
    printf("}\n");
    return 0;
}
