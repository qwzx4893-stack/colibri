// lma_hwprof_cuda: measured GPU-side numbers for the LM Atomic cost model.
// Prints one JSON object. All values are MEASURED here, none are datasheet numbers:
//   h2d/d2h GB/s at several sizes, pinned vs pageable
//   vram_read_gbs / vram_copy_gbs
//   gpu_expert_ms      one expert (gate, up, down int8 GEMV over `tokens` rows), warm VRAM
//   overlap            does an async H2D copy run concurrently with a kernel (copy engine)?
//   pinned_alloc_gb    how much page-locked host memory the OS lets us take
// usage: lma_hwprof_cuda [--hidden D] [--inter I]
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "lma_hwprof_cuda: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static double now_s() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// int8 weights [O][I], float activations [T][I] -> [T][O]. One block per output row.
__global__ void gemv_i8(float *y, const float *x, const signed char *w, const float *scale, int I, int O, int T) {
    int o = blockIdx.x;
    if (o >= O) return;
    const signed char *row = w + (size_t)o * I;
    for (int t = 0; t < T; t++) {
        float acc = 0.f;
        for (int i = threadIdx.x; i < I; i += blockDim.x) acc += x[(size_t)t * I + i] * (float)row[i];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        __shared__ float part[32];
        int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        if (lane == 0) part[wid] = acc;
        __syncthreads();
        if (threadIdx.x == 0) {
            float s = 0.f;
            for (int k = 0; k < (blockDim.x + 31) / 32; k++) s += part[k];
            y[(size_t)t * O + o] = s * scale[o];
        }
        __syncthreads();
    }
}

__global__ void read_kernel(const float4 *p, size_t n, float *sink) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n; i += stride) { float4 v = p[i]; acc += v.x + v.y + v.z + v.w; }
    if (acc == 12345.678f) *sink = acc;
}

__global__ void spin_kernel(float *p, int iters) {
    float v = p[threadIdx.x];
    for (int i = 0; i < iters; i++) v = v * 1.0000001f + 0.5f;
    p[threadIdx.x] = v;
}

static double copy_gbs(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind, int reps) {
    CK(cudaMemcpy(dst, src, bytes, kind));
    double best = 0;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        CK(cudaMemcpy(dst, src, bytes, kind));
        CK(cudaDeviceSynchronize());
        double gbs = (double)bytes / (now_s() - t0) / 1e9;
        if (gbs > best) best = gbs;
    }
    return best;
}

int main(int argc, char **argv) {
    int D = 2048, I = 1024;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--hidden")) D = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--inter")) I = atoi(argv[i + 1]);
    }
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) { printf("{\"tool\":\"lma_hwprof_cuda\",\"error\":\"no CUDA device\"}\n"); return 0; }
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    size_t free_b = 0, total_b = 0; CK(cudaMemGetInfo(&free_b, &total_b));
    int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);

    printf("{\n  \"tool\": \"lma_hwprof_cuda\", \"version\": 1,\n");
    printf("  \"device\": \"%s\", \"sm\": %d%d, \"sm_count\": %d, \"vram_total_gb\": %.2f, \"vram_free_gb\": %.2f,\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, total_b / 1e9, free_b / 1e9);
    printf("  \"cuda_driver\": %d, \"cuda_runtime\": %d, \"async_engines\": %d, \"concurrent_kernels\": %d,\n",
           drv, rt, prop.asyncEngineCount, prop.concurrentKernels);

    // ---- host <-> device ----
    const size_t sizes[] = {1u << 20, 8u << 20, 32u << 20, 256u << 20};
    printf("  \"transfer\": [\n");
    for (int si = 0; si < 4; si++) {
        size_t n = sizes[si];
        void *d = NULL; CK(cudaMalloc(&d, n));
        void *pin = NULL; CK(cudaMallocHost(&pin, n)); memset(pin, 1, n);
        std::vector<char> pageable(n, 1);
        double h2d_pin = copy_gbs(d, pin, n, cudaMemcpyHostToDevice, 6);
        double d2h_pin = copy_gbs(pin, d, n, cudaMemcpyDeviceToHost, 6);
        double h2d_pg = copy_gbs(d, pageable.data(), n, cudaMemcpyHostToDevice, 4);
        double d2h_pg = copy_gbs(pageable.data(), d, n, cudaMemcpyDeviceToHost, 4);
        printf("    {\"bytes\": %zu, \"h2d_pinned_gbs\": %.2f, \"d2h_pinned_gbs\": %.2f, \"h2d_pageable_gbs\": %.2f, \"d2h_pageable_gbs\": %.2f}%s\n",
               n, h2d_pin, d2h_pin, h2d_pg, d2h_pg, si < 3 ? "," : "");
        cudaFreeHost(pin); cudaFree(d);
    }
    printf("  ],\n");

    // ---- VRAM bandwidth ----
    {
        size_t n = (size_t)512 << 20;
        float *a = NULL, *b = NULL, *sink = NULL;
        CK(cudaMalloc(&a, n)); CK(cudaMalloc(&b, n)); CK(cudaMalloc(&sink, 4));
        CK(cudaMemset(a, 1, n));
        double best_r = 0, best_c = 0;
        for (int r = 0; r < 5; r++) {
            double t0 = now_s();
            read_kernel<<<prop.multiProcessorCount * 8, 256>>>((const float4 *)a, n / 16, sink);
            CK(cudaDeviceSynchronize());
            double g = (double)n / (now_s() - t0) / 1e9; if (g > best_r) best_r = g;
            t0 = now_s();
            CK(cudaMemcpy(b, a, n, cudaMemcpyDeviceToDevice)); CK(cudaDeviceSynchronize());
            g = (double)n * 2 / (now_s() - t0) / 1e9; if (g > best_c) best_c = g;
        }
        printf("  \"vram_read_gbs\": %.1f, \"vram_copy_gbs\": %.1f,\n", best_r, best_c);
        cudaFree(a); cudaFree(b); cudaFree(sink);
    }

    // ---- one expert on the GPU, warm, for T tokens ----
    {
        size_t ng = (size_t)I * D;
        signed char *w; float *sc, *x, *g, *u, *h;
        CK(cudaMalloc(&w, ng * 3)); CK(cudaMemset(w, 3, ng * 3));
        CK(cudaMalloc(&sc, sizeof(float) * (2 * (size_t)I + D)));
        std::vector<float> hs(2 * (size_t)I + D, 0.001f);
        CK(cudaMemcpy(sc, hs.data(), hs.size() * sizeof(float), cudaMemcpyHostToDevice));
        printf("  \"gpu_expert_ms_by_tokens\": {");
        const int toks[] = {1, 16, 64, 256};
        for (int ti = 0; ti < 4; ti++) {
            int T = toks[ti];
            CK(cudaMalloc(&x, sizeof(float) * (size_t)T * D)); CK(cudaMemset(x, 0, sizeof(float) * (size_t)T * D));
            CK(cudaMalloc(&g, sizeof(float) * (size_t)T * I)); CK(cudaMalloc(&u, sizeof(float) * (size_t)T * I));
            CK(cudaMalloc(&h, sizeof(float) * (size_t)T * D));
            for (int warm = 0; warm < 3; warm++) {
                gemv_i8<<<I, 128>>>(g, x, w, sc, D, I, T);
                gemv_i8<<<I, 128>>>(u, x, w + ng, sc + I, D, I, T);
                gemv_i8<<<D, 128>>>(h, g, w + 2 * ng, sc + 2 * I, I, D, T);
            }
            CK(cudaDeviceSynchronize());
            double t0 = now_s(); int reps = 30;
            for (int r = 0; r < reps; r++) {
                gemv_i8<<<I, 128>>>(g, x, w, sc, D, I, T);
                gemv_i8<<<I, 128>>>(u, x, w + ng, sc + I, D, I, T);
                gemv_i8<<<D, 128>>>(h, g, w + 2 * ng, sc + 2 * I, I, D, T);
            }
            CK(cudaDeviceSynchronize());
            printf("%s\"%d\": %.4f", ti ? ", " : "", T, (now_s() - t0) / reps * 1e3);
            cudaFree(x); cudaFree(g); cudaFree(u); cudaFree(h);
        }
        printf("},\n");
        cudaFree(w); cudaFree(sc);
    }

    // ---- copy/compute overlap ----
    {
        size_t n = (size_t)64 << 20;
        void *pin = NULL, *d = NULL; float *sp = NULL;
        CK(cudaMallocHost(&pin, n)); CK(cudaMalloc(&d, n)); CK(cudaMalloc(&sp, 4096));
        cudaStream_t s1, s2; CK(cudaStreamCreate(&s1)); CK(cudaStreamCreate(&s2));
        int iters = 4000000;
        spin_kernel<<<1, 256, 0, s1>>>(sp, iters); CK(cudaStreamSynchronize(s1));
        double t0 = now_s(); spin_kernel<<<1, 256, 0, s1>>>(sp, iters); CK(cudaStreamSynchronize(s1));
        double tk = now_s() - t0;
        t0 = now_s(); CK(cudaMemcpyAsync(d, pin, n, cudaMemcpyHostToDevice, s2)); CK(cudaStreamSynchronize(s2));
        double tc = now_s() - t0;
        t0 = now_s();
        spin_kernel<<<1, 256, 0, s1>>>(sp, iters);
        CK(cudaMemcpyAsync(d, pin, n, cudaMemcpyHostToDevice, s2));
        CK(cudaStreamSynchronize(s1)); CK(cudaStreamSynchronize(s2));
        double tb = now_s() - t0;
        printf("  \"overlap\": {\"kernel_ms\": %.2f, \"copy_ms\": %.2f, \"both_ms\": %.2f, \"overlap_fraction\": %.2f},\n",
               tk * 1e3, tc * 1e3, tb * 1e3, (tk + tc - tb) / (tk < tc ? tk : tc));
        cudaFreeHost(pin); cudaFree(d); cudaFree(sp);
    }

    // ---- how much page-locked memory can we take? ----
    {
        size_t step = (size_t)256 << 20, got = 0; std::vector<void *> keep;
        for (int i = 0; i < 64; i++) {
            void *p = NULL;
            if (cudaMallocHost(&p, step) != cudaSuccess) { cudaGetLastError(); break; }
            keep.push_back(p); got += step;
        }
        for (void *p : keep) cudaFreeHost(p);
        printf("  \"pinned_alloc_gb\": %.2f\n", got / 1e9);
    }
    printf("}\n");
    return 0;
}
