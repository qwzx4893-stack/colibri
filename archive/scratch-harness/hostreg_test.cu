// Can a file-backed mapped view be registered with CUDA and DMA'd directly (zero-copy from the page cache)?
#include <cuda_runtime.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
int main(int argc, char **argv) {
    const char *path = argv[1];
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) { printf("open fail\n"); return 1; }
    LARGE_INTEGER sz; GetFileSizeEx(f, &sz);
    HANDLE m = CreateFileMappingA(f, NULL, PAGE_READONLY, 0, 0, NULL);
    unsigned char *p = (unsigned char *)MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!p) { printf("map fail %lu\n", GetLastError()); return 1; }
    size_t n = (size_t)sz.QuadPart;
    // touch everything once so it is in the page cache
    volatile unsigned long long acc = 0; for (size_t i = 0; i < n; i += 4096) acc += p[i];
    void *d0; cudaMalloc(&d0, 64 << 20);
    { size_t blk = 2626560; int reps = 200;
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < reps; i++) { size_t off = ((size_t)(i * 7919) % ((n - blk) / 4096)) * 4096; cudaMemcpy(d0, p + off, blk, cudaMemcpyHostToDevice); }
      double dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
      printf("H2D from UNREGISTERED mapped view: %.2f GB/s\n", reps * (double)blk / dt / 1e9);
      t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < reps; i++) { size_t off = ((size_t)(i * 7919) % ((n - blk) / 4096)) * 4096; unsigned char *q = p + off;
          cudaHostRegister(q, blk, cudaHostRegisterReadOnly); cudaMemcpy(d0, q, blk, cudaMemcpyHostToDevice); cudaHostUnregister(q); }
      dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
      printf("register+copy+unregister per 2.6MB chunk: %.2f GB/s (%.2f ms each)\n", reps * (double)blk / dt / 1e9, dt / reps * 1e3); }
    cudaError_t e = cudaHostRegister(p, n, cudaHostRegisterReadOnly);
    printf("cudaHostRegister(ReadOnly) on mapped file view: %s\n", cudaGetErrorString(e));
    if (e != cudaSuccess) { cudaGetLastError(); e = cudaHostRegister(p, n, 0); printf("cudaHostRegister(0): %s\n", cudaGetErrorString(e)); }
    void *d; cudaMalloc(&d, 64 << 20);
    size_t blk = 2626560; int reps = 200;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; i++) { size_t off = ((size_t)(i * 7919) % ((n - blk) / 4096)) * 4096; cudaMemcpy(d, p + off, blk, cudaMemcpyHostToDevice); }
    double dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    printf("H2D from mapped view: %.2f GB/s (%s)\n", reps * (double)blk / dt / 1e9, cudaGetErrorString(cudaGetLastError()));
    return 0;
}
