/**
 * gpu_baseline.cu - DBSCAN GPU baseline
 *
 * GPU-accelerated brute-force DBSCAN:
 *   Phase 1 (GPU): Parallel neighbor counting - O(N) per point, N threads
 *   Phase 2 (CPU): Sequential BFS cluster expansion using precomputed neighbor counts
 *
 * This is a hybrid approach: the O(N^2) neighbor counting bottleneck runs on GPU,
 * while the inherently sequential BFS expansion runs on CPU.
 *
 * The more advanced CUDA-DClust+ algorithm (with spatial indexing and fully GPU-based
 * cluster expansion) is preserved in gpu_reference_cudadclust.cu for reference.
 *
 * Reference: Poudel & Gowanlock, "CUDA-DClust+: Revisiting Early
 *            GPU-Accelerated DBSCAN Clustering Designs", IEEE HiPC 2021.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// Constants from original common.h
#define UNPROCESSED -1
#define NOISE       -2

// Module-level state
static int    g_N = 0;
static float* d_xs = NULL;
static float* d_ys = NULL;
static int*   d_neighbor_counts = NULL;

// ===== GPU Kernel: Parallel neighbor counting =====
// Matches processPointCPU from multicore-cpu/dbscan.cpp
// Each thread counts neighbors for one point (brute-force O(N))
__global__ void countNeighborsKernel(
    int N, const float* __restrict__ xs, const float* __restrict__ ys,
    float eps2, int* __restrict__ neighbor_counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float myX = xs[i];
    float myY = ys[i];
    int count = 0;

    for (int j = 0; j < N; j++) {
        float dx = myX - xs[j];
        float dy = myY - ys[j];
        float dist2 = dx * dx + dy * dy;
        if (dist2 <= eps2) count++;
    }
    // count includes self; subtract 1 for actual neighbor count
    neighbor_counts[i] = count - 1;
}

// ===== Interface =====

extern "C" void solution_free(void)
{
    if (d_xs)              { cudaFree(d_xs);              d_xs = NULL; }
    if (d_ys)              { cudaFree(d_ys);              d_ys = NULL; }
    if (d_neighbor_counts) { cudaFree(d_neighbor_counts); d_neighbor_counts = NULL; }
    g_N = 0;
}

extern "C" void solution_compute(int N, const float* xs, const float* ys,
                                  float eps, int minPts, int* labels)
{
    if (g_N != N) {
        solution_free();
        cudaMalloc(&d_xs, N * sizeof(float));
        cudaMalloc(&d_ys, N * sizeof(float));
        cudaMalloc(&d_neighbor_counts, N * sizeof(int));
        g_N = N;
    }

    cudaMemcpy(d_xs, xs, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ys, ys, N * sizeof(float), cudaMemcpyHostToDevice);

    float eps2 = eps * eps;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Phase 1 (GPU): Parallel neighbor counting
    countNeighborsKernel<<<blocks, threads>>>(N, d_xs, d_ys, eps2, d_neighbor_counts);
    cudaDeviceSynchronize();

    // Phase 2 (CPU): Sequential BFS expansion with precomputed counts
    int* h_counts = (int*)malloc(N * sizeof(int));
    cudaMemcpy(h_counts, d_neighbor_counts, N * sizeof(int), cudaMemcpyDeviceToHost);

    // DBSCAN BFS (matches clusterThread + expandCluster from original)
    for (int i = 0; i < N; i++) labels[i] = UNPROCESSED;

    int nextClusterId = 1;
    int* seeds = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        if (labels[i] != UNPROCESSED) continue;

        if (h_counts[i] < minPts) {
            labels[i] = NOISE;
            continue;
        }

        // Core point: start new cluster
        int clusterId = nextClusterId++;
        labels[i] = clusterId;

        int head = 0, tail = 0;

        // Add neighbors of i as seeds
        for (int j = 0; j < N; j++) {
            if (j == i) continue;
            float dx = xs[i] - xs[j];
            float dy = ys[i] - ys[j];
            if (dx*dx + dy*dy <= eps2) {
                if (labels[j] == UNPROCESSED) seeds[tail++] = j;
                if (labels[j] == UNPROCESSED || labels[j] == NOISE)
                    labels[j] = clusterId;
            }
        }

        // BFS expand
        while (head < tail) {
            int q = seeds[head++];
            if (h_counts[q] < minPts) continue;

            for (int j = 0; j < N; j++) {
                if (labels[j] != UNPROCESSED && labels[j] != NOISE) continue;
                float dx = xs[q] - xs[j];
                float dy = ys[q] - ys[j];
                if (dx*dx + dy*dy <= eps2) {
                    if (labels[j] == UNPROCESSED) seeds[tail++] = j;
                    labels[j] = clusterId;
                }
            }
        }
    }

    free(seeds);
    free(h_counts);

    cudaDeviceSynchronize();
}
