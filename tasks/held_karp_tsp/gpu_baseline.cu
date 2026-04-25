// gpu_baseline.cu — Held-Karp DP for TSP Tour Cost (GPU baseline)
//
// Faithfully derived from Google OR-Tools hamiltonian_path.h::Solve()
// The DP recurrence is identical:
//   f(S, j) = min_{i in S\{j}} (f(S\{j}, i) + cost(i, j))
// with f({j}, j) = cost(0, j) and TSP_cost = f(V, 0).
//
// GPU adaptation: uses the simpler bitmask-indexed dp[mask * n + j] layout
// (vs or-tools' LatticeMemoryManager) because GPU memory access patterns
// differ from CPU. One CUDA thread per (mask, j) pair within each
// cardinality level, synchronized between levels via kernel launches.
//
// For each batch instance b, the GPU runs the full DP on device.

#include <cuda_runtime.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INF_COST 1000000000

// ===== DP kernel: one thread per (mask, j) pair at given cardinality =====
__global__ void held_karp_kernel(int n,
                                int subset_count,
                                int target_card,
                                const int* __restrict__ cost,
                                int* __restrict__ dp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = subset_count * n;
    if (idx >= total) return;

    int mask = idx / n;
    int j = idx % n;
    if (j == 0) return;  // node 0 is depot, skip
    if (__popc(mask) != target_card) return;

    int bitj = 1 << (j - 1);
    if ((mask & bitj) == 0) return;  // j not in set

    int prev_mask = mask ^ bitj;
    if (prev_mask == 0) return;  // singleton handled in init

    int best = INF_COST;
    int pm = prev_mask;
    while (pm) {
        int lowbit = pm & (-pm);
        int k = __ffs(lowbit);  // 1-indexed position
        int cand = dp[prev_mask * n + k] + cost[k * n + j];
        if (cand < best) best = cand;
        pm ^= lowbit;
    }
    dp[mask * n + j] = best;
}

// ===== Init kernel: f({j}, j) = cost(0, j) =====
__global__ void held_karp_init(int n, const int* __restrict__ cost,
                               int* __restrict__ dp, int subset_count)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j == 0 || j >= n) return;
    int mask = 1 << (j - 1);
    dp[mask * n + j] = cost[j];  // cost[0 * n + j]
}

// ===== Final reduction: TSP cost = min_j f(full, j) + cost(j, 0) =====
__global__ void held_karp_final(int n, int full_mask,
                                const int* __restrict__ cost,
                                const int* __restrict__ dp,
                                int* __restrict__ result)
{
    // Single thread does the final reduction (n is small)
    if (threadIdx.x != 0) return;
    int best = INF_COST;
    for (int j = 1; j < n; j++) {
        int cand = dp[full_mask * n + j] + cost[j * n + 0];
        if (cand < best) best = cand;
    }
    *result = best;
}

// ===== Host interface =====

static int    g_n = 0;
static int*   d_cost = nullptr;
static int*   d_dp = nullptr;
static int*   d_result = nullptr;

extern "C" void solution_free(void) {
    if (d_cost)   { cudaFree(d_cost);   d_cost   = nullptr; }
    if (d_dp)     { cudaFree(d_dp);     d_dp     = nullptr; }
    if (d_result) { cudaFree(d_result); d_result = nullptr; }
    g_n = 0;
}

extern "C" void solution_compute(int B, int n, const int* costs,
                                 int* tour_costs_out)
{
    int m = n - 1;
    int subset_count = 1 << m;
    size_t dp_bytes = (size_t)subset_count * n * sizeof(int);
    size_t cost_bytes = (size_t)n * n * sizeof(int);

    if (g_n != n) {
        solution_free();
        cudaMalloc(&d_cost, cost_bytes);
        cudaMalloc(&d_dp, dp_bytes);
        cudaMalloc(&d_result, sizeof(int));
        g_n = n;
    }

    int block = 256;

    for (int b = 0; b < B; b++) {
        const int* cost = costs + (size_t)b * n * n;

        // Upload cost matrix
        cudaMemcpy(d_cost, cost, cost_bytes, cudaMemcpyHostToDevice);

        // Clear DP table
        cudaMemset(d_dp, 0x3f, dp_bytes);  // fill with large value

        // Init: f({j}, j) = cost(0, j)
        int g1 = (n + block - 1) / block;
        held_karp_init<<<g1, block>>>(n, d_cost, d_dp, subset_count);

        // DP by cardinality
        int total = subset_count * n;
        int grid = (total + block - 1) / block;
        for (int card = 2; card <= m; card++) {
            held_karp_kernel<<<grid, block>>>(n, subset_count, card,
                                              d_cost, d_dp);
        }

        // Extract TSP cost
        int full_mask = subset_count - 1;
        held_karp_final<<<1, 32>>>(n, full_mask, d_cost, d_dp, d_result);

        cudaMemcpy(&tour_costs_out[b], d_result, sizeof(int),
                   cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
}
