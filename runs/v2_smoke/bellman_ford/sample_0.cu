// ORBench v2.1 smoke-test solution for bellman_ford
//
// Implements the two LLM-facing functions:
//   solution_init / solution_compute
//
// NOTE: This is a CPU-based implementation wrapped in .cu for correctness-first.
// It exercises the full harness → task_io → solution pipeline end-to-end.
// A real LLM solution would use CUDA kernels for GPU acceleration.

#include <stdlib.h>
#include <string.h>

#define INF_VAL 1e30f

// ===== Module-level state =====
static int g_V, g_E;
static const int*   g_row_offsets;
static const int*   g_col_indices;
static const float* g_weights;
static float*       g_dist_buf;   // working buffer, size V

extern "C" void solution_init(int V, int E,
                               const int* h_row_offsets,
                               const int* h_col_indices,
                               const float* h_weights) {
    g_V = V;
    g_E = E;
    g_row_offsets = h_row_offsets;
    g_col_indices = h_col_indices;
    g_weights     = h_weights;
    g_dist_buf    = (float*)malloc((size_t)V * sizeof(float));
}

extern "C" void solution_compute(int num_requests,
                                  const int* sources,
                                  const int* targets,
                                  float* distances) {
    for (int r = 0; r < num_requests; r++) {
        int src = sources[r];
        int tgt = targets[r];

        // Bellman-Ford from src
        for (int i = 0; i < g_V; i++) g_dist_buf[i] = INF_VAL;
        if (src >= 0 && src < g_V) g_dist_buf[src] = 0.0f;

        for (int round = 0; round < g_V - 1; round++) {
            int updated = 0;
            for (int u = 0; u < g_V; u++) {
                float du = g_dist_buf[u];
                if (du >= INF_VAL) continue;
                int start = g_row_offsets[u];
                int end   = g_row_offsets[u + 1];
                for (int idx = start; idx < end; idx++) {
                    float nd = du + g_weights[idx];
                    if (nd < g_dist_buf[g_col_indices[idx]]) {
                        g_dist_buf[g_col_indices[idx]] = nd;
                        updated = 1;
                    }
                }
            }
            if (!updated) break;
        }

        distances[r] = (tgt >= 0 && tgt < g_V) ? g_dist_buf[tgt] : INF_VAL;
    }
}







