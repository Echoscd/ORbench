// task_io.cu — bellman_ford GPU I/O adapter layer (task author writes, LLM does NOT touch)
//
// Three-layer architecture: harness → task_io → solution
// This file bridges harness (generic) and solution (LLM-written pure computation).
//
// Build: nvcc -O2 -arch=sm_89 -I framework/
//        framework/harness_gpu.cu tasks/bellman_ford/task_io.cu solution.cu -o solution_gpu

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// ===== LLM-implemented interface (declaration) =====
#ifdef __cplusplus
extern "C" {
#endif

// solution_init: receive host-side graph data, do cudaMalloc + H2D etc.
// Called once, NOT timed.
extern void solution_init(int V, int E,
                          const int* h_row_offsets,
                          const int* h_col_indices,
                          const float* h_weights);

// solution_compute: process a batch of (source, target) queries, write results to distances
// Called multiple times (warmup + timed), MUST be idempotent.
// distances[i] = shortest distance from sources[i] to targets[i]
extern void solution_compute(int num_requests,
                             const int* h_sources,
                             const int* h_targets,
                             float* h_distances);

// solution_free: release GPU resources
// Called once, NOT timed.
extern void solution_free(void);

#ifdef __cplusplus
}
#endif

// ===== task_io internal state =====
typedef struct {
    int num_requests;
    int* sources;
    int* targets;
    float* distances;  // output buffer
} TaskIOContext;

// ===== Parse requests.txt =====
static TaskIOContext* parse_requests(const char* data_dir) {
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    char path[512];
    snprintf(path, sizeof(path), "%s/requests.txt", data_dir);
    FILE* f = fopen(path, "r");
    if (!f) { free(ctx); return NULL; }

    // Count lines
    char line[256];
    int n = 0;
    while (fgets(line, sizeof(line), f)) {
        int s, t;
        if (sscanf(line, "%d %d", &s, &t) == 2) n++;
    }
    rewind(f);

    ctx->num_requests = n;
    ctx->sources   = (int*)malloc(n * sizeof(int));
    ctx->targets   = (int*)malloc(n * sizeof(int));
    ctx->distances = (float*)calloc(n, sizeof(float));

    int idx = 0;
    while (fgets(line, sizeof(line), f)) {
        int s, t;
        if (sscanf(line, "%d %d", &s, &t) == 2) {
            ctx->sources[idx] = s;
            ctx->targets[idx] = t;
            idx++;
        }
    }
    fclose(f);
    return ctx;
}

// ===== harness calls these four functions =====
#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    // 1. Extract graph from TaskData, pass host pointers to LLM's solution_init
    int V = (int)get_param(data, "V");
    int E = (int)get_param(data, "E");
    solution_init(V, E,
                  get_tensor_int(data, "row_offsets"),
                  get_tensor_int(data, "col_indices"),
                  get_tensor_float(data, "weights"));

    // 2. Parse requests.txt (task author knows the format)
    TaskIOContext* ctx = parse_requests(data_dir);
    if (!ctx) {
        fprintf(stderr, "[task_io] Failed to parse requests.txt\n");
        return NULL;
    }
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    // Delegate to LLM's pure computation function
    solution_compute(ctx->num_requests,
                     ctx->sources,
                     ctx->targets,
                     ctx->distances);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    // Output format controlled by task_io, guaranteed consistent with expected_output.txt
    for (int i = 0; i < ctx->num_requests; i++)
        fprintf(f, "%.6e\n", ctx->distances[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    solution_free();
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    free(ctx->sources);
    free(ctx->targets);
    free(ctx->distances);
    free(ctx);
}

#ifdef __cplusplus
}
#endif

