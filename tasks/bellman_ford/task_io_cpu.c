// task_io_cpu.c — bellman_ford CPU I/O adapter layer
//
// Same logic as task_io.cu, but pure C (no cuda_runtime.h).
// Build: gcc -O2 -I framework/
//        framework/harness_cpu.c tasks/bellman_ford/task_io_cpu.c
//        tasks/bellman_ford/cpu_reference.c -o solution_cpu -lm

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ===== LLM / CPU-baseline interface (same three functions) =====
extern void solution_init(int V, int E,
                          const int* h_row_offsets,
                          const int* h_col_indices,
                          const float* h_weights);

extern void solution_compute(int num_requests,
                             const int* h_sources,
                             const int* h_targets,
                             float* h_distances);

extern void solution_free(void);

// ===== task_io internal state (identical to task_io.cu) =====
typedef struct {
    int num_requests;
    int* sources;
    int* targets;
    float* distances;
} TaskIOContext;

static TaskIOContext* parse_requests(const char* data_dir) {
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    char path[512];
    snprintf(path, sizeof(path), "%s/requests.txt", data_dir);
    FILE* f = fopen(path, "r");
    if (!f) { free(ctx); return NULL; }

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
void* task_setup(const TaskData* data, const char* data_dir) {
    int V = (int)get_param(data, "V");
    int E = (int)get_param(data, "E");
    solution_init(V, E,
                  get_tensor_int(data, "row_offsets"),
                  get_tensor_int(data, "col_indices"),
                  get_tensor_float(data, "weights"));

    TaskIOContext* ctx = parse_requests(data_dir);
    if (!ctx) {
        fprintf(stderr, "[task_io] Failed to parse requests.txt\n");
        return NULL;
    }
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->num_requests,
                     ctx->sources,
                     ctx->targets,
                     ctx->distances);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
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

