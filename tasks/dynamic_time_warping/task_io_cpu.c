// task_io_cpu.c -- dynamic_time_warping CPU I/O adapter (compute_only interface)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_compute(
    int N,
    int total_query_len,
    int total_target_len,
    const int* query_series,
    const int* target_series,
    const int* query_offsets,
    const int* target_offsets,
    int* distances
);

extern void solution_free(void);

typedef struct {
    int N;
    int total_query_len;
    int total_target_len;
    const int* query_series;
    const int* target_series;
    const int* query_offsets;
    const int* target_offsets;
    int* distances;
} DTWContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;

    DTWContext* ctx = (DTWContext*)calloc(1, sizeof(DTWContext));
    if (!ctx) return NULL;

    ctx->N = (int)get_param(data, "N");
    ctx->total_query_len = (int)get_param(data, "total_query_len");
    ctx->total_target_len = (int)get_param(data, "total_target_len");
    ctx->query_series = get_tensor_int(data, "query_series");
    ctx->target_series = get_tensor_int(data, "target_series");
    ctx->query_offsets = get_tensor_int(data, "query_offsets");
    ctx->target_offsets = get_tensor_int(data, "target_offsets");

    if (!ctx->query_series || !ctx->target_series || !ctx->query_offsets || !ctx->target_offsets) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }

    ctx->distances = (int*)calloc((size_t)ctx->N, sizeof(int));
    if (!ctx->distances) {
        free(ctx);
        return NULL;
    }

    return ctx;
}

void task_run(void* test_data) {
    DTWContext* ctx = (DTWContext*)test_data;
    solution_compute(
        ctx->N,
        ctx->total_query_len,
        ctx->total_target_len,
        ctx->query_series,
        ctx->target_series,
        ctx->query_offsets,
        ctx->target_offsets,
        ctx->distances
    );
}

void task_write_output(void* test_data, const char* output_path) {
    DTWContext* ctx = (DTWContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++) {
        fprintf(f, "%d\n", ctx->distances[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    DTWContext* ctx = (DTWContext*)test_data;
    solution_free();
    free(ctx->distances);
    free(ctx);
}
