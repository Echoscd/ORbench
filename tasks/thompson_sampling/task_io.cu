// task_io.cu — thompson_sampling GPU I/O adapter (compute_only interface)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_compute(
    int N, int T, int M,
    const float* arm_means,
    uint64_t seed,
    float* avg_regret_out,
    float* avg_counts_out
);
extern void solution_free(void);

#ifdef __cplusplus
}
#endif

typedef struct {
    int N, T, M;
    uint64_t seed;
    const float* arm_means;
    float  avg_regret;
    float* avg_counts;
} TSContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    TSContext* ctx = (TSContext*)calloc(1, sizeof(TSContext));
    ctx->N    = (int)get_param(data, "N");
    ctx->T    = (int)get_param(data, "T");
    ctx->M    = (int)get_param(data, "M");
    ctx->seed = (uint64_t)get_param(data, "seed");
    ctx->arm_means = get_tensor_float(data, "arm_means");
    if (!ctx->arm_means) {
        fprintf(stderr, "[task_io] Missing arm_means tensor\n");
        free(ctx);
        return NULL;
    }
    ctx->avg_counts = (float*)calloc(ctx->N, sizeof(float));
    ctx->avg_regret = 0.0f;
    return ctx;
}

void task_run(void* test_data) {
    TSContext* ctx = (TSContext*)test_data;
    solution_compute(
        ctx->N, ctx->T, ctx->M,
        ctx->arm_means, ctx->seed,
        &ctx->avg_regret, ctx->avg_counts
    );
}

void task_write_output(void* test_data, const char* output_path) {
    TSContext* ctx = (TSContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    fprintf(f, "%.6e\n", ctx->avg_regret);
    for (int i = 0; i < ctx->N; i++)
        fprintf(f, "%.6e\n", ctx->avg_counts[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TSContext* ctx = (TSContext*)test_data;
    solution_free();
    free(ctx->avg_counts);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
