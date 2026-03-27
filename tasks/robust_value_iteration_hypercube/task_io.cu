// task_io.cu — robust_value_iteration_hypercube GPU I/O adapter (compute_only interface)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_compute(int S, int A, int T, float gamma, const float* rew, const float* P_up, const float* P_down, float* V);
extern void solution_free(void);

#ifdef __cplusplus
}
#endif

typedef struct {
    int S, A, T;
    float gamma;
    float* rew;
    float* P_up;
    float* P_down;
    float* V;
} Ctx;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    Ctx* ctx = (Ctx*)calloc(1, sizeof(Ctx));
    ctx->S     = (int)get_param(data, "S");
    ctx->A     = (int)get_param(data, "A");
    ctx->T     = (int)get_param(data, "T");
    ctx->gamma = (float)get_param(data, "gamma_x10000") / 10000.0f;
    ctx->rew   = get_tensor_float(data, "rew");
    ctx->P_up  = get_tensor_float(data, "P_up");
    ctx->P_down = get_tensor_float(data, "P_down");
    ctx->V     = (float*)calloc((size_t)ctx->S, sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    Ctx* ctx = (Ctx*)test_data;
    solution_compute(ctx->S, ctx->A, ctx->T, ctx->gamma, ctx->rew, ctx->P_up, ctx->P_down, ctx->V);
}

void task_write_output(void* test_data, const char* output_path) {
    Ctx* ctx = (Ctx*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->S; i++)
        fprintf(f, "%.6f\n", ctx->V[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    Ctx* ctx = (Ctx*)test_data;
    solution_free();
    free(ctx->V);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
