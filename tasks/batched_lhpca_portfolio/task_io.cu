// task_io.cu — batched_lhpca_portfolio GPU I/O adapter (compute_only)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_compute(int S, int T, int N, int K, const float* R, float* w);
extern void solution_free(void);

#ifdef __cplusplus
}
#endif

typedef struct {
    int S, T, N, K;
    float* R;
    float* w;
} Ctx;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    Ctx* ctx = (Ctx*)calloc(1, sizeof(Ctx));
    ctx->S = (int)get_param(data, "S");
    ctx->T = (int)get_param(data, "T");
    ctx->N = (int)get_param(data, "N");
    ctx->K = (int)get_param(data, "K");
    ctx->R = get_tensor_float(data, "R");
    ctx->w = (float*)calloc((size_t)ctx->S * ctx->N, sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    Ctx* ctx = (Ctx*)test_data;
    solution_compute(ctx->S, ctx->T, ctx->N, ctx->K, ctx->R, ctx->w);
}

void task_write_output(void* test_data, const char* output_path) {
    Ctx* ctx = (Ctx*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->S * ctx->N; i++)
        fprintf(f, "%.6e\n", ctx->w[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    Ctx* ctx = (Ctx*)test_data;
    solution_free();
    free(ctx->w);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
