// task_io_cpu.c — motzkin_straus_blp_eval CPU I/O adapter (compute_only)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void solution_compute(int N, int M, float mu, const float* A, const float* x, const float* q, const float* s, float* obj, float* max_viol);
extern void solution_free(void);

typedef struct {
    int N, M;
    float mu;
    float *A, *x, *q, *s;
    float *obj, *max_viol;
} Ctx;

void* task_setup(const TaskData* data, const char* data_dir) {
    Ctx* ctx = (Ctx*)calloc(1, sizeof(Ctx));
    ctx->N = (int)get_param(data, "N");
    ctx->M = (int)get_param(data, "M");

    float* fp = get_tensor_float(data, "fparams");
    ctx->mu = fp[0];

    ctx->A = get_tensor_float(data, "A");
    ctx->x = get_tensor_float(data, "x");
    ctx->q = get_tensor_float(data, "q");
    ctx->s = get_tensor_float(data, "s");

    ctx->obj      = (float*)calloc((size_t)ctx->M, sizeof(float));
    ctx->max_viol = (float*)calloc((size_t)ctx->M, sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    Ctx* ctx = (Ctx*)test_data;
    solution_compute(ctx->N, ctx->M, ctx->mu, ctx->A, ctx->x, ctx->q, ctx->s, ctx->obj, ctx->max_viol);
}

void task_write_output(void* test_data, const char* output_path) {
    Ctx* ctx = (Ctx*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->M; i++)
        fprintf(f, "%.6f\n", ctx->obj[i]);
    for (int i = 0; i < ctx->M; i++)
        fprintf(f, "%.6f\n", ctx->max_viol[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    Ctx* ctx = (Ctx*)test_data;
    solution_free();
    free(ctx->obj);
    free(ctx->max_viol);
    free(ctx);
}
