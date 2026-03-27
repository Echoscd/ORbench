// task_io_cpu.c — self_exciting_pricing_dp CPU I/O adapter (compute_only)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void solution_compute(
    int N_x, int N_T, int N_lambda,
    float dx, float dt, float alpha, float beta, float phi,
    float c, float a, float gamma, float k,
    float lambda_min, float lambda_max,
    float* V_out
);
extern void solution_free(void);

typedef struct {
    int N_x, N_T, N_lambda;
    float dx, dt, alpha, beta, phi, c, a, gamma, k, lambda_min, lambda_max;
    float* V_out;
} Ctx;

void* task_setup(const TaskData* data, const char* data_dir) {
    Ctx* ctx = (Ctx*)calloc(1, sizeof(Ctx));
    ctx->N_x      = (int)get_param(data, "N_x");
    ctx->N_T      = (int)get_param(data, "N_T");
    ctx->N_lambda = (int)get_param(data, "N_lambda");

    float* fp = get_tensor_float(data, "float_params");
    ctx->dx         = fp[0];
    ctx->dt         = fp[1];
    ctx->alpha      = fp[2];
    ctx->beta       = fp[3];
    ctx->phi        = fp[4];
    ctx->c          = fp[5];
    ctx->a          = fp[6];
    ctx->gamma      = fp[7];
    ctx->k          = fp[8];
    ctx->lambda_min = fp[9];
    ctx->lambda_max = fp[10];

    ctx->V_out = (float*)calloc((size_t)ctx->N_x, sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    Ctx* ctx = (Ctx*)test_data;
    solution_compute(
        ctx->N_x, ctx->N_T, ctx->N_lambda,
        ctx->dx, ctx->dt, ctx->alpha, ctx->beta, ctx->phi,
        ctx->c, ctx->a, ctx->gamma, ctx->k,
        ctx->lambda_min, ctx->lambda_max,
        ctx->V_out
    );
}

void task_write_output(void* test_data, const char* output_path) {
    Ctx* ctx = (Ctx*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N_x; i++)
        fprintf(f, "%.6f\n", ctx->V_out[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    Ctx* ctx = (Ctx*)test_data;
    solution_free();
    free(ctx->V_out);
    free(ctx);
}
