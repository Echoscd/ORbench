// task_io_cpu.c — hawkes_dynamic_pricing_hjb CPU I/O adapter (compute_only)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void solution_compute(
    int batch_size, int J, int N, int num_actions,
    float h, float tau, float lambda0, float a, float b_param,
    const float* alpha0, const float* beta,
    float* U, float* Lambda
);
extern void solution_free(void);

typedef struct {
    int batch_size, J, N, num_actions;
    float h, tau, lambda0, a, b_param;
    float* alpha0;
    float* beta;
    float* U;
    float* Lambda;
} Ctx;

void* task_setup(const TaskData* data, const char* data_dir) {
    Ctx* ctx = (Ctx*)calloc(1, sizeof(Ctx));
    ctx->batch_size  = (int)get_param(data, "batch_size");
    ctx->J           = (int)get_param(data, "J");
    ctx->N           = (int)get_param(data, "N");
    ctx->num_actions = (int)get_param(data, "num_actions");
    ctx->h           = (float)get_param(data, "h_x1e6") / 1e6f;
    ctx->tau         = (float)get_param(data, "tau_x1e6") / 1e6f;
    ctx->lambda0     = (float)get_param(data, "lambda0_x1e6") / 1e6f;
    ctx->a           = (float)get_param(data, "a_x1e6") / 1e6f;
    ctx->b_param     = (float)get_param(data, "b_param_x1e6") / 1e6f;
    ctx->alpha0      = get_tensor_float(data, "alpha0");
    ctx->beta        = get_tensor_float(data, "beta");

    size_t out_size = (size_t)ctx->batch_size * ctx->J * ctx->N;
    ctx->U      = (float*)calloc(out_size, sizeof(float));
    ctx->Lambda = (float*)calloc(out_size, sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    Ctx* ctx = (Ctx*)test_data;
    solution_compute(
        ctx->batch_size, ctx->J, ctx->N, ctx->num_actions,
        ctx->h, ctx->tau, ctx->lambda0, ctx->a, ctx->b_param,
        ctx->alpha0, ctx->beta,
        ctx->U, ctx->Lambda
    );
}

void task_write_output(void* test_data, const char* output_path) {
    Ctx* ctx = (Ctx*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int b = 0; b < ctx->batch_size; b++)
        for (int j = 0; j < ctx->J; j++)
            fprintf(f, "%.6f\n", ctx->U[(b * ctx->J + j) * ctx->N]);
    for (int b = 0; b < ctx->batch_size; b++)
        for (int j = 0; j < ctx->J; j++)
            fprintf(f, "%.6f\n", ctx->Lambda[(b * ctx->J + j) * ctx->N]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    Ctx* ctx = (Ctx*)test_data;
    solution_free();
    free(ctx->U);
    free(ctx->Lambda);
    free(ctx);
}
