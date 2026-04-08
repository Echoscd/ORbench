// task_io_cpu.c -- asian_option_pricing_mc CPU I/O adapter (compute_only interface)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_compute(
    int N,
    int num_paths,
    int num_steps,
    const float* s0,
    const float* strike,
    const float* rate,
    const float* sigma,
    const float* maturity,
    const int* option_type,
    const float* shocks,
    float* prices
);

extern void solution_free(void);

typedef struct {
    int N;
    int num_paths;
    int num_steps;
    const float* s0;
    const float* strike;
    const float* rate;
    const float* sigma;
    const float* maturity;
    const int* option_type;
    const float* shocks;
    float* prices;
} AsianMCContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;

    AsianMCContext* ctx = (AsianMCContext*)calloc(1, sizeof(AsianMCContext));
    if (!ctx) return NULL;

    ctx->N = (int)get_param(data, "N");
    ctx->num_paths = (int)get_param(data, "num_paths");
    ctx->num_steps = (int)get_param(data, "num_steps");

    ctx->s0 = get_tensor_float(data, "s0");
    ctx->strike = get_tensor_float(data, "strike");
    ctx->rate = get_tensor_float(data, "rate");
    ctx->sigma = get_tensor_float(data, "sigma");
    ctx->maturity = get_tensor_float(data, "maturity");
    ctx->option_type = get_tensor_int(data, "option_type");
    ctx->shocks = get_tensor_float(data, "shocks");

    if (!ctx->s0 || !ctx->strike || !ctx->rate || !ctx->sigma || !ctx->maturity || !ctx->option_type || !ctx->shocks) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }

    ctx->prices = (float*)calloc((size_t)ctx->N, sizeof(float));
    if (!ctx->prices) {
        free(ctx);
        return NULL;
    }

    return ctx;
}

void task_run(void* test_data) {
    AsianMCContext* ctx = (AsianMCContext*)test_data;
    solution_compute(
        ctx->N,
        ctx->num_paths,
        ctx->num_steps,
        ctx->s0,
        ctx->strike,
        ctx->rate,
        ctx->sigma,
        ctx->maturity,
        ctx->option_type,
        ctx->shocks,
        ctx->prices
    );
}

void task_write_output(void* test_data, const char* output_path) {
    AsianMCContext* ctx = (AsianMCContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++) {
        fprintf(f, "%.8f\n", ctx->prices[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    AsianMCContext* ctx = (AsianMCContext*)test_data;
    solution_free();
    free(ctx->prices);
    free(ctx);
}
