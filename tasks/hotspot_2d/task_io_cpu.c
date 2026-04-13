#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../framework/orbench_io.h"

typedef struct {
    int rows;
    int cols;
    int iterations;
    int output_stride;
    int num_samples;
    float* sampled_out;
} HotspotCtx;

void solution_init(
    int rows,
    int cols,
    int iterations,
    const float* h_temp_init,
    const float* h_power
);
void solution_compute(int output_stride, float* h_sampled_out);
void solution_free(void);

static int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

void* task_setup(const TaskData* data, const char* data_dir) {
    HotspotCtx* ctx = (HotspotCtx*)calloc(1, sizeof(HotspotCtx));
    int total;
    (void)data_dir;
    if (!ctx) return NULL;

    ctx->rows = (int)get_param(data, "rows");
    ctx->cols = (int)get_param(data, "cols");
    ctx->iterations = (int)get_param(data, "iterations");
    ctx->output_stride = (int)get_param(data, "output_stride");
    total = ctx->rows * ctx->cols;
    ctx->num_samples = ceil_div_int(total, ctx->output_stride);
    ctx->sampled_out = (float*)malloc((size_t)ctx->num_samples * sizeof(float));
    if (!ctx->sampled_out) {
        free(ctx);
        return NULL;
    }

    solution_init(
        ctx->rows,
        ctx->cols,
        ctx->iterations,
        get_tensor_float(data, "temp_init"),
        get_tensor_float(data, "power")
    );
    return ctx;
}

void task_run(void* vctx) {
    HotspotCtx* ctx = (HotspotCtx*)vctx;
    solution_compute(ctx->output_stride, ctx->sampled_out);
}

void task_write_output(void* vctx, const char* output_path) {
    HotspotCtx* ctx = (HotspotCtx*)vctx;
    FILE* f = fopen(output_path, "w");
    int i;
    if (!f) return;
    for (i = 0; i < ctx->num_samples; ++i) {
        fprintf(f, "%.6f\n", ctx->sampled_out[i]);
    }
    fclose(f);
}

void task_cleanup(void* vctx) {
    HotspotCtx* ctx = (HotspotCtx*)vctx;
    if (!ctx) return;
    solution_free();
    if (ctx->sampled_out) free(ctx->sampled_out);
    free(ctx);
}
