#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_init(int n, const double* h_matrix);
extern void solution_compute(double* h_out_w);
extern void solution_free(void);

typedef struct {
    int n;
    const double* matrix;
    double* out_w;
} Ctx;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    Ctx* ctx = (Ctx*)calloc(1, sizeof(Ctx));
    if (!ctx) return NULL;
    ctx->n = (int)get_param(data, "n");
    ctx->matrix = (const double*)get_tensor(data, "matrix");
    if (!ctx->matrix) { free(ctx); return NULL; }
    ctx->out_w = (double*)calloc((size_t)ctx->n, sizeof(double));
    if (!ctx->out_w) { free(ctx); return NULL; }
    solution_init(ctx->n, ctx->matrix);
    return ctx;
}

void task_run(void* test_data) {
    Ctx* ctx = (Ctx*)test_data;
    solution_compute(ctx->out_w);
}

void task_write_output(void* test_data, const char* output_path) {
    Ctx* ctx = (Ctx*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->n; ++i) {
        fprintf(f, "%.17g\n", ctx->out_w[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    Ctx* ctx = (Ctx*)test_data;
    solution_free();
    free(ctx->out_w);
    free(ctx);
}
