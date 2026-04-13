#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_init(int V, int E, int max_iters, float damping, float epsilon,
                          const int* in_row_offsets,
                          const int* in_col_indices,
                          const int* out_degree);
extern void solution_compute(float* scores_out);
extern void solution_free(void);

typedef struct {
    int V;
    float* scores_out;
} Ctx;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    Ctx* ctx = (Ctx*)calloc(1, sizeof(Ctx));
    ctx->V = (int)get_param(data, "V");
    int E = (int)get_param(data, "E");
    int max_iters = (int)get_param(data, "max_iters");
    float* fp = get_tensor_float(data, "fparams");
    float damping = fp[0];
    float epsilon = fp[1];

    solution_init(ctx->V, E, max_iters, damping, epsilon,
                  get_tensor_int(data, "in_row_offsets"),
                  get_tensor_int(data, "in_col_indices"),
                  get_tensor_int(data, "out_degree"));

    ctx->scores_out = (float*)calloc((size_t)ctx->V, sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    Ctx* ctx = (Ctx*)test_data;
    solution_compute(ctx->scores_out);
}

void task_write_output(void* test_data, const char* output_path) {
    Ctx* ctx = (Ctx*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->V; i++) {
        fprintf(f, "%.8e\n", ctx->scores_out[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    Ctx* ctx = (Ctx*)test_data;
    solution_free();
    free(ctx->scores_out);
    free(ctx);
}
