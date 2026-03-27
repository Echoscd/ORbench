// task_io.cu — gittins_index GPU I/O adapter (compute_only interface)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_compute(
    int N, int a_x10000, int S, int num_bisect,
    float* V_out
);
extern void solution_free(void);

#ifdef __cplusplus
}
#endif

typedef struct {
    int N, a_x10000, S, num_bisect;
    float* V_out;
} GittinsContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    GittinsContext* ctx = (GittinsContext*)calloc(1, sizeof(GittinsContext));
    ctx->N          = (int)get_param(data, "N");
    ctx->a_x10000   = (int)get_param(data, "a_x10000");
    ctx->S          = (int)get_param(data, "S");
    ctx->num_bisect = (int)get_param(data, "num_bisect");
    ctx->V_out      = (float*)calloc((size_t)ctx->S, sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    GittinsContext* ctx = (GittinsContext*)test_data;
    solution_compute(ctx->N, ctx->a_x10000, ctx->S, ctx->num_bisect, ctx->V_out);
}

void task_write_output(void* test_data, const char* output_path) {
    GittinsContext* ctx = (GittinsContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->S; i++)
        fprintf(f, "%.6e\n", ctx->V_out[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    GittinsContext* ctx = (GittinsContext*)test_data;
    solution_free();
    free(ctx->V_out);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
