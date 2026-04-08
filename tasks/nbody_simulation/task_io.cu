// task_io.cu -- nbody_simulation GPU I/O adapter (compute_only interface)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
extern void solution_compute(
    int N,
    float softening,
    const float* pos_x,
    const float* pos_y,
    const float* pos_z,
    const float* mass,
    float* fx,
    float* fy,
    float* fz
);
extern void solution_free(void);
#ifdef __cplusplus
}
#endif

typedef struct {
    int N;
    float softening;
    const float* pos_x;
    const float* pos_y;
    const float* pos_z;
    const float* mass;
    float* fx;
    float* fy;
    float* fz;
} NBodyContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    NBodyContext* ctx = (NBodyContext*)calloc(1, sizeof(NBodyContext));
    if (!ctx) return NULL;

    ctx->N = (int)get_param(data, "N");
    const int softening_x1e6 = (int)get_param(data, "softening_x1e6");
    ctx->softening = ((float)softening_x1e6) / 1000000.0f;

    ctx->pos_x = get_tensor_float(data, "pos_x");
    ctx->pos_y = get_tensor_float(data, "pos_y");
    ctx->pos_z = get_tensor_float(data, "pos_z");
    ctx->mass  = get_tensor_float(data, "mass");

    if (!ctx->pos_x || !ctx->pos_y || !ctx->pos_z || !ctx->mass) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }

    ctx->fx = (float*)calloc((size_t)ctx->N, sizeof(float));
    ctx->fy = (float*)calloc((size_t)ctx->N, sizeof(float));
    ctx->fz = (float*)calloc((size_t)ctx->N, sizeof(float));
    if (!ctx->fx || !ctx->fy || !ctx->fz) {
        free(ctx->fx); free(ctx->fy); free(ctx->fz); free(ctx);
        return NULL;
    }
    return ctx;
}

void task_run(void* test_data) {
    NBodyContext* ctx = (NBodyContext*)test_data;
    solution_compute(ctx->N, ctx->softening,
                     ctx->pos_x, ctx->pos_y, ctx->pos_z, ctx->mass,
                     ctx->fx, ctx->fy, ctx->fz);
}

void task_write_output(void* test_data, const char* output_path) {
    NBodyContext* ctx = (NBodyContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++) {
        fprintf(f, "%.6e\n", ctx->fx[i]);
        fprintf(f, "%.6e\n", ctx->fy[i]);
        fprintf(f, "%.6e\n", ctx->fz[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    NBodyContext* ctx = (NBodyContext*)test_data;
    solution_free();
    free(ctx->fx);
    free(ctx->fy);
    free(ctx->fz);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
