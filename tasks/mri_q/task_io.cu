// task_io.cu -- MRI-Q GPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_init(
    int num_k,
    int num_x,
    const float* h_kx,
    const float* h_ky,
    const float* h_kz,
    const float* h_phi_r,
    const float* h_phi_i,
    const float* h_x,
    const float* h_y,
    const float* h_z
);

extern void solution_compute(float* h_qr, float* h_qi);
extern void solution_free(void);

#ifdef __cplusplus
}
#endif

typedef struct {
    int num_x;
    int sample_stride;
    float* qr;
    float* qi;
} TaskIOContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int num_k = (int)get_param(data, "num_k");
    int num_x = (int)get_param(data, "num_x");
    int sample_stride = (int)get_param(data, "sample_stride");

    const float* kx    = get_tensor_float(data, "kx");
    const float* ky    = get_tensor_float(data, "ky");
    const float* kz    = get_tensor_float(data, "kz");
    const float* phi_r = get_tensor_float(data, "phi_r");
    const float* phi_i = get_tensor_float(data, "phi_i");
    const float* x     = get_tensor_float(data, "x");
    const float* y     = get_tensor_float(data, "y");
    const float* z     = get_tensor_float(data, "z");

    if (!kx || !ky || !kz || !phi_r || !phi_i || !x || !y || !z) {
        fprintf(stderr, "[task_io] Missing tensor(s)\n");
        return NULL;
    }

    solution_init(num_k, num_x, kx, ky, kz, phi_r, phi_i, x, y, z);

    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    if (!ctx) return NULL;
    ctx->num_x = num_x;
    ctx->sample_stride = sample_stride;
    ctx->qr = (float*)calloc((size_t)num_x, sizeof(float));
    ctx->qi = (float*)calloc((size_t)num_x, sizeof(float));
    if (!ctx->qr || !ctx->qi) {
        free(ctx->qr);
        free(ctx->qi);
        free(ctx);
        return NULL;
    }
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->qr, ctx->qi);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    int stride = ctx->sample_stride > 0 ? ctx->sample_stride : 1;
    for (int i = 0; i < ctx->num_x; i += stride) {
        fprintf(f, "%.6e %.6e\n", ctx->qr[i], ctx->qi[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    if (!ctx) return;
    solution_free();
    free(ctx->qr);
    free(ctx->qi);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
