// task_io.cu — inventory_replenishment_dp GPU I/O adapter (compute_only)

#include "orbench_io.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_compute(
    int N_I, int N_B, int N_Phi, int N_Psi, int N_x, int N_phi, int T,
    const float* c_t, const float* h_t, const float* b_t,
    float mu, float nu, float alpha, float y,
    const float* grid_I, const float* grid_B, const float* grid_Phi, const float* grid_Psi,
    const float* actions, const float* shocks, const float* shock_probs,
    float* V_out
);
extern void solution_free(void);


#ifdef __cplusplus
}
#endif

typedef struct {
    int N_I, N_B, N_Phi, N_Psi, N_x, N_phi, T;
    float mu, nu, alpha, y;
    float *c_t, *h_t, *b_t;
    float *grid_I, *grid_B, *grid_Phi, *grid_Psi;
    float *actions, *shocks, *shock_probs;
    float *V_out;
} Ctx;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    Ctx* ctx = (Ctx*)calloc(1, sizeof(Ctx));
    ctx->N_I   = (int)get_param(data, "N_I");
    ctx->N_B   = (int)get_param(data, "N_B");
    ctx->N_Phi = (int)get_param(data, "N_Phi");
    ctx->N_Psi = (int)get_param(data, "N_Psi");
    ctx->N_x   = (int)get_param(data, "N_x");
    ctx->N_phi = (int)get_param(data, "N_phi");
    ctx->T     = (int)get_param(data, "T");

    float* fp = get_tensor_float(data, "fparams");
    ctx->mu    = fp[0];
    ctx->nu    = fp[1];
    ctx->alpha = fp[2];
    ctx->y     = fp[3];

    ctx->c_t         = get_tensor_float(data, "c_t");
    ctx->h_t         = get_tensor_float(data, "h_t");
    ctx->b_t         = get_tensor_float(data, "b_t");
    ctx->grid_I      = get_tensor_float(data, "grid_I");
    ctx->grid_B      = get_tensor_float(data, "grid_B");
    ctx->grid_Phi    = get_tensor_float(data, "grid_Phi");
    ctx->grid_Psi    = get_tensor_float(data, "grid_Psi");
    ctx->actions     = get_tensor_float(data, "actions");
    ctx->shocks      = get_tensor_float(data, "shocks");
    ctx->shock_probs = get_tensor_float(data, "shock_probs");

    int total = ctx->N_I * ctx->N_B * ctx->N_Phi * ctx->N_Psi;
    ctx->V_out = (float*)calloc((size_t)total, sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    Ctx* ctx = (Ctx*)test_data;
    solution_compute(
        ctx->N_I, ctx->N_B, ctx->N_Phi, ctx->N_Psi, ctx->N_x, ctx->N_phi, ctx->T,
        ctx->c_t, ctx->h_t, ctx->b_t,
        ctx->mu, ctx->nu, ctx->alpha, ctx->y,
        ctx->grid_I, ctx->grid_B, ctx->grid_Phi, ctx->grid_Psi,
        ctx->actions, ctx->shocks, ctx->shock_probs,
        ctx->V_out
    );
}

void task_write_output(void* test_data, const char* output_path) {
    Ctx* ctx = (Ctx*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    int total = ctx->N_I * ctx->N_B * ctx->N_Phi * ctx->N_Psi;
    for (int i = 0; i < total; i++)
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

#ifdef __cplusplus
}
#endif
