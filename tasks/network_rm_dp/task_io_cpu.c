// task_io_cpu.c — network_rm_dp CPU I/O adapter (compute_only interface)
//
// In compute_only mode, task_setup only parses input to HOST memory.
// task_run calls solution_compute with all parameters (no solution_init).
//
// Build: gcc -O2 -DORBENCH_COMPUTE_ONLY -I framework/
//        framework/harness_cpu.c tasks/network_rm_dp/task_io_cpu.c
//        tasks/network_rm_dp/cpu_reference.c -o solution_cpu -lm

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ===== compute_only interface =====
extern void solution_compute(
    int m, int n, int T, int L, int S,
    const int* capacity,
    const int* consumption,
    const float* demand_prob,
    const int* demand_cons,
    const float* demand_revenue,
    float* V_out
);

extern void solution_free(void);

// ===== task_io internal state =====
typedef struct {
    int m, n, T, L, S;
    const int*   capacity;
    const int*   consumption;
    const float* demand_prob;
    const int*   demand_cons;
    const float* demand_revenue;
    float* V_out;
} RmDpContext;

// ===== harness calls these four functions =====
void* task_setup(const TaskData* data, const char* data_dir) {
    RmDpContext* ctx = (RmDpContext*)calloc(1, sizeof(RmDpContext));

    ctx->m = (int)get_param(data, "m");
    ctx->n = (int)get_param(data, "n");
    ctx->T = (int)get_param(data, "T");
    ctx->L = (int)get_param(data, "L");
    ctx->S = (int)get_param(data, "S");

    ctx->capacity       = get_tensor_int(data, "capacity");
    ctx->consumption    = get_tensor_int(data, "consumption");
    ctx->demand_prob    = get_tensor_float(data, "demand_prob");
    ctx->demand_cons    = get_tensor_int(data, "demand_cons");
    ctx->demand_revenue = get_tensor_float(data, "demand_revenue");

    if (!ctx->capacity || !ctx->consumption || !ctx->demand_prob ||
        !ctx->demand_cons || !ctx->demand_revenue) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }

    ctx->V_out = (float*)calloc((size_t)ctx->S, sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    RmDpContext* ctx = (RmDpContext*)test_data;
    solution_compute(
        ctx->m, ctx->n, ctx->T, ctx->L, ctx->S,
        ctx->capacity, ctx->consumption,
        ctx->demand_prob, ctx->demand_cons, ctx->demand_revenue,
        ctx->V_out
    );
}

void task_write_output(void* test_data, const char* output_path) {
    RmDpContext* ctx = (RmDpContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->S; i++)
        fprintf(f, "%.6e\n", ctx->V_out[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    RmDpContext* ctx = (RmDpContext*)test_data;
    solution_free();
    free(ctx->V_out);
    free(ctx);
}
