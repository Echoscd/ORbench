// task_io.cu — nash_flows_over_time GPU I/O adapter (init_compute)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_init(
    int num_nodes, int num_edges, int num_steps,
    const int* edge_u, const int* edge_v,
    const float* edge_capacity, const int* edge_transit_time
);
extern void solution_compute(float inflow_rate, float* out_total_arrived);
extern void solution_free(void);

#ifdef __cplusplus
}
#endif

typedef struct {
    int num_requests;
    float* inflow_rates;
    float* results;
} Ctx;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    int num_nodes = (int)get_param(data, "num_nodes");
    int num_edges = (int)get_param(data, "num_edges");
    int num_steps = (int)get_param(data, "num_steps");
    int* edge_u          = get_tensor_int(data, "edge_u");
    int* edge_v          = get_tensor_int(data, "edge_v");
    float* edge_capacity = get_tensor_float(data, "edge_capacity");
    int* edge_transit_time = get_tensor_int(data, "edge_transit_time");

    solution_init(num_nodes, num_edges, num_steps,
                  edge_u, edge_v, edge_capacity, edge_transit_time);

    Ctx* ctx = (Ctx*)calloc(1, sizeof(Ctx));
    char req_path[512];
    snprintf(req_path, sizeof(req_path), "%s/requests.txt", data_dir);
    FILE* f = fopen(req_path, "r");
    if (!f) {
        ctx->num_requests = 0;
        return ctx;
    }

    float rates[1024];
    int n = 0;
    while (n < 1024 && fscanf(f, "%f", &rates[n]) == 1) n++;
    fclose(f);

    ctx->num_requests = n;
    ctx->inflow_rates = (float*)malloc(n * sizeof(float));
    memcpy(ctx->inflow_rates, rates, n * sizeof(float));
    ctx->results = (float*)calloc(n, sizeof(float));

    return ctx;
}

void task_run(void* test_data) {
    Ctx* ctx = (Ctx*)test_data;
    for (int i = 0; i < ctx->num_requests; i++) {
        solution_compute(ctx->inflow_rates[i], &ctx->results[i]);
    }
}

void task_write_output(void* test_data, const char* output_path) {
    Ctx* ctx = (Ctx*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->num_requests; i++)
        fprintf(f, "%.6f\n", ctx->results[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    Ctx* ctx = (Ctx*)test_data;
    solution_free();
    free(ctx->inflow_rates);
    free(ctx->results);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
