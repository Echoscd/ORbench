// task_io.cu -- regex_match GPU I/O adapter layer
//
// Build: nvcc -O2 -arch=sm_89 -I framework/
//        framework/harness_gpu.cu tasks/regex_match/task_io.cu solution.cu -o solution_gpu

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// ===== LLM-implemented interface =====
#ifdef __cplusplus
extern "C" {
#endif

extern void solution_init(int num_states, int num_symbols, int start_state,
                          int num_strings, int total_chars,
                          const int* trans_offsets, const int* trans_targets,
                          const int* eps_offsets, const int* eps_targets,
                          const int* is_accept,
                          const int* str_offsets, const int* str_data);

extern void solution_compute(int num_strings, int* results);

#ifdef __cplusplus
}
#endif

// ===== task_io internal state =====
typedef struct {
    int num_strings;
    int* results;
} TaskIOContext;

// ===== harness calls these four functions =====
#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    int num_states   = (int)get_param(data, "num_states");
    int num_symbols  = (int)get_param(data, "num_symbols");
    int start_state  = (int)get_param(data, "start_state");
    int num_strings  = (int)get_param(data, "num_strings");
    int total_chars  = (int)get_param(data, "total_chars");

    const int* trans_offsets = get_tensor_int(data, "trans_offsets");
    const int* trans_targets = get_tensor_int(data, "trans_targets");
    const int* eps_offsets   = get_tensor_int(data, "eps_offsets");
    const int* eps_targets   = get_tensor_int(data, "eps_targets");
    const int* is_accept     = get_tensor_int(data, "is_accept");
    const int* str_offsets   = get_tensor_int(data, "str_offsets");
    const int* str_data      = get_tensor_int(data, "str_data");

    if (!trans_offsets || !trans_targets || !eps_offsets || !eps_targets ||
        !is_accept || !str_offsets || !str_data) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        return NULL;
    }

    solution_init(num_states, num_symbols, start_state,
                  num_strings, total_chars,
                  trans_offsets, trans_targets,
                  eps_offsets, eps_targets,
                  is_accept, str_offsets, str_data);

    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    ctx->num_strings = num_strings;
    ctx->results = (int*)calloc((size_t)num_strings, sizeof(int));

    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->num_strings, ctx->results);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->num_strings; i++) {
        fprintf(f, "%d\n", ctx->results[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    free(ctx->results);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
