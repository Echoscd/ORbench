// task_io_cpu.c — hybrid_sort CPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_init(int N, const int* h_keys);
extern void solution_compute(int* h_sorted_keys);
extern void solution_free(void);

typedef struct {
    int N;
    int* sorted_keys;
} TaskIOContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int N = (int)get_param(data, "N");
    const int* keys = get_tensor_int(data, "keys");
    if (!keys) {
        fprintf(stderr, "[task_io] Missing tensor: keys\n");
        return NULL;
    }
    solution_init(N, keys);
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    ctx->N = N;
    ctx->sorted_keys = (int*)malloc((size_t)N * sizeof(int));
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->sorted_keys);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++) fprintf(f, "%d\n", ctx->sorted_keys[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_free();
    free(ctx->sorted_keys);
    free(ctx);
}
