// task_io.cu -- bfs_frontier GPU I/O adapter layer

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif
extern void solution_init(int V, int E, int source,
                          const int* row_offsets,
                          const int* col_indices);
extern void solution_compute(int* distances);
#ifdef __cplusplus
}
#endif

typedef struct {
    int V;
    int* distances;
} TaskIOContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int V = (int)get_param(data, "V");
    int E = (int)get_param(data, "E");
    int source = (int)get_param(data, "source");

    const int* row_offsets = get_tensor_int(data, "row_offsets");
    const int* col_indices = get_tensor_int(data, "col_indices");
    if (!row_offsets || !col_indices) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        return NULL;
    }

    solution_init(V, E, source, row_offsets, col_indices);

    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    if (!ctx) return NULL;
    ctx->V = V;
    ctx->distances = (int*)calloc((size_t)V, sizeof(int));
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->distances);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->V; i++) {
        fprintf(f, "%d\n", ctx->distances[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    free(ctx->distances);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
