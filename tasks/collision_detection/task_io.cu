// task_io.cu -- collision_detection GPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_compute(int N, int total_verts,
                             int world_size_x100, int cell_size_x100,
                             const int* poly_offsets,
                             const float* vertices_x, const float* vertices_y,
                             const float* aabb,
                             int* counts);

extern void solution_free(void);

#ifdef __cplusplus
}
#endif

typedef struct {
    int N;
    int total_verts;
    int world_size_x100;
    int cell_size_x100;
    const int* poly_offsets;
    const float* vertices_x;
    const float* vertices_y;
    const float* aabb;
    int* counts;
} TaskIOContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    int N              = (int)get_param(data, "N");
    int total_verts    = (int)get_param(data, "total_verts");
    int world_size_x100 = (int)get_param(data, "world_size_x100");
    int cell_size_x100  = (int)get_param(data, "cell_size_x100");

    const int*   poly_offsets = get_tensor_int(data, "poly_offsets");
    const float* vertices_x   = get_tensor_float(data, "vertices_x");
    const float* vertices_y   = get_tensor_float(data, "vertices_y");
    const float* aabb         = get_tensor_float(data, "aabb");

    if (!poly_offsets || !vertices_x || !vertices_y || !aabb) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        return NULL;
    }

    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    ctx->N = N;
    ctx->total_verts = total_verts;
    ctx->world_size_x100 = world_size_x100;
    ctx->cell_size_x100 = cell_size_x100;
    ctx->poly_offsets = poly_offsets;
    ctx->vertices_x = vertices_x;
    ctx->vertices_y = vertices_y;
    ctx->aabb = aabb;
    ctx->counts = (int*)calloc((size_t)N, sizeof(int));
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->N, ctx->total_verts, ctx->world_size_x100, ctx->cell_size_x100,
                     ctx->poly_offsets, ctx->vertices_x, ctx->vertices_y, ctx->aabb,
                     ctx->counts);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++)
        fprintf(f, "%d\n", ctx->counts[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_free();
    free(ctx->counts);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
