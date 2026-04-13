// task_io.cu -- Pathfinder GPU I/O adapter layer

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void solution_init(int rows, int cols, const int* wall);
extern void solution_compute(int* output);
extern void solution_free(void);

#ifdef __cplusplus
}
#endif

typedef struct {
    int cols;
    int* output;
} TaskIOContext;

#ifdef __cplusplus
extern "C" {
#endif

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;

    int rows = (int)get_param(data, "rows");
    int cols = (int)get_param(data, "cols");
    const int* wall = get_tensor_int(data, "wall");

    if (!wall) {
        fprintf(stderr, "[task_io] Missing tensor: wall\n");
        return NULL;
    }

    solution_init(rows, cols, wall);

    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    ctx->cols = cols;
    ctx->output = (int*)calloc((size_t)cols, sizeof(int));
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->output);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->cols; ++i) {
        fprintf(f, "%d\n", ctx->output[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_free();
    free(ctx->output);
    free(ctx);
}

#ifdef __cplusplus
}
#endif
