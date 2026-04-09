#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_compute(
    int N,
    int total_seq_len,
    int min_loop_len,
    const int* seqs,
    const int* offsets,
    int* scores
);
extern void solution_free(void);

typedef struct {
    int N;
    int total_seq_len;
    int min_loop_len;
    const int* seqs;
    const int* offsets;
    int* scores;
} TaskIOContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    if (!ctx) return NULL;
    ctx->N = (int)get_param(data, "N");
    ctx->total_seq_len = (int)get_param(data, "total_seq_len");
    ctx->min_loop_len = (int)get_param(data, "min_loop_len");
    ctx->seqs = get_tensor_int(data, "seqs");
    ctx->offsets = get_tensor_int(data, "offsets");
    if (!ctx->seqs || !ctx->offsets) {
        free(ctx);
        return NULL;
    }
    ctx->scores = (int*)calloc((size_t)ctx->N, sizeof(int));
    if (!ctx->scores) {
        free(ctx);
        return NULL;
    }
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->N, ctx->total_seq_len, ctx->min_loop_len,
                     ctx->seqs, ctx->offsets, ctx->scores);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; ++i) fprintf(f, "%d\n", ctx->scores[i]);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_free();
    free(ctx->scores);
    free(ctx);
}
