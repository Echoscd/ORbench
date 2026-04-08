// task_io_cpu.c -- smith_waterman CPU I/O adapter (compute_only interface)

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void solution_compute(
    int N,
    int total_query_len,
    int total_target_len,
    int match_score,
    int mismatch_penalty,
    int gap_penalty,
    const int* query_seqs,
    const int* target_seqs,
    const int* query_offsets,
    const int* target_offsets,
    int* scores
);

extern void solution_free(void);

typedef struct {
    int N;
    int total_query_len;
    int total_target_len;
    int match_score;
    int mismatch_penalty;
    int gap_penalty;
    const int* query_seqs;
    const int* target_seqs;
    const int* query_offsets;
    const int* target_offsets;
    int* scores;
} SmithWatermanContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;

    SmithWatermanContext* ctx = (SmithWatermanContext*)calloc(1, sizeof(SmithWatermanContext));
    if (!ctx) return NULL;

    ctx->N = (int)get_param(data, "N");
    ctx->total_query_len = (int)get_param(data, "total_query_len");
    ctx->total_target_len = (int)get_param(data, "total_target_len");
    ctx->match_score = (int)get_param(data, "match_score");
    ctx->mismatch_penalty = (int)get_param(data, "mismatch_penalty");
    ctx->gap_penalty = (int)get_param(data, "gap_penalty");

    ctx->query_seqs = get_tensor_int(data, "query_seqs");
    ctx->target_seqs = get_tensor_int(data, "target_seqs");
    ctx->query_offsets = get_tensor_int(data, "query_offsets");
    ctx->target_offsets = get_tensor_int(data, "target_offsets");

    if (!ctx->query_seqs || !ctx->target_seqs || !ctx->query_offsets || !ctx->target_offsets) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
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
    SmithWatermanContext* ctx = (SmithWatermanContext*)test_data;
    solution_compute(
        ctx->N,
        ctx->total_query_len,
        ctx->total_target_len,
        ctx->match_score,
        ctx->mismatch_penalty,
        ctx->gap_penalty,
        ctx->query_seqs,
        ctx->target_seqs,
        ctx->query_offsets,
        ctx->target_offsets,
        ctx->scores
    );
}

void task_write_output(void* test_data, const char* output_path) {
    SmithWatermanContext* ctx = (SmithWatermanContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->N; i++) {
        fprintf(f, "%d\n", ctx->scores[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    SmithWatermanContext* ctx = (SmithWatermanContext*)test_data;
    solution_free();
    free(ctx->scores);
    free(ctx);
}
