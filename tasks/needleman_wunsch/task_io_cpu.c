#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_init(
    int len_a,
    int len_b,
    int alphabet_size,
    int gap_penalty,
    const int* h_seq_a,
    const int* h_seq_b,
    const int* h_score_matrix
);
extern void solution_compute(int* h_out_last_row);
extern void solution_free(void);

typedef struct {
    int len_a, len_b, alphabet_size, gap_penalty;
    const int* seq_a;
    const int* seq_b;
    const int* score_matrix;
    int* out_last_row;
} Ctx;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    Ctx* ctx = (Ctx*)calloc(1, sizeof(Ctx));
    if (!ctx) return NULL;

    ctx->len_a = (int)get_param(data, "len_a");
    ctx->len_b = (int)get_param(data, "len_b");
    ctx->alphabet_size = (int)get_param(data, "alphabet_size");
    ctx->gap_penalty = (int)get_param(data, "gap_penalty");
    ctx->seq_a = get_tensor_int(data, "seq_a");
    ctx->seq_b = get_tensor_int(data, "seq_b");
    ctx->score_matrix = get_tensor_int(data, "score_matrix");

    if (!ctx->seq_a || !ctx->seq_b || !ctx->score_matrix) {
        free(ctx);
        return NULL;
    }

    ctx->out_last_row = (int*)calloc((size_t)(ctx->len_b + 1), sizeof(int));
    if (!ctx->out_last_row) {
        free(ctx);
        return NULL;
    }

    solution_init(ctx->len_a, ctx->len_b, ctx->alphabet_size, ctx->gap_penalty,
                  ctx->seq_a, ctx->seq_b, ctx->score_matrix);
    return ctx;
}

void task_run(void* test_data) {
    Ctx* ctx = (Ctx*)test_data;
    solution_compute(ctx->out_last_row);
}

void task_write_output(void* test_data, const char* output_path) {
    Ctx* ctx = (Ctx*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int j = 0; j <= ctx->len_b; ++j) {
        fprintf(f, "%d\n", ctx->out_last_row[j]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    Ctx* ctx = (Ctx*)test_data;
    solution_free();
    free(ctx->out_last_row);
    free(ctx);
}
