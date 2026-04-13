#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void solution_init(
    int num_rows,
    int num_cols,
    int nnz,
    int num_diags,
    const int* h_jad_ptr,
    const int* h_col_idx,
    const float* h_values,
    const int* h_perm,
    const int* h_row_nnz,
    const float* h_x
);
extern void solution_compute(float* h_y_out);
extern void solution_free(void);

typedef struct {
    int num_rows;
    float* y_out;
} Ctx;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    Ctx* ctx = (Ctx*)calloc(1, sizeof(Ctx));
    ctx->num_rows = (int)get_param(data, "num_rows");
    int num_cols = (int)get_param(data, "num_cols");
    int nnz = (int)get_param(data, "nnz");
    int num_diags = (int)get_param(data, "num_diags");

    solution_init(
        ctx->num_rows,
        num_cols,
        nnz,
        num_diags,
        get_tensor_int(data, "jad_ptr"),
        get_tensor_int(data, "col_idx"),
        get_tensor_float(data, "values"),
        get_tensor_int(data, "perm"),
        get_tensor_int(data, "row_nnz"),
        get_tensor_float(data, "x")
    );

    ctx->y_out = (float*)calloc((size_t)ctx->num_rows, sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    Ctx* ctx = (Ctx*)test_data;
    solution_compute(ctx->y_out);
}

void task_write_output(void* test_data, const char* output_path) {
    Ctx* ctx = (Ctx*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->num_rows; i++) {
        fprintf(f, "%.6e\n", ctx->y_out[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    Ctx* ctx = (Ctx*)test_data;
    solution_free();
    free(ctx->y_out);
    free(ctx);
}
