#include <stdlib.h>

typedef struct {
    int num_rows;
    int num_cols;
    int nnz;
    int num_diags;
    const int* jad_ptr;
    const int* col_idx;
    const float* values;
    const int* perm;
    const int* row_nnz;
    const float* x;
} SpMVJDSCtx;

static SpMVJDSCtx g_ctx;

void solution_init(
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
) {
    g_ctx.num_rows = num_rows;
    g_ctx.num_cols = num_cols;
    g_ctx.nnz = nnz;
    g_ctx.num_diags = num_diags;
    g_ctx.jad_ptr = h_jad_ptr;
    g_ctx.col_idx = h_col_idx;
    g_ctx.values = h_values;
    g_ctx.perm = h_perm;
    g_ctx.row_nnz = h_row_nnz;
    g_ctx.x = h_x;
}

void solution_compute(float* h_y_out) {
    int rp, d;
    for (rp = 0; rp < g_ctx.num_rows; rp++) {
        float sum = 0.0f;
        int nnz_in_row = g_ctx.row_nnz[rp];
        for (d = 0; d < nnz_in_row; d++) {
            int idx = g_ctx.jad_ptr[d] + rp;
            sum += g_ctx.values[idx] * g_ctx.x[g_ctx.col_idx[idx]];
        }
        h_y_out[g_ctx.perm[rp]] = sum;
    }
}

void solution_free(void) {
    g_ctx.num_rows = 0;
    g_ctx.num_cols = 0;
    g_ctx.nnz = 0;
    g_ctx.num_diags = 0;
    g_ctx.jad_ptr = NULL;
    g_ctx.col_idx = NULL;
    g_ctx.values = NULL;
    g_ctx.perm = NULL;
    g_ctx.row_nnz = NULL;
    g_ctx.x = NULL;
}
