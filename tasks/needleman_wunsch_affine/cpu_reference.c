// cpu_reference.c -- Needleman-Wunsch global alignment with affine gaps
// compute_only interface

#include <stdlib.h>
#include <string.h>

#define NEG_INF (-1000000000)

static inline int max2(int a, int b) { return a > b ? a : b; }
static inline int max3(int a, int b, int c) {
    int m = a > b ? a : b;
    return m > c ? m : c;
}

void solution_compute(
    int N,
    int total_query_len,
    int total_target_len,
    int match_score,
    int mismatch_penalty,
    int gap_open_penalty,
    int gap_extend_penalty,
    const int* query_seqs,
    const int* target_seqs,
    const int* query_offsets,
    const int* target_offsets,
    int* scores
) {
    (void)total_query_len;
    (void)total_target_len;

    for (int p = 0; p < N; ++p) {
        int qs = query_offsets[p];
        int qe = query_offsets[p + 1];
        int ts = target_offsets[p];
        int te = target_offsets[p + 1];
        int m = qe - qs;
        int n = te - ts;

        int* M_prev = (int*)malloc((size_t)(n + 1) * sizeof(int));
        int* M_curr = (int*)malloc((size_t)(n + 1) * sizeof(int));
        int* X_prev = (int*)malloc((size_t)(n + 1) * sizeof(int));
        int* X_curr = (int*)malloc((size_t)(n + 1) * sizeof(int));
        int* Y_prev = (int*)malloc((size_t)(n + 1) * sizeof(int));
        int* Y_curr = (int*)malloc((size_t)(n + 1) * sizeof(int));

        M_prev[0] = 0;
        X_prev[0] = NEG_INF;
        Y_prev[0] = NEG_INF;
        for (int j = 1; j <= n; ++j) {
            M_prev[j] = NEG_INF;
            X_prev[j] = NEG_INF;
            Y_prev[j] = (j == 1) ? (gap_open_penalty + gap_extend_penalty)
                                 : (Y_prev[j - 1] + gap_extend_penalty);
        }

        for (int i = 1; i <= m; ++i) {
            M_curr[0] = NEG_INF;
            X_curr[0] = (i == 1) ? (gap_open_penalty + gap_extend_penalty)
                                 : (X_prev[0] + gap_extend_penalty);
            Y_curr[0] = NEG_INF;

            int qi = query_seqs[qs + i - 1];
            for (int j = 1; j <= n; ++j) {
                int tj = target_seqs[ts + j - 1];
                int sub = (qi == tj) ? match_score : mismatch_penalty;

                int prev_best = max3(M_prev[j - 1], X_prev[j - 1], Y_prev[j - 1]);
                M_curr[j] = (prev_best <= NEG_INF / 2) ? NEG_INF : (prev_best + sub);

                int x_from_m = (M_prev[j] <= NEG_INF / 2) ? NEG_INF : (M_prev[j] + gap_open_penalty + gap_extend_penalty);
                int x_from_x = (X_prev[j] <= NEG_INF / 2) ? NEG_INF : (X_prev[j] + gap_extend_penalty);
                X_curr[j] = max2(x_from_m, x_from_x);

                int y_from_m = (M_curr[j - 1] <= NEG_INF / 2) ? NEG_INF : (M_curr[j - 1] + gap_open_penalty + gap_extend_penalty);
                int y_from_y = (Y_curr[j - 1] <= NEG_INF / 2) ? NEG_INF : (Y_curr[j - 1] + gap_extend_penalty);
                Y_curr[j] = max2(y_from_m, y_from_y);
            }

            {
                int* tmp;
                tmp = M_prev; M_prev = M_curr; M_curr = tmp;
                tmp = X_prev; X_prev = X_curr; X_curr = tmp;
                tmp = Y_prev; Y_prev = Y_curr; Y_curr = tmp;
            }
        }

        scores[p] = max3(M_prev[n], X_prev[n], Y_prev[n]);

        free(M_prev); free(M_curr);
        free(X_prev); free(X_curr);
        free(Y_prev); free(Y_curr);
    }
}

void solution_free(void) {
}
