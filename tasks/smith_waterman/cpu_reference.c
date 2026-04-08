// cpu_reference.c -- smith_waterman CPU baseline (compute_only interface)
//
// No solution_init. All work happens in solution_compute.
// NO file I/O. All I/O handled by task_io_cpu.c.

#include <stdlib.h>
#include <string.h>

static int max2(int a, int b) { return (a > b) ? a : b; }
static int max4(int a, int b, int c, int d) { return max2(max2(a, b), max2(c, d)); }

void solution_compute(
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
) {
    (void)total_query_len;
    (void)total_target_len;

    for (int p = 0; p < N; p++) {
        int q_start = query_offsets[p];
        int q_end   = query_offsets[p + 1];
        int t_start = target_offsets[p];
        int t_end   = target_offsets[p + 1];
        int m = q_end - q_start;
        int n = t_end - t_start;

        int* prev = (int*)calloc((size_t)(n + 1), sizeof(int));
        int* curr = (int*)calloc((size_t)(n + 1), sizeof(int));
        int best = 0;

        if (!prev || !curr) {
            free(prev);
            free(curr);
            return;
        }

        for (int i = 1; i <= m; i++) {
            int qi = query_seqs[q_start + i - 1];
            for (int j = 1; j <= n; j++) {
                int tj = target_seqs[t_start + j - 1];
                int subst = (qi == tj) ? match_score : mismatch_penalty;

                int val = max4(
                    0,
                    prev[j - 1] + subst,
                    prev[j] + gap_penalty,
                    curr[j - 1] + gap_penalty
                );
                curr[j] = val;
                if (val > best) best = val;
            }

            {
                int* tmp = prev;
                prev = curr;
                curr = tmp;
            }
            memset(curr, 0, (size_t)(n + 1) * sizeof(int));
        }

        scores[p] = best;
        free(prev);
        free(curr);
    }
}

void solution_free(void) {
    // CPU baseline has no persistent state.
}
