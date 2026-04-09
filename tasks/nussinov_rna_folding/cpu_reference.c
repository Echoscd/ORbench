#include <stdlib.h>
#include <stdint.h>

static inline int can_pair(int a, int b) {
    return (a == 0 && b == 3) || (a == 3 && b == 0) ||
           (a == 1 && b == 2) || (a == 2 && b == 1) ||
           (a == 2 && b == 3) || (a == 3 && b == 2);
}

void solution_compute(
    int N,
    int total_seq_len,
    int min_loop_len,
    const int* seqs,
    const int* offsets,
    int* scores
) {
    (void)total_seq_len;
    for (int p = 0; p < N; ++p) {
        int start = offsets[p];
        int end = offsets[p + 1];
        int n = end - start;
        if (n <= 0) {
            scores[p] = 0;
            continue;
        }
        int* dp = (int*)calloc((size_t)n * (size_t)n, sizeof(int));
        if (!dp) {
            scores[p] = -1;
            continue;
        }
        const int* s = seqs + start;

        for (int span = min_loop_len + 1; span < n; ++span) {
            for (int i = 0; i + span < n; ++i) {
                int j = i + span;
                int best = 0;

                int v1 = dp[(size_t)(i + 1) * n + j];
                if (v1 > best) best = v1;

                int v2 = dp[(size_t)i * n + (j - 1)];
                if (v2 > best) best = v2;

                if (can_pair(s[i], s[j])) {
                    int v3 = 1;
                    if (i + 1 <= j - 1) v3 += dp[(size_t)(i + 1) * n + (j - 1)];
                    if (v3 > best) best = v3;
                }

                for (int k = i + 1; k < j; ++k) {
                    int left = dp[(size_t)i * n + k];
                    int right = dp[(size_t)(k + 1) * n + j];
                    int cand = left + right;
                    if (cand > best) best = cand;
                }
                dp[(size_t)i * n + j] = best;
            }
        }
        scores[p] = dp[(size_t)0 * n + (n - 1)];
        free(dp);
    }
}

void solution_free(void) {}
