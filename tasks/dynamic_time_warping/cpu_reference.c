// cpu_reference.c -- dynamic_time_warping CPU baseline
//
// Computes exact Dynamic Time Warping (DTW) distance for a batch of integer-valued
// time-series pairs using a rolling-row DP.

#include <stdlib.h>
#include <string.h>

#define INF 0x3f3f3f3f

static inline int iabs_int(int x) {
    return (x < 0) ? -x : x;
}

static inline int min3_int(int a, int b, int c) {
    int m = (a < b) ? a : b;
    return (m < c) ? m : c;
}

void solution_compute(
    int N,
    int total_query_len,
    int total_target_len,
    const int* query_series,
    const int* target_series,
    const int* query_offsets,
    const int* target_offsets,
    int* distances
) {
    (void)total_query_len;
    (void)total_target_len;

    for (int p = 0; p < N; p++) {
        int qs = query_offsets[p];
        int qe = query_offsets[p + 1];
        int ts = target_offsets[p];
        int te = target_offsets[p + 1];
        int m = qe - qs;
        int n = te - ts;

        int* prev = (int*)malloc((size_t)(n + 1) * sizeof(int));
        int* curr = (int*)malloc((size_t)(n + 1) * sizeof(int));
        if (!prev || !curr) {
            free(prev);
            free(curr);
            return;
        }

        prev[0] = 0;
        for (int j = 1; j <= n; j++) prev[j] = INF;

        for (int i = 1; i <= m; i++) {
            curr[0] = INF;
            int qi = query_series[qs + i - 1];
            for (int j = 1; j <= n; j++) {
                int tj = target_series[ts + j - 1];
                int cost = iabs_int(qi - tj);
                int best_prev = min3_int(prev[j], curr[j - 1], prev[j - 1]);
                curr[j] = cost + best_prev;
            }
            {
                int* tmp = prev;
                prev = curr;
                curr = tmp;
            }
        }

        distances[p] = prev[n];
        free(prev);
        free(curr);
    }
}

void solution_free(void) {
    // No persistent state.
}
