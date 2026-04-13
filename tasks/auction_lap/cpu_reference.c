#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>

static int g_n = 0;
static int* g_profit = NULL;

static double* g_price = NULL;
static int* g_owner = NULL;        // column -> row
static int* g_assignment = NULL;   // row -> column
static double* g_best_bid = NULL;  // per column
static int* g_best_bidder = NULL;  // per column

#ifdef __cplusplus
extern "C" {
#endif

void solution_init(int n, const int* h_profit) {
    g_n = n;

    size_t mat_elems = (size_t)n * (size_t)n;
    g_profit = (int*)malloc(mat_elems * sizeof(int));
    memcpy(g_profit, h_profit, mat_elems * sizeof(int));

    g_price = (double*)malloc((size_t)n * sizeof(double));
    g_owner = (int*)malloc((size_t)n * sizeof(int));
    g_assignment = (int*)malloc((size_t)n * sizeof(int));
    g_best_bid = (double*)malloc((size_t)n * sizeof(double));
    g_best_bidder = (int*)malloc((size_t)n * sizeof(int));
}

void solution_compute(long long* out_total_profit) {
    const int n = g_n;
    const double eps = 1.0 / ((double)n + 1.0);

    for (int j = 0; j < n; ++j) {
        g_price[j] = 0.0;
        g_owner[j] = -1;
    }
    for (int i = 0; i < n; ++i) {
        g_assignment[i] = -1;
    }

    int unassigned = n;

    while (unassigned > 0) {
        for (int j = 0; j < n; ++j) {
            g_best_bid[j] = -1.0;
            g_best_bidder[j] = -1;
        }

        // Bidding phase: each unassigned row places one bid.
        for (int i = 0; i < n; ++i) {
            if (g_assignment[i] != -1) continue;

            double best_val = -DBL_MAX;
            double second_val = -DBL_MAX;
            int best_col = -1;
            const int* row = g_profit + (size_t)i * (size_t)n;

            for (int j = 0; j < n; ++j) {
                double val = (double)row[j] - g_price[j];
                if (val > best_val) {
                    second_val = best_val;
                    best_val = val;
                    best_col = j;
                } else if (val > second_val) {
                    second_val = val;
                }
            }

            double bid = best_val - second_val + eps;
            if (bid > g_best_bid[best_col] ||
                (bid == g_best_bid[best_col] && i < g_best_bidder[best_col])) {
                g_best_bid[best_col] = bid;
                g_best_bidder[best_col] = i;
            }
        }

        // Assignment phase: each column accepts its highest bid.
        for (int j = 0; j < n; ++j) {
            int bidder = g_best_bidder[j];
            if (bidder == -1) continue;

            int prev_owner = g_owner[j];
            if (prev_owner != -1) {
                g_assignment[prev_owner] = -1;
                unassigned += 1;
            }

            g_price[j] += g_best_bid[j];
            g_owner[j] = bidder;
            if (g_assignment[bidder] == -1) {
                unassigned -= 1;
            }
            g_assignment[bidder] = j;
        }
    }

    long long total = 0;
    for (int i = 0; i < n; ++i) {
        total += (long long)g_profit[(size_t)i * (size_t)n + (size_t)g_assignment[i]];
    }
    *out_total_profit = total;
}

void solution_free(void) {
    free(g_profit); g_profit = NULL;
    free(g_price); g_price = NULL;
    free(g_owner); g_owner = NULL;
    free(g_assignment); g_assignment = NULL;
    free(g_best_bid); g_best_bid = NULL;
    free(g_best_bidder); g_best_bidder = NULL;
    g_n = 0;
}

#ifdef __cplusplus
}
#endif
