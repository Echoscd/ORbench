#include <stdlib.h>
#include <stddef.h>

static int g_n = 0;
static double* g_matrix = 0;
static double* g_work = 0;

void solution_init(int n, const double* h_matrix) {
    g_n = n;
    g_matrix = (double*)h_matrix;
    (void)g_matrix;
}

void solution_compute(double* h_out_w) {
    int n = g_n;
    const double* src = g_matrix;
    static double* work = 0;
    static int work_cap = 0;
    static double* u1 = 0;
    static int u1_cap = 0;

    if (work_cap < n * n) {
        if (work) free(work);
        work = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
        work_cap = n * n;
    }
    if (u1_cap < n) {
        if (u1) free(u1);
        u1 = (double*)malloc((size_t)n * sizeof(double));
        u1_cap = n;
    }

    for (int i = 0; i < n * n; ++i) work[i] = src[i];

    for (int k = 0; k < n; ++k) {
        for (int j = k; j < n; ++j) {
            double s = 0.0;
            for (int p = 0; p < k; ++p) s += work[k * n + p] * work[p * n + j];
            work[k * n + j] = work[k * n + j] - s;
        }
        double pivot = work[k * n + k];
        for (int i = k + 1; i < n; ++i) {
            double s = 0.0;
            for (int p = 0; p < k; ++p) s += work[i * n + p] * work[p * n + k];
            work[i * n + k] = (work[i * n + k] - s) / pivot;
        }
    }

    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = i; j < n; ++j) s += work[i * n + j];
        u1[i] = s;
    }
    for (int i = 0; i < n; ++i) {
        double s = u1[i];
        for (int j = 0; j < i; ++j) s += work[i * n + j] * u1[j];
        h_out_w[i] = s;
    }
}

void solution_free(void) {}
