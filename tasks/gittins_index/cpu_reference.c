// cpu_reference.c — Gittins Index DP (compute_only interface)
//
// Computes the Gittins Index table for a Bernoulli bandit using
// layer-by-layer backward recursion with binary search calibration.
//
// Reference: Gittins, J.C. (1979) "Bandit Processes and Dynamic Allocation Indices"
//
// State (alpha, beta): alpha successes, beta failures, Beta posterior
// Index: s = k*(k+1)/2 + alpha  where k = alpha + beta
// Total states S = N*(N+1)/2

#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline int idx(int alpha, int beta) {
    int k = alpha + beta;
    return k * (k + 1) / 2 + alpha;
}

static inline float R_val(int alpha, int beta) {
    return (float)(alpha + 1) / (float)(alpha + beta + 2);
}

void solution_compute(
    int N, int a_x10000, int S, int num_bisect,
    float* V_out
) {
    float a = (float)a_x10000 / 10000.0f;

    float* nu = (float*)calloc((size_t)S, sizeof(float));

    // Boundary layer: k = N-1
    {
        int k = N - 1;
        for (int alpha = 0; alpha <= k; alpha++) {
            int beta = k - alpha;
            nu[idx(alpha, beta)] = R_val(alpha, beta);
        }
    }

    // Backward recursion: k = N-2 down to 0
    float* V = (float*)malloc((size_t)N * (size_t)N * sizeof(float));

    for (int k = N - 2; k >= 0; k--) {
        int M = N - k;

        for (int alpha = 0; alpha <= k; alpha++) {
            int beta = k - alpha;

            float lo = 0.0f, hi = 1.0f;

            for (int iter = 0; iter < num_bisect; iter++) {
                float lam = (lo + hi) * 0.5f;

                memset(V, 0, (size_t)M * (size_t)M * sizeof(float));

                // d = M-1 (farthest layer before truncation)
                for (int r = 0; r < M; r++) {
                    int s = M - 1 - r;
                    int a2 = alpha + r;
                    int b2 = beta + s;
                    if (nu[idx(a2, b2)] < lam) {
                        V[r * M + s] = 0.0f;
                    } else {
                        float rv = R_val(a2, b2);
                        float cont = rv - lam;
                        V[r * M + s] = cont > 0.0f ? cont : 0.0f;
                    }
                }

                // d = M-2 down to 1
                for (int d = M - 2; d >= 1; d--) {
                    for (int r = 0; r <= d; r++) {
                        int s = d - r;
                        int a2 = alpha + r;
                        int b2 = beta + s;

                        if (nu[idx(a2, b2)] < lam) {
                            V[r * M + s] = 0.0f;
                        } else {
                            float rv = R_val(a2, b2);
                            float p = rv;
                            float v_succ = V[(r + 1) * M + s];
                            float v_fail = V[r * M + (s + 1)];
                            float cont = rv - lam + a * (p * v_succ + (1.0f - p) * v_fail);
                            V[r * M + s] = cont > 0.0f ? cont : 0.0f;
                        }
                    }
                }

                // V at root (d=0)
                float rv = R_val(alpha, beta);
                float p = rv;
                float v_succ = V[1 * M + 0];
                float v_fail = V[0 * M + 1];
                float v_root = rv - lam + a * (p * v_succ + (1.0f - p) * v_fail);

                if (v_root > 1e-12f) {
                    lo = lam;
                } else {
                    hi = lam;
                }
            }

            nu[idx(alpha, beta)] = (lo + hi) * 0.5f;
        }
    }

    memcpy(V_out, nu, (size_t)S * sizeof(float));

    free(nu);
    free(V);
}

void solution_free(void) {
}
