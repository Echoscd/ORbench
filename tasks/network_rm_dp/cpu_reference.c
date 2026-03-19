// cpu_reference.c — network_rm_dp CPU baseline (compute_only interface)
//
// No solution_init. All work happens in solution_compute.
// NO file I/O. All I/O handled by task_io_cpu.c.
//
// Build (via task_io + harness):
//   gcc -O2 -DORBENCH_COMPUTE_ONLY -I framework/
//       framework/harness_cpu.c tasks/network_rm_dp/task_io_cpu.c
//       tasks/network_rm_dp/cpu_reference.c -o solution_cpu -lm

#include <stdlib.h>
#include <string.h>

#define MAX_RESOURCES 8

// ===== State encoding/decoding =====

static int encode_state(const int* c, const int* cap, int m) {
    int s = 0, stride = 1;
    for (int i = 0; i < m; i++) {
        s += c[i] * stride;
        stride *= (cap[i] + 1);
    }
    return s;
}

static void decode_state(int s, int* c, const int* cap, int m) {
    for (int i = 0; i < m; i++) {
        c[i] = s % (cap[i] + 1);
        s /= (cap[i] + 1);
    }
}

void solution_compute(
    int m, int n, int T, int L, int S,
    const int* capacity,
    const int* consumption,
    const float* demand_prob,
    const int* demand_cons,
    const float* demand_revenue,
    float* V_out
) {
    float* V_next = (float*)calloc((size_t)S, sizeof(float));
    float* V_curr = (float*)malloc((size_t)S * sizeof(float));

    // Backward induction: t = T, T-1, ..., 1
    for (int t = T; t >= 1; t--) {
        for (int s = 0; s < S; s++) {
            int c[MAX_RESOURCES];
            decode_state(s, c, capacity, m);

            float best_value = -1e30f;

            for (int l = 0; l < L; l++) {
                float expected_value = 0.0f;

                for (int k = 0; k <= n; k++) {
                    float prob = demand_prob[l * (n + 1) + k];
                    if (prob < 1e-12f) continue;

                    float rev_k = demand_revenue[l * (n + 1) + k];
                    const int* cons = &demand_cons[(l * (n + 1) + k) * m];

                    // Check feasibility and compute new state
                    int feasible = 1;
                    int new_c[MAX_RESOURCES];
                    for (int i = 0; i < m; i++) {
                        new_c[i] = c[i] - cons[i];
                        if (new_c[i] < 0) { feasible = 0; break; }
                    }

                    float future;
                    float actual_rev;
                    if (!feasible) {
                        // Infeasible: customer can't buy → no revenue, state unchanged
                        future = V_next[s];
                        actual_rev = 0.0f;
                    } else {
                        int new_s = encode_state(new_c, capacity, m);
                        future = V_next[new_s];
                        actual_rev = rev_k;
                    }

                    expected_value += prob * (actual_rev + future);
                }

                if (expected_value > best_value) {
                    best_value = expected_value;
                }
            }

            V_curr[s] = best_value;
        }

        // Swap buffers
        float* tmp = V_next;
        V_next = V_curr;
        V_curr = tmp;
    }

    // After loop, V_next holds V[1][·]
    memcpy(V_out, V_next, (size_t)S * sizeof(float));
    free(V_next);
    free(V_curr);
}

void solution_free(void) {
    // CPU version has no persistent state to free
}
