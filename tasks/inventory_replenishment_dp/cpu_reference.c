#include <stdlib.h>
#include <math.h>
#include <string.h>

static void get_interp_weights(float val, const float* grid, int N, int* idx, float* w) {
    if (val <= grid[0]) {
        *idx = 0;
        *w = 0.0f;
    } else if (val >= grid[N-1]) {
        *idx = N - 2;
        *w = 1.0f;
    } else {
        int low = 0, high = N - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (grid[mid] <= val) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        *idx = high;
        if (*idx < 0) *idx = 0;
        if (*idx >= N - 1) *idx = N - 2;
        *w = (val - grid[*idx]) / (grid[*idx+1] - grid[*idx]);
    }
}

void solution_compute(
    int N_I, int N_B, int N_Phi, int N_Psi, int N_x, int N_phi, int T,
    const float* c_t, const float* h_t, const float* b_t,
    float mu, float nu, float alpha, float y,
    const float* grid_I, const float* grid_B, const float* grid_Phi, const float* grid_Psi,
    const float* actions, const float* shocks, const float* shock_probs,
    float* V_out
) {
    int total_states = N_I * N_B * N_Phi * N_Psi;
    float* V_next = (float*)calloc(total_states, sizeof(float));
    float* V_curr = (float*)malloc(total_states * sizeof(float));

    for (int t = T - 1; t >= 0; t--) {
        for (int iI = 0; iI < N_I; iI++) {
            for (int iB = 0; iB < N_B; iB++) {
                for (int iPhi = 0; iPhi < N_Phi; iPhi++) {
                    for (int iPsi = 0; iPsi < N_Psi; iPsi++) {
                        float min_cost = 1e30f;

                        for (int ix = 0; ix < N_x; ix++) {
                            float x = actions[ix];
                            float expected_cost = 0.0f;

                            for (int iphi = 0; iphi < N_phi; iphi++) {
                                float phi = shocks[iphi];
                                float prob = shock_probs[iphi];

                                float xi = mu + nu * (phi + alpha * grid_Phi[iPhi]);
                                float I_next = grid_I[iI] + y + x - xi;
                                float B_next = grid_B[iB] + fmaxf(0.0f, -I_next);
                                float Phi_next = grid_Phi[iPhi] + phi;
                                float Psi_next = grid_Psi[iPsi] + phi * phi;

                                float cost = c_t[t] * fabsf(x) + h_t[t] * fmaxf(0.0f, I_next) + b_t[t] * fmaxf(0.0f, -I_next);

                                int idxI, idxB, idxPhi, idxPsi;
                                float wI, wB, wPhi, wPsi;
                                get_interp_weights(I_next, grid_I, N_I, &idxI, &wI);
                                get_interp_weights(B_next, grid_B, N_B, &idxB, &wB);
                                get_interp_weights(Phi_next, grid_Phi, N_Phi, &idxPhi, &wPhi);
                                get_interp_weights(Psi_next, grid_Psi, N_Psi, &idxPsi, &wPsi);

                                float future_cost = 0.0f;
                                for (int dI = 0; dI <= 1; dI++) {
                                    float cI = dI ? wI : (1.0f - wI);
                                    for (int dB = 0; dB <= 1; dB++) {
                                        float cB = dB ? wB : (1.0f - wB);
                                        for (int dPhi = 0; dPhi <= 1; dPhi++) {
                                            float cPhi = dPhi ? wPhi : (1.0f - wPhi);
                                            for (int dPsi = 0; dPsi <= 1; dPsi++) {
                                                float cPsi = dPsi ? wPsi : (1.0f - wPsi);
                                                
                                                int flat_idx = (((idxI + dI) * N_B + (idxB + dB)) * N_Phi + (idxPhi + dPhi)) * N_Psi + (idxPsi + dPsi);
                                                future_cost += cI * cB * cPhi * cPsi * V_next[flat_idx];
                                            }
                                        }
                                    }
                                }
                                expected_cost += prob * (cost + future_cost);
                            }
                            if (expected_cost < min_cost) {
                                min_cost = expected_cost;
                            }
                        }
                        int flat_idx = (((iI) * N_B + (iB)) * N_Phi + (iPhi)) * N_Psi + (iPsi);
                        V_curr[flat_idx] = min_cost;
                    }
                }
            }
        }
        memcpy(V_next, V_curr, total_states * sizeof(float));
    }
    memcpy(V_out, V_next, total_states * sizeof(float));
    free(V_next);
    free(V_curr);
}

void solution_free(void) {}
