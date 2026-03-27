#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void solution_compute(int S, int T, int N, int K, const float* R, float* w) {
    float* R_norm = (float*)malloc(T * N * sizeof(float));
    float* M = (float*)malloc(T * T * sizeof(float));
    float* Phi = (float*)malloc(T * K * sizeof(float));
    float* Z = (float*)malloc(K * N * sizeof(float));
    float* F = (float*)malloc(T * N * sizeof(float));
    float* E = (float*)malloc(T * N * sizeof(float));
    float* C_F = (float*)malloc(N * N * sizeof(float));
    float* V_E = (float*)malloc(N * sizeof(float));
    float* Sigma_hat = (float*)malloc(N * N * sizeof(float));
    float* b = (float*)malloc(N * sizeof(float));
    float* x = (float*)malloc(N * sizeof(float));
    float* V_new = (float*)malloc(T * K * sizeof(float));
    
    for (int s = 0; s < S; s++) {
        const float* R_s = R + s * T * N;
        float* w_s = w + s * N;
        
        // 1. R_norm
        for (int j = 0; j < N; j++) {
            float norm = 0.0f;
            for (int i = 0; i < T; i++) {
                norm += R_s[i * N + j] * R_s[i * N + j];
            }
            norm = sqrtf(norm);
            if (norm == 0.0f) norm = 1.0f;
            for (int i = 0; i < T; i++) {
                R_norm[i * N + j] = R_s[i * N + j] / norm;
            }
        }
        
        // 2. M = R_norm * R_norm^T
        for (int i = 0; i < T; i++) {
            for (int j = i; j < T; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += R_norm[i * N + k] * R_norm[j * N + k];
                }
                M[i * T + j] = sum;
                M[j * T + i] = sum;
            }
        }
        
        // 3. Phi (Simultaneous Iteration)
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < K; j++) {
                Phi[i * K + j] = R_norm[i * N + j];
            }
        }
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < j; k++) {
                float dot = 0.0f;
                for (int i = 0; i < T; i++) {
                    dot += Phi[i * K + k] * Phi[i * K + j];
                }
                for (int i = 0; i < T; i++) {
                    Phi[i * K + j] -= dot * Phi[i * K + k];
                }
            }
            float norm = 0.0f;
            for (int i = 0; i < T; i++) {
                norm += Phi[i * K + j] * Phi[i * K + j];
            }
            norm = sqrtf(norm);
            if (norm > 1e-7f) {
                for (int i = 0; i < T; i++) {
                    Phi[i * K + j] /= norm;
                }
            }
        }
        
        for (int iter = 0; iter < 100; iter++) {
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < K; j++) {
                    float sum = 0.0f;
                    for (int l = 0; l < T; l++) {
                        sum += M[i * T + l] * Phi[l * K + j];
                    }
                    V_new[i * K + j] = sum;
                }
            }
            for (int j = 0; j < K; j++) {
                for (int k = 0; k < j; k++) {
                    float dot = 0.0f;
                    for (int i = 0; i < T; i++) {
                        dot += V_new[i * K + k] * V_new[i * K + j];
                    }
                    for (int i = 0; i < T; i++) {
                        V_new[i * K + j] -= dot * V_new[i * K + k];
                    }
                }
                float norm = 0.0f;
                for (int i = 0; i < T; i++) {
                    norm += V_new[i * K + j] * V_new[i * K + j];
                }
                norm = sqrtf(norm);
                if (norm > 1e-7f) {
                    for (int i = 0; i < T; i++) {
                        V_new[i * K + j] /= norm;
                    }
                }
            }
            for (int i = 0; i < T * K; i++) {
                Phi[i] = V_new[i];
            }
        }
        
        // 4. Z = Phi^T * R_s
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < T; k++) {
                    sum += Phi[k * K + i] * R_s[k * N + j];
                }
                Z[i * N + j] = sum;
            }
        }
        
        // 5. F = Phi * Z
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += Phi[i * K + k] * Z[k * N + j];
                }
                F[i * N + j] = sum;
            }
        }
        
        // 6. E = R_s - F
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < N; j++) {
                E[i * N + j] = R_s[i * N + j] - F[i * N + j];
            }
        }
        
        // 7. C_F = (1/T) * F^T * F
        for (int i = 0; i < N; i++) {
            for (int j = i; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < T; k++) {
                    sum += F[k * N + i] * F[k * N + j];
                }
                C_F[i * N + j] = sum / T;
                C_F[j * N + i] = sum / T;
            }
        }
        
        // 8. V_E
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int i = 0; i < T; i++) {
                sum += E[i * N + j] * E[i * N + j];
            }
            V_E[j] = sum / T;
        }
        
        // 9. Sigma_hat = C_F + diag(V_E)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                Sigma_hat[i * N + j] = C_F[i * N + j];
                if (i == j) {
                    Sigma_hat[i * N + j] += V_E[i];
                }
            }
        }
        
        // 10. Solve Sigma_hat * x = 1
        for (int i = 0; i < N; i++) b[i] = 1.0f;
        
        int chol_ok = 1;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= i; j++) {
                float sum = Sigma_hat[i * N + j];
                for (int k = 0; k < j; k++) {
                    sum -= Sigma_hat[i * N + k] * Sigma_hat[j * N + k];
                }
                if (i == j) {
                    if (sum <= 1e-7f) { chol_ok = 0; break; }
                    Sigma_hat[i * N + i] = sqrtf(sum);
                } else {
                    Sigma_hat[i * N + j] = sum / Sigma_hat[j * N + j];
                }
            }
            if (!chol_ok) break;
        }
        
        if (chol_ok) {
            for (int i = 0; i < N; i++) {
                float sum = b[i];
                for (int k = 0; k < i; k++) {
                    sum -= Sigma_hat[i * N + k] * x[k];
                }
                x[i] = sum / Sigma_hat[i * N + i];
            }
            for (int i = N - 1; i >= 0; i--) {
                float sum = x[i];
                for (int k = i + 1; k < N; k++) {
                    sum -= Sigma_hat[k * N + i] * x[k];
                }
                x[i] = sum / Sigma_hat[i * N + i];
            }
        } else {
            for (int i = 0; i < N; i++) x[i] = 1.0f / N;
        }
        
        // 11. w = x / sum(x)
        float sum_x = 0.0f;
        for (int i = 0; i < N; i++) sum_x += x[i];
        for (int i = 0; i < N; i++) w_s[i] = x[i] / sum_x;
    }
    
    free(R_norm);
    free(M);
    free(Phi);
    free(Z);
    free(F);
    free(E);
    free(C_F);
    free(V_E);
    free(Sigma_hat);
    free(b);
    free(x);
    free(V_new);
}

void solution_free(void) {
    // No persistent state to free
}
