#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void solution_compute(
    int batch_size, int J, int N, int num_actions,
    float h, float tau, float lambda0, float a, float b_param,
    const float* alpha0, const float* beta,
    float* U, float* Lambda
) {
    for (int b = 0; b < batch_size; b++) {
        for (int j = 0; j < J; j++) {
            U[(b * J + j) * N + (N - 1)] = 0.0f;
            Lambda[(b * J + j) * N + (N - 1)] = 0.0f;
        }
    }

    for (int n = N - 2; n >= 0; n--) {
        for (int b = 0; b < batch_size; b++) {
            float a0 = alpha0[b];
            float bet = beta[b];
            
            for (int j = 0; j < J; j++) {
                float real_j = (float)(j + 1);
                float x = real_j * h;
                
                float bb1 = real_j * (1.0f - bet * tau);
                int bb2 = (int)floorf(bb1);
                
                float cc1 = bb1 + (a0 / h) * (1.0f - bet * tau);
                int cc2 = (int)floorf(cc1);
                
                float Vbb, Vcc;
                
                if (bb2 == 0) {
                    float u1 = U[(b * J + 1) * N + (n + 1)];
                    float u0 = U[(b * J + 0) * N + (n + 1)];
                    float deltabb = u1 - u0;
                    Vbb = u0 - (1.0f - bb1) * deltabb;
                } else {
                    float u_bb2 = U[(b * J + bb2) * N + (n + 1)];
                    float u_bb2_minus_1 = U[(b * J + bb2 - 1) * N + (n + 1)];
                    float deltabb = u_bb2 - u_bb2_minus_1;
                    Vbb = u_bb2_minus_1 + (bb1 - (float)bb2) * deltabb;
                }
                
                if (cc1 >= (float)J) {
                    float u_J_1 = U[(b * J + J - 1) * N + (n + 1)];
                    float u_J_2 = U[(b * J + J - 2) * N + (n + 1)];
                    float deltacc = u_J_1 - u_J_2;
                    Vcc = u_J_1 + (cc1 - (float)J) * deltacc;
                } else {
                    int safe_cc2 = cc2;
                    if (safe_cc2 < 1) safe_cc2 = 1;
                    
                    float u_cc2 = U[(b * J + safe_cc2) * N + (n + 1)];
                    float u_cc2_minus_1 = U[(b * J + safe_cc2 - 1) * N + (n + 1)];
                    float deltacc = u_cc2 - u_cc2_minus_1;
                    Vcc = u_cc2_minus_1 + (cc1 - (float)safe_cc2) * deltacc;
                }
                
                float max_A = -1e30f;
                float best_lambda = lambda0;
                
                for (int pp = 0; pp < num_actions; pp++) {
                    float lambda_val = lambda0 + pp * 0.001f;
                    float A_val = lambda_val * ((a - lambda_val) / b_param + Vcc - Vbb);
                    if (A_val > max_A) {
                        max_A = A_val;
                        best_lambda = lambda_val;
                    }
                }
                
                Lambda[(b * J + j) * N + n] = best_lambda;
                U[(b * J + j) * N + n] = Vbb + (x + 0.001f * x * x + 1.0f) * tau * max_A;
            }
        }
    }
}

void solution_free(void) {
    // No persistent state to free
}
