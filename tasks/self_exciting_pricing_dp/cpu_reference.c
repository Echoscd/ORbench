#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void solution_compute(
    int N_x, int N_T, int N_lambda,
    float dx, float dt, float alpha, float beta, float phi,
    float c, float a, float gamma, float k,
    float lambda_min, float lambda_max,
    float* V_out
) {
    float* V_prev = (float*)calloc(N_x, sizeof(float));
    float* V_new = (float*)calloc(N_x, sizeof(float));
    
    float* lambdas = (float*)malloc(N_lambda * sizeof(float));
    float* r_lambdas = (float*)malloc(N_lambda * sizeof(float));
    
    float lambda_step = (lambda_max - lambda_min) / (N_lambda - 1);
    for (int j = 0; j < N_lambda; j++) {
        lambdas[j] = lambda_min + j * lambda_step;
        r_lambdas[j] = lambdas[j] * (logf(a / lambdas[j]) - c);
    }
    
    int alpha_idx_shift = (int)roundf(alpha / dx);
    
    for (int t = 0; t < N_T; t++) {
        for (int i = 0; i < N_x; i++) {
            float x = i * dx;
            float V_deriv = (i == 0) ? 0.0f : (V_prev[i] - V_prev[i-1]) / dx;
            
            int idx_alpha = i + alpha_idx_shift;
            if (idx_alpha > N_x - 1) idx_alpha = N_x - 1;
            
            float V_jump = V_prev[idx_alpha] - V_prev[i];
            
            float max_val = -1e30f;
            for (int j = 0; j < N_lambda; j++) {
                float val = r_lambdas[j] + lambdas[j] * V_jump;
                if (val > max_val) {
                    max_val = val;
                }
            }
            
            float h_x = k * powf(x, gamma);
            
            V_new[i] = V_prev[i] + dt * (-x * beta * V_deriv + (h_x + phi) * max_val);
        }
        
        // Swap
        float* temp = V_prev;
        V_prev = V_new;
        V_new = temp;
    }
    
    for (int i = 0; i < N_x; i++) {
        V_out[i] = V_prev[i];
    }
    
    free(V_prev);
    free(V_new);
    free(lambdas);
    free(r_lambdas);
}

void solution_free(void) {
    // No persistent state to free
}

#ifdef __cplusplus
}
#endif
