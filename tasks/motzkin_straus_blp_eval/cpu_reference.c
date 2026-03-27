#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void solution_compute(int N, int M, float mu, const float* A, const float* x, const float* q, const float* s, float* obj, float* max_viol) {
    for (int m = 0; m < M; ++m) {
        const float* current_x = x + m * N;
        const float* current_q = q + m * N;
        float current_s = s[m];
        
        float current_obj = current_s;
        float current_max_viol = 0.0f;
        
        // Constraint 5: s >= 0
        if (current_s < 0.0f) {
            current_max_viol = fmaxf(current_max_viol, -current_s);
        }
        
        float sum_x = 0.0f;
        float sum_min_xq = 0.0f;
        
        for (int i = 0; i < N; ++i) {
            float xi = current_x[i];
            float qi = current_q[i];
            
            // Objective part
            sum_min_xq += fminf(xi, qi);
            
            // Constraint 4: x_i >= 0
            if (xi < 0.0f) {
                current_max_viol = fmaxf(current_max_viol, -xi);
            }
            
            // Constraint 3: 0 <= q_i <= 1
            if (qi < 0.0f) {
                current_max_viol = fmaxf(current_max_viol, -qi);
            } else if (qi > 1.0f) {
                current_max_viol = fmaxf(current_max_viol, qi - 1.0f);
            }
            
            sum_x += xi;
            
            // Constraint 1: s - (Ax)_i - q_i = 0
            float ax_i = 0.0f;
            for (int j = 0; j < N; ++j) {
                ax_i += A[i * N + j] * current_x[j];
            }
            float viol1 = fabsf(current_s - ax_i - qi);
            current_max_viol = fmaxf(current_max_viol, viol1);
        }
        
        // Constraint 2: sum(x_i) - 1 = 0
        float viol2 = fabsf(sum_x - 1.0f);
        current_max_viol = fmaxf(current_max_viol, viol2);
        
        current_obj -= mu * sum_min_xq;
        
        obj[m] = current_obj;
        max_viol[m] = current_max_viol;
    }
}

void solution_free(void) {
    // No persistent state to free
}

#ifdef __cplusplus
}
#endif
