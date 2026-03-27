#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    float val;
    int idx;
} ValueIndex;

int compare_vi(const void* a, const void* b) {
    float va = ((ValueIndex*)a)->val;
    float vb = ((ValueIndex*)b)->val;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

void solution_compute(int S, int A, int T, float gamma, const float* rew, const float* P_up, const float* P_down, float* V) {
    for (int s = 0; s < S; ++s) {
        V[s] = 0.0f;
    }
    
    float* V_next = (float*)malloc(S * sizeof(float));
    ValueIndex* vi = (ValueIndex*)malloc(S * sizeof(ValueIndex));
    
    float* total_P_down = (float*)malloc(S * A * sizeof(float));
    for (int s = 0; s < S; ++s) {
        for (int a = 0; a < A; ++a) {
            float sum = 0.0f;
            for (int j = 0; j < S; ++j) {
                sum += P_down[(s * A + a) * S + j];
            }
            total_P_down[s * A + a] = sum;
        }
    }
    
    for (int t = 0; t < T; ++t) {
        for (int s = 0; s < S; ++s) {
            vi[s].val = V[s];
            vi[s].idx = s;
        }
        qsort(vi, S, sizeof(ValueIndex), compare_vi);
        
        for (int s = 0; s < S; ++s) {
            float max_Q = -1e30f;
            for (int a = 0; a < A; ++a) {
                int sa_offset = (s * A + a) * S;
                float run_sum = total_P_down[s * A + a];
                
                int index = 0;
                int p_idx = vi[0].idx;
                run_sum += P_up[sa_offset + p_idx] - P_down[sa_offset + p_idx];
                
                while (run_sum < 1.0f && index < S - 1) {
                    index++;
                    p_idx = vi[index].idx;
                    run_sum += P_up[sa_offset + p_idx] - P_down[sa_offset + p_idx];
                }
                
                float expected_V = 0.0f;
                for (int j = 0; j < index; ++j) {
                    int p_j = vi[j].idx;
                    expected_V += P_up[sa_offset + p_j] * V[p_j];
                }
                
                int p_index = vi[index].idx;
                expected_V += (1.0f - (run_sum - P_up[sa_offset + p_index])) * V[p_index];
                
                for (int j = index + 1; j < S; ++j) {
                    int p_j = vi[j].idx;
                    expected_V += P_down[sa_offset + p_j] * V[p_j];
                }
                
                float Q_sa = rew[s * A + a] + gamma * expected_V;
                if (Q_sa > max_Q) {
                    max_Q = Q_sa;
                }
            }
            V_next[s] = max_Q;
        }
        
        for (int s = 0; s < S; ++s) {
            V[s] = V_next[s];
        }
    }
    
    free(V_next);
    free(vi);
    free(total_P_down);
}

void solution_free(void) {
    // No persistent state to free
}
