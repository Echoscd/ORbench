#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int V;
    int E;
    int max_iters;
    float damping;
    float epsilon;
    const int* in_row_offsets;
    const int* in_col_indices;
    const int* out_degree;
    float* scores;
    float* next_scores;
    float* outgoing_contrib;
} PRState;

static PRState g;

void solution_init(int V, int E, int max_iters, float damping, float epsilon,
                   const int* in_row_offsets,
                   const int* in_col_indices,
                   const int* out_degree) {
    g.V = V;
    g.E = E;
    g.max_iters = max_iters;
    g.damping = damping;
    g.epsilon = epsilon;
    g.in_row_offsets = in_row_offsets;
    g.in_col_indices = in_col_indices;
    g.out_degree = out_degree;

    g.scores = (float*)malloc((size_t)V * sizeof(float));
    g.next_scores = (float*)malloc((size_t)V * sizeof(float));
    g.outgoing_contrib = (float*)malloc((size_t)V * sizeof(float));
}

void solution_compute(float* scores_out) {
    const int V = g.V;
    const float init_score = 1.0f / (float)V;
    const float base_score = (1.0f - g.damping) / (float)V;

    for (int i = 0; i < V; i++) {
        g.scores[i] = init_score;
        g.next_scores[i] = init_score;
    }

    for (int iter = 0; iter < g.max_iters; iter++) {
        for (int v = 0; v < V; v++) {
            int od = g.out_degree[v];
            g.outgoing_contrib[v] = g.scores[v] / (float)od;
        }

        double error = 0.0;
        for (int u = 0; u < V; u++) {
            float incoming_total = 0.0f;
            for (int ei = g.in_row_offsets[u]; ei < g.in_row_offsets[u + 1]; ei++) {
                int v = g.in_col_indices[ei];
                incoming_total += g.outgoing_contrib[v];
            }
            float ns = base_score + g.damping * incoming_total;
            error += fabs((double)ns - (double)g.scores[u]);
            g.next_scores[u] = ns;
        }

        float* tmp = g.scores;
        g.scores = g.next_scores;
        g.next_scores = tmp;

        if (error < (double)g.epsilon) {
            break;
        }
    }

    memcpy(scores_out, g.scores, (size_t)V * sizeof(float));
}

void solution_free(void) {
    free(g.scores);
    free(g.next_scores);
    free(g.outgoing_contrib);
    memset(&g, 0, sizeof(g));
}
