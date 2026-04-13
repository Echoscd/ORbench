// cpu_reference.c -- BFS CPU baseline
// Computes unweighted single-source shortest hop distances on a CSR graph.

#include <stdlib.h>
#include <string.h>

static int g_V;
static int g_E;
static int g_source;
static const int* g_row_offsets;
static const int* g_col_indices;

void solution_init(int V, int E, int source,
                   const int* row_offsets,
                   const int* col_indices) {
    g_V = V;
    g_E = E;
    g_source = source;
    g_row_offsets = row_offsets;
    g_col_indices = col_indices;
}

void solution_compute(int* distances) {
    int* queue = (int*)malloc((size_t)g_V * sizeof(int));
    if (!queue) return;

    for (int i = 0; i < g_V; i++) distances[i] = -1;

    int head = 0, tail = 0;
    distances[g_source] = 0;
    queue[tail++] = g_source;

    while (head < tail) {
        int u = queue[head++];
        int du = distances[u];
        int start = g_row_offsets[u];
        int end = g_row_offsets[u + 1];
        for (int e = start; e < end; e++) {
            int v = g_col_indices[e];
            if (distances[v] == -1) {
                distances[v] = du + 1;
                queue[tail++] = v;
            }
        }
    }

    free(queue);
}
