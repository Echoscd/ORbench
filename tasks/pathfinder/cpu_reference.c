// cpu_reference.c -- Pathfinder CPU baseline for ORBench
//
// Adapted to ORBench from the Rodinia Pathfinder benchmark idea / implementation.
// Source inspiration: yuhc/gpu-rodinia, openmp/pathfinder/pathfinder.cpp
//
// Pure computation only. No file I/O here.

#include <stdlib.h>
#include <string.h>

static int g_rows = 0;
static int g_cols = 0;
static const int* g_wall = NULL;
static int* g_prev = NULL;
static int* g_curr = NULL;

void solution_init(int rows, int cols, const int* wall) {
    g_rows = rows;
    g_cols = cols;
    g_wall = wall;

    g_prev = (int*)malloc((size_t)cols * sizeof(int));
    g_curr = (int*)malloc((size_t)cols * sizeof(int));
}

void solution_compute(int* output) {
    if (g_rows <= 0 || g_cols <= 0 || !g_wall || !g_prev || !g_curr || !output) {
        return;
    }

    memcpy(g_prev, g_wall, (size_t)g_cols * sizeof(int));

    for (int r = 1; r < g_rows; ++r) {
        const int* row = g_wall + (size_t)r * (size_t)g_cols;
        for (int c = 0; c < g_cols; ++c) {
            int best = g_prev[c];
            if (c > 0 && g_prev[c - 1] < best) best = g_prev[c - 1];
            if (c + 1 < g_cols && g_prev[c + 1] < best) best = g_prev[c + 1];
            g_curr[c] = row[c] + best;
        }

        int* tmp = g_prev;
        g_prev = g_curr;
        g_curr = tmp;
    }

    memcpy(output, g_prev, (size_t)g_cols * sizeof(int));
}

void solution_free(void) {
    free(g_prev);
    free(g_curr);
    g_prev = NULL;
    g_curr = NULL;
    g_wall = NULL;
    g_rows = 0;
    g_cols = 0;
}
