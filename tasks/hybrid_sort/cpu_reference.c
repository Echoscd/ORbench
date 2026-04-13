// cpu_reference.c — hybrid_sort CPU baseline

#include <stdlib.h>
#include <string.h>

static int g_N = 0;
static const int* g_keys = NULL;
static int* g_work = NULL;

static int cmp_int_asc(const void* a, const void* b) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    return (ia > ib) - (ia < ib);
}

void solution_init(int N, const int* h_keys) {
    g_N = N;
    g_keys = h_keys;
    g_work = (int*)malloc((size_t)N * sizeof(int));
}

void solution_compute(int* h_sorted_keys) {
    memcpy(g_work, g_keys, (size_t)g_N * sizeof(int));
    qsort(g_work, (size_t)g_N, sizeof(int), cmp_int_asc);
    memcpy(h_sorted_keys, g_work, (size_t)g_N * sizeof(int));
}

void solution_free(void) {
    free(g_work);
    g_work = NULL;
    g_keys = NULL;
    g_N = 0;
}
