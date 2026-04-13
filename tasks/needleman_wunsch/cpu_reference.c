#include <stdlib.h>
#include <string.h>

static int g_len_a = 0;
static int g_len_b = 0;
static int g_alphabet = 0;
static int g_gap = 0;
static int* g_seq_a = NULL;
static int* g_seq_b = NULL;
static int* g_score = NULL;

void solution_init(
    int len_a,
    int len_b,
    int alphabet_size,
    int gap_penalty,
    const int* h_seq_a,
    const int* h_seq_b,
    const int* h_score_matrix
) {
    g_len_a = len_a;
    g_len_b = len_b;
    g_alphabet = alphabet_size;
    g_gap = gap_penalty;

    g_seq_a = (int*)malloc((size_t)len_a * sizeof(int));
    g_seq_b = (int*)malloc((size_t)len_b * sizeof(int));
    g_score = (int*)malloc((size_t)alphabet_size * (size_t)alphabet_size * sizeof(int));

    memcpy(g_seq_a, h_seq_a, (size_t)len_a * sizeof(int));
    memcpy(g_seq_b, h_seq_b, (size_t)len_b * sizeof(int));
    memcpy(g_score, h_score_matrix,
           (size_t)alphabet_size * (size_t)alphabet_size * sizeof(int));
}

void solution_compute(int* h_out_last_row) {
    int* prev = (int*)malloc((size_t)(g_len_b + 1) * sizeof(int));
    int* curr = (int*)malloc((size_t)(g_len_b + 1) * sizeof(int));

    prev[0] = 0;
    for (int j = 1; j <= g_len_b; ++j) {
        prev[j] = prev[j - 1] - g_gap;
    }

    for (int i = 1; i <= g_len_a; ++i) {
        curr[0] = -i * g_gap;
        int a = g_seq_a[i - 1] * g_alphabet;
        for (int j = 1; j <= g_len_b; ++j) {
            int diag = prev[j - 1] + g_score[a + g_seq_b[j - 1]];
            int up = prev[j] - g_gap;
            int left = curr[j - 1] - g_gap;
            int best = diag;
            if (up > best) best = up;
            if (left > best) best = left;
            curr[j] = best;
        }
        int* tmp = prev;
        prev = curr;
        curr = tmp;
    }

    for (int j = 0; j <= g_len_b; ++j) {
        h_out_last_row[j] = prev[j];
    }

    free(prev);
    free(curr);
}

void solution_free(void) {
    free(g_seq_a); g_seq_a = NULL;
    free(g_seq_b); g_seq_b = NULL;
    free(g_score); g_score = NULL;
    g_len_a = g_len_b = g_alphabet = g_gap = 0;
}
