// cpu_reference.c -- Smith-Waterman local sequence alignment CPU baseline
//
// Implements solution_init / solution_compute for local alignment scoring.
// NO file I/O. All I/O handled by task_io_cpu.c.
//
// Build (via task_io + harness):
//   gcc -O2 -I framework/
//       framework/harness_cpu.c tasks/smith_waterman/task_io_cpu.c
//       tasks/smith_waterman/cpu_reference.c -o solution_cpu -lm

#include <stdlib.h>
#include <string.h>

// ===== Module-level state =====
static int g_N;
static int g_match_score;
static int g_mismatch_penalty;
static int g_gap_penalty;
static const int* g_query_seqs;
static const int* g_target_seqs;
static const int* g_query_offsets;
static const int* g_target_offsets;

void solution_init(int N,
                   const int* query_seqs, const int* target_seqs,
                   const int* query_offsets, const int* target_offsets,
                   int match_score, int mismatch_penalty, int gap_penalty) {
    g_N = N;
    g_query_seqs = query_seqs;
    g_target_seqs = target_seqs;
    g_query_offsets = query_offsets;
    g_target_offsets = target_offsets;
    g_match_score = match_score;
    g_mismatch_penalty = mismatch_penalty;
    g_gap_penalty = gap_penalty;
}

static int max2(int a, int b) { return a > b ? a : b; }
static int max3(int a, int b, int c) { return max2(a, max2(b, c)); }

void solution_compute(int N, int* scores) {
    for (int p = 0; p < N; p++) {
        int q_start = g_query_offsets[p];
        int q_end   = g_query_offsets[p + 1];
        int t_start = g_target_offsets[p];
        int t_end   = g_target_offsets[p + 1];
        int m = q_end - q_start;   // query length
        int n = t_end - t_start;   // target length

        // Allocate DP matrix (m+1) x (n+1), row-major
        // Use two rows to save memory: prev and curr
        int* prev = (int*)calloc((size_t)(n + 1), sizeof(int));
        int* curr = (int*)calloc((size_t)(n + 1), sizeof(int));

        int best = 0;

        for (int i = 1; i <= m; i++) {
            int qi = g_query_seqs[q_start + i - 1];
            for (int j = 1; j <= n; j++) {
                int tj = g_target_seqs[t_start + j - 1];
                int s = (qi == tj) ? g_match_score : g_mismatch_penalty;

                int val = max3(
                    prev[j - 1] + s,          // diagonal: match/mismatch
                    prev[j]     + g_gap_penalty, // up: gap in target
                    curr[j - 1] + g_gap_penalty  // left: gap in query
                );
                if (val < 0) val = 0;  // local alignment: no negative scores
                curr[j] = val;
                if (val > best) best = val;
            }
            // Swap rows
            int* tmp = prev;
            prev = curr;
            curr = tmp;
            memset(curr, 0, (size_t)(n + 1) * sizeof(int));
        }

        scores[p] = best;
        free(prev);
        free(curr);
    }
}
