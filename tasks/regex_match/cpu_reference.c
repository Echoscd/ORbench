// cpu_reference.c -- regex_match CPU baseline (Thompson NFA simulation)
//
// For each input string, simulates the NFA by maintaining a set of active
// states (as a bitmask for states <= 128, or an array otherwise), processing
// one character at a time, and checking if any accept state is reached.
//
// NO file I/O here. All I/O is handled by task_io_cpu.c.

#include <stdlib.h>
#include <string.h>

// Maximum NFA states we support (use array-based state set)
#define MAX_STATES 256

// ===== Module-level state =====
static int g_num_states;
static int g_num_symbols;
static int g_start_state;
static int g_num_strings;
static int g_total_chars;

static const int* g_trans_offsets;   // [num_states * num_symbols + 1]
static const int* g_trans_targets;   // [num_trans]
static const int* g_eps_offsets;     // [num_states + 1]
static const int* g_eps_targets;     // [num_eps]
static const int* g_is_accept;       // [num_states]
static const int* g_str_offsets;     // [num_strings + 1]
static const int* g_str_data;        // [total_chars]

void solution_init(int num_states, int num_symbols, int start_state,
                   int num_strings, int total_chars,
                   const int* trans_offsets, const int* trans_targets,
                   const int* eps_offsets, const int* eps_targets,
                   const int* is_accept,
                   const int* str_offsets, const int* str_data) {
    g_num_states = num_states;
    g_num_symbols = num_symbols;
    g_start_state = start_state;
    g_num_strings = num_strings;
    g_total_chars = total_chars;
    g_trans_offsets = trans_offsets;
    g_trans_targets = trans_targets;
    g_eps_offsets = eps_offsets;
    g_eps_targets = eps_targets;
    g_is_accept = is_accept;
    g_str_offsets = str_offsets;
    g_str_data = str_data;
}

// ===== Epsilon closure via BFS =====
// active[s] = 1 if state s is in the set
// Uses a queue for BFS
static void epsilon_closure(const int* eps_offsets, const int* eps_targets,
                            int num_states, char* active) {
    int queue[MAX_STATES];
    int qhead = 0, qtail = 0;

    // Initialize queue with all currently active states
    for (int s = 0; s < num_states; s++) {
        if (active[s]) {
            queue[qtail++] = s;
        }
    }

    // BFS
    while (qhead < qtail) {
        int s = queue[qhead++];
        int start = eps_offsets[s];
        int end = eps_offsets[s + 1];
        for (int i = start; i < end; i++) {
            int t = eps_targets[i];
            if (!active[t]) {
                active[t] = 1;
                queue[qtail++] = t;
            }
        }
    }
}

// ===== Match a single string =====
static int nfa_match_one(const int* str, int str_len) {
    char current[MAX_STATES];
    char next[MAX_STATES];

    int ns = g_num_states;
    int nsym = g_num_symbols;

    // Start: epsilon closure of {start_state}
    memset(current, 0, (size_t)ns);
    current[g_start_state] = 1;
    epsilon_closure(g_eps_offsets, g_eps_targets, ns, current);

    // Process each character
    for (int pos = 0; pos < str_len; pos++) {
        int c = str[pos];
        if (c < 0 || c >= nsym) {
            // Invalid symbol -> dead
            return 0;
        }

        memset(next, 0, (size_t)ns);

        // For each active state, follow transitions on symbol c
        for (int s = 0; s < ns; s++) {
            if (!current[s]) continue;
            int idx = s * nsym + c;
            int start = g_trans_offsets[idx];
            int end = g_trans_offsets[idx + 1];
            for (int i = start; i < end; i++) {
                next[g_trans_targets[i]] = 1;
            }
        }

        // Epsilon closure of next states
        epsilon_closure(g_eps_offsets, g_eps_targets, ns, next);

        // Swap
        memcpy(current, next, (size_t)ns);

        // Early termination: no active states
        int any_active = 0;
        for (int s = 0; s < ns; s++) {
            if (current[s]) { any_active = 1; break; }
        }
        if (!any_active) return 0;
    }

    // Check if any accept state is active
    for (int s = 0; s < ns; s++) {
        if (current[s] && g_is_accept[s]) return 1;
    }
    return 0;
}

// ===== solution_compute: match all strings =====
void solution_compute(int num_strings, int* results) {
    for (int i = 0; i < num_strings; i++) {
        int offset = g_str_offsets[i];
        int len = g_str_offsets[i + 1] - offset;
        results[i] = nfa_match_one(&g_str_data[offset], len);
    }
}
