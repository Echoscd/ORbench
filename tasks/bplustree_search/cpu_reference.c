#include <stddef.h>

static int g_num_nodes = 0;
static int g_order = 0;
static int g_max_keys = 0;
static int g_root_idx = 0;
static const int* g_is_leaf = NULL;
static const int* g_key_counts = NULL;
static const int* g_keys = NULL;
static const int* g_children = NULL;
static const int* g_values = NULL;

void solution_init(
    int num_nodes,
    int order,
    int max_keys,
    int root_idx,
    const int* h_is_leaf,
    const int* h_key_counts,
    const int* h_keys,
    const int* h_children,
    const int* h_values
) {
    g_num_nodes = num_nodes;
    g_order = order;
    g_max_keys = max_keys;
    g_root_idx = root_idx;
    g_is_leaf = h_is_leaf;
    g_key_counts = h_key_counts;
    g_keys = h_keys;
    g_children = h_children;
    g_values = h_values;
}

static int lookup_one(int query) {
    int node = g_root_idx;
    while (node >= 0 && node < g_num_nodes) {
        const int* node_keys = g_keys + (size_t)node * (size_t)g_max_keys;
        const int count = g_key_counts[node];
        if (g_is_leaf[node]) {
            const int* node_vals = g_values + (size_t)node * (size_t)g_max_keys;
            for (int i = 0; i < count; ++i) {
                if (node_keys[i] == query) return node_vals[i];
                if (node_keys[i] > query) break;
            }
            return -1;
        } else {
            const int* node_children = g_children + (size_t)node * (size_t)g_order;
            int child_pos = count;
            for (int i = 0; i < count; ++i) {
                if (query < node_keys[i]) {
                    child_pos = i;
                    break;
                }
            }
            node = node_children[child_pos];
        }
    }
    return -1;
}

void solution_compute(int num_queries, const int* h_query_keys, int* h_out_values) {
    for (int i = 0; i < num_queries; ++i) {
        h_out_values[i] = lookup_one(h_query_keys[i]);
    }
}

void solution_free(void) {
    g_num_nodes = 0;
    g_order = 0;
    g_max_keys = 0;
    g_root_idx = 0;
    g_is_leaf = NULL;
    g_key_counts = NULL;
    g_keys = NULL;
    g_children = NULL;
    g_values = NULL;
}
