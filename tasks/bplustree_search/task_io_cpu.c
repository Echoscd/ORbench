#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

extern void solution_init(
    int num_nodes,
    int order,
    int max_keys,
    int root_idx,
    const int* h_is_leaf,
    const int* h_key_counts,
    const int* h_keys,
    const int* h_children,
    const int* h_values
);

extern void solution_compute(
    int num_queries,
    const int* h_query_keys,
    int* h_out_values
);

extern void solution_free(void);

typedef struct {
    int num_nodes, order, max_keys, root_idx, num_queries;
    const int* is_leaf;
    const int* key_counts;
    const int* keys;
    const int* children;
    const int* values;
    const int* query_keys;
    int* out_values;
} BPTContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    BPTContext* ctx = (BPTContext*)calloc(1, sizeof(BPTContext));
    if (!ctx) return NULL;

    ctx->num_nodes  = (int)get_param(data, "num_nodes");
    ctx->order      = (int)get_param(data, "order");
    ctx->max_keys   = (int)get_param(data, "max_keys");
    ctx->root_idx   = (int)get_param(data, "root_idx");
    ctx->num_queries = (int)get_param(data, "num_queries");

    ctx->is_leaf    = get_tensor_int(data, "is_leaf");
    ctx->key_counts = get_tensor_int(data, "key_counts");
    ctx->keys       = get_tensor_int(data, "keys");
    ctx->children   = get_tensor_int(data, "children");
    ctx->values     = get_tensor_int(data, "values");
    ctx->query_keys = get_tensor_int(data, "query_keys");

    if (!ctx->is_leaf || !ctx->key_counts || !ctx->keys || !ctx->children ||
        !ctx->values || !ctx->query_keys) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        free(ctx);
        return NULL;
    }

    ctx->out_values = (int*)calloc((size_t)ctx->num_queries, sizeof(int));
    if (!ctx->out_values) {
        free(ctx);
        return NULL;
    }

    solution_init(
        ctx->num_nodes, ctx->order, ctx->max_keys, ctx->root_idx,
        ctx->is_leaf, ctx->key_counts, ctx->keys, ctx->children, ctx->values
    );
    return ctx;
}

void task_run(void* test_data) {
    BPTContext* ctx = (BPTContext*)test_data;
    solution_compute(ctx->num_queries, ctx->query_keys, ctx->out_values);
}

void task_write_output(void* test_data, const char* output_path) {
    BPTContext* ctx = (BPTContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    for (int i = 0; i < ctx->num_queries; ++i) {
        fprintf(f, "%d\n", ctx->out_values[i]);
    }
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    BPTContext* ctx = (BPTContext*)test_data;
    solution_free();
    free(ctx->out_values);
    free(ctx);
}
