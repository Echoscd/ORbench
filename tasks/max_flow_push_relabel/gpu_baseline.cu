// gpu_baseline.cu — Push-Relabel Maximum Flow (GPU baseline)
//
// This baseline runs the push-relabel algorithm on a single GPU thread,
// faithfully matching the CPU reference. The LLM's task is to parallelize:
//   * GlobalUpdate → parallel BFS (frontier expansion with atomics)
//   * Discharge → parallel push on independent active nodes
//   * Relabel → parallel relabel of non-adjacent nodes
//
// Optimization: all graph data uploaded to device once in solution_init.

#include <cuda_runtime.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ===== Device-side push-relabel (serial, single thread) =====
__global__ void push_relabel_kernel(
    int num_nodes, int num_total_arcs,
    int source, int sink,
    const int* adj_start, const int* adj_list,
    const int* arc_head, const int* arc_opposite,
    int* residual, const int* initial_cap,
    long long* excess, int* height, int* first_arc,
    int* bfs_queue, int* in_queue,
    int* result)
{
    if (threadIdx.x != 0) return;

    // InitializePreflow
    for (int i = 0; i < num_nodes; i++) { excess[i] = 0; height[i] = 0; first_arc[i] = adj_start[i]; }
    for (int a = 0; a < num_total_arcs; a++) residual[a] = initial_cap[a];
    height[source] = num_nodes;

    // SaturateSource
    for (int i = adj_start[source]; i < adj_start[source + 1]; i++) {
        int arc = adj_list[i];
        int flow = residual[arc];
        if (flow > 0 && height[arc_head[arc]] < num_nodes) {
            residual[arc] -= flow;
            residual[arc_opposite[arc]] += flow;
            excess[source] -= flow;
            excess[arc_head[arc]] += flow;
        }
    }

    // GlobalUpdate (BFS from sink)
    for (int i = 0; i < num_nodes; i++) in_queue[i] = 0;
    in_queue[sink] = 1; in_queue[source] = 1;
    int qfront = 0, qback = 0;
    bfs_queue[qback++] = sink;
    while (qfront < qback) {
        int node = bfs_queue[qfront++];
        int cd = height[node] + 1;
        for (int i = adj_start[node]; i < adj_start[node + 1]; i++) {
            int arc = adj_list[i];
            int head = arc_head[arc];
            if (in_queue[head]) continue;
            if (residual[arc_opposite[arc]] > 0) {
                height[head] = cd;
                in_queue[head] = 1;
                bfs_queue[qback++] = head;
            }
        }
    }
    for (int i = 0; i < num_nodes; i++) {
        if (!in_queue[i]) height[i] = 2 * num_nodes - 1;
    }

    // Build active stack
    int active_stack[4096];  // max active nodes (adjust if needed)
    int atop = 0;
    for (int i = 1; i < qback; i++) {
        int node = bfs_queue[i];
        if (node != source && node != sink && excess[node] > 0) {
            if (atop < 4096) active_stack[atop++] = node;
        }
    }

    // Discharge loop
    while (atop > 0) {
        int node = active_stack[--atop];
        if (excess[node] <= 0) continue;
        if (node == source || node == sink) continue;
        if (height[node] >= num_nodes) continue;

        // Discharge
        while (excess[node] > 0 && height[node] < num_nodes) {
            int pushed = 0;
            for (int i = first_arc[node]; i < adj_start[node + 1]; i++) {
                int arc = adj_list[i];
                if (residual[arc] > 0 && height[node] == height[arc_head[arc]] + 1) {
                    int head = arc_head[arc];
                    if (excess[head] == 0 && head != source && head != sink) {
                        if (atop < 4096) active_stack[atop++] = head;
                    }
                    int flow = residual[arc];
                    if ((long long)flow > excess[node]) flow = (int)excess[node];
                    residual[arc] -= flow;
                    residual[arc_opposite[arc]] += flow;
                    excess[node] -= flow;
                    excess[head] += flow;
                    pushed = 1;
                    if (excess[node] == 0) { first_arc[node] = i; break; }
                }
            }
            if (!pushed) {
                // Relabel
                int minh = INT_MAX;
                int best = adj_start[node];
                for (int i = adj_start[node]; i < adj_start[node + 1]; i++) {
                    int arc = adj_list[i];
                    if (residual[arc] > 0) {
                        int h = height[arc_head[arc]];
                        if (h < minh) { minh = h; best = i; }
                    }
                }
                height[node] = minh + 1;
                first_arc[node] = best;
            }
        }
    }

    *result = (int)excess[sink];
}

// ===== Host state =====
static int g_num_nodes = 0, g_num_total_arcs = 0;
static int g_source = 0, g_sink = 0;

static int* d_adj_start = nullptr;
static int* d_adj_list = nullptr;
static int* d_arc_head = nullptr;
static int* d_arc_opposite = nullptr;
static int* d_residual = nullptr;
static int* d_initial_cap = nullptr;
static long long* d_excess = nullptr;
static int* d_height = nullptr;
static int* d_first_arc = nullptr;
static int* d_bfs_queue = nullptr;
static int* d_in_queue = nullptr;
static int* d_result = nullptr;

extern "C" void solution_free(void)
{
    if (d_adj_start)    { cudaFree(d_adj_start);    d_adj_start    = nullptr; }
    if (d_adj_list)     { cudaFree(d_adj_list);     d_adj_list     = nullptr; }
    if (d_arc_head)     { cudaFree(d_arc_head);     d_arc_head     = nullptr; }
    if (d_arc_opposite) { cudaFree(d_arc_opposite); d_arc_opposite = nullptr; }
    if (d_residual)     { cudaFree(d_residual);     d_residual     = nullptr; }
    if (d_initial_cap)  { cudaFree(d_initial_cap);  d_initial_cap  = nullptr; }
    if (d_excess)       { cudaFree(d_excess);       d_excess       = nullptr; }
    if (d_height)       { cudaFree(d_height);       d_height       = nullptr; }
    if (d_first_arc)    { cudaFree(d_first_arc);    d_first_arc    = nullptr; }
    if (d_bfs_queue)    { cudaFree(d_bfs_queue);    d_bfs_queue    = nullptr; }
    if (d_in_queue)     { cudaFree(d_in_queue);     d_in_queue     = nullptr; }
    if (d_result)       { cudaFree(d_result);       d_result       = nullptr; }
    g_num_nodes = 0;
    g_num_total_arcs = 0;
}

extern "C" void solution_compute(int num_nodes, int num_arcs,
                                 const int* tails, const int* heads, const int* caps,
                                 int source, int sink,
                                 int* max_flow_out)
{
    int total = 2 * num_arcs;

    // Build graph on host (same as CPU reference)
    int* h_arc_head = (int*)malloc(total * sizeof(int));
    int* h_arc_opp  = (int*)malloc(total * sizeof(int));
    int* h_init_cap = (int*)malloc(total * sizeof(int));

    for (int i = 0; i < num_arcs; i++) {
        h_arc_head[i]            = heads[i];
        h_arc_head[num_arcs + i] = tails[i];
        h_arc_opp[i]             = num_arcs + i;
        h_arc_opp[num_arcs + i]  = i;
        h_init_cap[i]            = caps[i];
        h_init_cap[num_arcs + i] = 0;
    }

    int* degree = (int*)calloc(num_nodes, sizeof(int));
    for (int a = 0; a < total; a++) {
        int tail = (a < num_arcs) ? tails[a] : heads[a - num_arcs];
        degree[tail]++;
    }
    int* h_adj_start = (int*)malloc((num_nodes + 1) * sizeof(int));
    h_adj_start[0] = 0;
    for (int i = 0; i < num_nodes; i++) h_adj_start[i + 1] = h_adj_start[i] + degree[i];
    int* h_adj_list = (int*)malloc(total * sizeof(int));
    memset(degree, 0, num_nodes * sizeof(int));
    for (int a = 0; a < total; a++) {
        int tail = (a < num_arcs) ? tails[a] : heads[a - num_arcs];
        h_adj_list[h_adj_start[tail] + degree[tail]] = a;
        degree[tail]++;
    }
    free(degree);

    if (g_num_nodes != num_nodes || g_num_total_arcs != total) {
        solution_free();
        cudaMalloc(&d_adj_start, (num_nodes + 1) * sizeof(int));
        cudaMalloc(&d_adj_list, total * sizeof(int));
        cudaMalloc(&d_arc_head, total * sizeof(int));
        cudaMalloc(&d_arc_opposite, total * sizeof(int));
        cudaMalloc(&d_residual, total * sizeof(int));
        cudaMalloc(&d_initial_cap, total * sizeof(int));
        cudaMalloc(&d_excess, num_nodes * sizeof(long long));
        cudaMalloc(&d_height, num_nodes * sizeof(int));
        cudaMalloc(&d_first_arc, num_nodes * sizeof(int));
        cudaMalloc(&d_bfs_queue, num_nodes * sizeof(int));
        cudaMalloc(&d_in_queue, num_nodes * sizeof(int));
        cudaMalloc(&d_result, sizeof(int));
        g_num_nodes = num_nodes;
        g_num_total_arcs = total;
    }

    cudaMemcpy(d_adj_start, h_adj_start, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj_list, h_adj_list, total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arc_head, h_arc_head, total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arc_opposite, h_arc_opp, total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_initial_cap, h_init_cap, total * sizeof(int), cudaMemcpyHostToDevice);

    free(h_arc_head); free(h_arc_opp); free(h_init_cap);
    free(h_adj_start); free(h_adj_list);

    g_source = source;
    g_sink = sink;

    push_relabel_kernel<<<1, 1>>>(
        num_nodes, total, source, sink,
        d_adj_start, d_adj_list, d_arc_head, d_arc_opposite,
        d_residual, d_initial_cap, d_excess, d_height, d_first_arc,
        d_bfs_queue, d_in_queue, d_result);

    cudaMemcpy(max_flow_out, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
