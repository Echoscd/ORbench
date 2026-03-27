#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_num_nodes;
static int g_num_edges;
static int g_num_steps;
static int* g_edge_u;
static int* g_edge_v;
static float* g_edge_capacity;
static int* g_edge_transit_time;

void solution_init(
    int num_nodes,
    int num_edges,
    int num_steps,
    const int* edge_u,
    const int* edge_v,
    const float* edge_capacity,
    const int* edge_transit_time
) {
    g_num_nodes = num_nodes;
    g_num_edges = num_edges;
    g_num_steps = num_steps;
    
    g_edge_u = (int*)malloc(num_edges * sizeof(int));
    g_edge_v = (int*)malloc(num_edges * sizeof(int));
    g_edge_capacity = (float*)malloc(num_edges * sizeof(float));
    g_edge_transit_time = (int*)malloc(num_edges * sizeof(int));
    
    memcpy(g_edge_u, edge_u, num_edges * sizeof(int));
    memcpy(g_edge_v, edge_v, num_edges * sizeof(int));
    memcpy(g_edge_capacity, edge_capacity, num_edges * sizeof(float));
    memcpy(g_edge_transit_time, edge_transit_time, num_edges * sizeof(int));
}

void solution_compute(float inflow_rate, float* out_total_arrived) {
    float* node_flow = (float*)calloc(g_num_nodes, sizeof(float));
    float* queue_volume = (float*)calloc(g_num_edges, sizeof(float));
    float* pipeline = (float*)calloc(g_num_edges * 32, sizeof(float));
    
    int* dist = (int*)malloc(g_num_nodes * sizeof(int));
    int* dist_next = (int*)malloc(g_num_nodes * sizeof(int));
    int* next_edge = (int*)malloc(g_num_nodes * sizeof(int));
    int* next_edge_next = (int*)malloc(g_num_nodes * sizeof(int));
    
    float total_arrived = 0.0f;
    
    for (int t = 0; t < g_num_steps; t++) {
        // 1. Process arrivals from pipelines
        for (int e = 0; e < g_num_edges; e++) {
            int p_idx = e * 32 + (t % 32);
            float arrive = pipeline[p_idx];
            pipeline[p_idx] = 0.0f;
            node_flow[g_edge_v[e]] += arrive;
        }
        
        // 2. Add external inflow
        node_flow[0] += inflow_rate;
        
        // 3. Route node flows
        for (int i = 0; i < g_num_nodes; i++) {
            dist[i] = 1000000000;
            next_edge[i] = -1;
        }
        dist[g_num_nodes - 1] = 0;
        
        for (int iter = 0; iter < g_num_nodes; iter++) {
            int changed = 0;
            for (int i = 0; i < g_num_nodes; i++) {
                dist_next[i] = dist[i];
                next_edge_next[i] = next_edge[i];
            }
            
            for (int e = 0; e < g_num_edges; e++) {
                int u = g_edge_u[e];
                int v = g_edge_v[e];
                int w = g_edge_transit_time[e] * 1000 + (int)((queue_volume[e] / g_edge_capacity[e]) * 1000.0f);
                
                if (dist[v] < 500000000) {
                    int new_dist = dist[v] + w;
                    if (new_dist < dist_next[u]) {
                        dist_next[u] = new_dist;
                        next_edge_next[u] = e;
                        changed = 1;
                    } else if (new_dist == dist_next[u]) {
                        if (next_edge_next[u] == -1 || e < next_edge_next[u]) {
                            if (next_edge_next[u] != e) {
                                next_edge_next[u] = e;
                                changed = 1;
                            }
                        }
                    }
                }
            }
            
            for (int i = 0; i < g_num_nodes; i++) {
                dist[i] = dist_next[i];
                next_edge[i] = next_edge_next[i];
            }
            
            if (!changed) break;
        }
        
        // Move flow to queues
        for (int u = 0; u < g_num_nodes - 1; u++) {
            if (node_flow[u] > 0.0f && next_edge[u] != -1) {
                int e = next_edge[u];
                queue_volume[e] += node_flow[u];
                node_flow[u] = 0.0f;
            }
        }
        
        // 4. Process edge queues into pipelines
        for (int e = 0; e < g_num_edges; e++) {
            float leave = queue_volume[e];
            if (leave > g_edge_capacity[e]) {
                leave = g_edge_capacity[e];
            }
            queue_volume[e] -= leave;
            
            int arrival_time = t + g_edge_transit_time[e];
            int p_idx = e * 32 + (arrival_time % 32);
            pipeline[p_idx] += leave;
        }
        
        // 5. Accumulate destination flow
        total_arrived += node_flow[g_num_nodes - 1];
        node_flow[g_num_nodes - 1] = 0.0f;
    }
    
    *out_total_arrived = total_arrived;
    
    free(node_flow);
    free(queue_volume);
    free(pipeline);
    free(dist);
    free(dist_next);
    free(next_edge);
    free(next_edge_next);
}

void solution_free(void) {
    free(g_edge_u);
    free(g_edge_v);
    free(g_edge_capacity);
    free(g_edge_transit_time);
}
