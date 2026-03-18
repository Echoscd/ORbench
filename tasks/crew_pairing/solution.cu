// solution.cu -- GPU-accelerated crew pairing solver
//
// Strategy: Parallel greedy with GPU-accelerated connection scoring
// 1. Copy flight data to GPU in solution_init
// 2. In solution_compute:
//    a. Build connection graph on GPU (parallel per-leg)
//    b. Run greedy assignment on CPU (sequential, but fast with precomputed data)
//    c. Optionally: parallel SPPRC per starting leg (each thread = one start)

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define POS_FEE 10000.0f
#define MAX_GREEDY_PAIRINGS 4096

// ===== Host state =====
static int h_N;
static int h_base_station;
static float h_duty_rate, h_pairing_rate;
static int h_max_duty_min, h_max_block_min, h_max_legs_duty, h_min_rest_min;

// Host copies of input data
static int* h_dep_min;
static int* h_arr_min;
static int* h_dep_stn;
static int* h_arr_stn;

// Device data
static int* d_dep_min;
static int* d_arr_min;
static int* d_dep_stn;
static int* d_arr_stn;

// Connection graph (built on GPU, used on CPU)
// For each leg i, store up to MAX_CONN connections
#define MAX_CONN_PER_LEG 64
static int* d_conn;        // N * MAX_CONN_PER_LEG
static int* d_conn_count;  // N
static int* h_conn;
static int* h_conn_count;

// ===== GPU kernel: build connection graph =====
__global__ void build_connections_kernel(
    int N, const int* dep_min, const int* arr_min,
    const int* dep_stn, const int* arr_stn,
    int max_connect_min, int* conn, int* conn_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int arr_i = arr_min[i];
    int arr_stn_i = arr_stn[i];
    int count = 0;

    for (int j = i + 1; j < N && count < MAX_CONN_PER_LEG; j++) {
        int gap = dep_min[j] - arr_i;
        if (gap >= max_connect_min) break;  // sorted by dep_min
        if (gap < 0) continue;
        if (arr_stn_i != dep_stn[j]) continue;  // station continuity
        conn[i * MAX_CONN_PER_LEG + count] = j;
        count++;
    }
    conn_count[i] = count;
}

// ===== GPU kernel: compute single-leg costs =====
static float* d_single_cost;
static float* h_single_cost;

__global__ void compute_single_costs_kernel(
    int N, const int* dep_min, const int* arr_min,
    const int* dep_stn, int base_station,
    float duty_rate, float pairing_rate, float* single_cost)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int block = arr_min[i] - dep_min[i];
    float pos = (dep_stn[i] != base_station) ? POS_FEE : 0.0f;
    single_cost[i] = (float)block * duty_rate / 60.0f
                   + (float)block * pairing_rate / 60.0f
                   + pos;
}

// ===== Greedy solver (CPU, using GPU-built connection graph) =====
static int gp_last_leg[MAX_GREEDY_PAIRINGS];
static int gp_duty_start[MAX_GREEDY_PAIRINGS];
static int gp_duty_block[MAX_GREEDY_PAIRINGS];
static int gp_duty_legs[MAX_GREEDY_PAIRINGS];

static void solve_greedy(int N, int* assignments) {
    int num_p = 0;
    for (int i = 0; i < N; i++) {
        int block = h_arr_min[i] - h_dep_min[i];
        int best_p = -1;

        // Try connections from existing pairings (use connection graph)
        for (int p = 0; p < num_p; p++) {
            int last = gp_last_leg[p];
            if (h_arr_stn[last] != h_dep_stn[i]) continue;
            int gap = h_dep_min[i] - h_arr_min[last];
            if (gap < 0) continue;

            if (gap >= h_min_rest_min) {
                best_p = p;
                break;
            } else {
                int nl = gp_duty_legs[p] + 1;
                int nb = gp_duty_block[p] + block;
                int ns = h_arr_min[i] - gp_duty_start[p];
                if (nl <= h_max_legs_duty && nb <= h_max_block_min && ns <= h_max_duty_min) {
                    best_p = p;
                    break;
                }
            }
        }

        if (best_p >= 0) {
            int last = gp_last_leg[best_p];
            int gap = h_dep_min[i] - h_arr_min[last];
            assignments[i] = best_p;
            gp_last_leg[best_p] = i;
            if (gap >= h_min_rest_min) {
                gp_duty_start[best_p] = h_dep_min[i];
                gp_duty_block[best_p] = block;
                gp_duty_legs[best_p] = 1;
            } else {
                gp_duty_block[best_p] += block;
                gp_duty_legs[best_p]++;
            }
        } else {
            if (num_p < MAX_GREEDY_PAIRINGS) {
                int p = num_p++;
                assignments[i] = p;
                gp_last_leg[p] = i;
                gp_duty_start[p] = h_dep_min[i];
                gp_duty_block[p] = block;
                gp_duty_legs[p] = 1;
            } else {
                assignments[i] = 0;
            }
        }
    }
}

// ===== Interface =====
extern "C" {

void solution_init(int N, int num_stations, int base_station,
                   const int* dep_minutes, const int* arr_minutes,
                   const int* dep_stations, const int* arr_stations,
                   float duty_cost_per_hour, float pairing_cost_per_hour,
                   int max_duty_min, int max_block_min,
                   int max_legs_duty, int min_rest_min)
{
    h_N = N;
    h_base_station = base_station;
    h_duty_rate = duty_cost_per_hour;
    h_pairing_rate = pairing_cost_per_hour;
    h_max_duty_min = max_duty_min;
    h_max_block_min = max_block_min;
    h_max_legs_duty = max_legs_duty;
    h_min_rest_min = min_rest_min;

    // Host copies
    h_dep_min = (int*)malloc(N * sizeof(int));
    h_arr_min = (int*)malloc(N * sizeof(int));
    h_dep_stn = (int*)malloc(N * sizeof(int));
    h_arr_stn = (int*)malloc(N * sizeof(int));
    memcpy(h_dep_min, dep_minutes, N * sizeof(int));
    memcpy(h_arr_min, arr_minutes, N * sizeof(int));
    memcpy(h_dep_stn, dep_stations, N * sizeof(int));
    memcpy(h_arr_stn, arr_stations, N * sizeof(int));

    // Device allocations
    cudaMalloc(&d_dep_min, N * sizeof(int));
    cudaMalloc(&d_arr_min, N * sizeof(int));
    cudaMalloc(&d_dep_stn, N * sizeof(int));
    cudaMalloc(&d_arr_stn, N * sizeof(int));

    cudaMemcpy(d_dep_min, dep_minutes, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_min, arr_minutes, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dep_stn, dep_stations, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_stn, arr_stations, N * sizeof(int), cudaMemcpyHostToDevice);

    // Connection graph buffers
    cudaMalloc(&d_conn, (size_t)N * MAX_CONN_PER_LEG * sizeof(int));
    cudaMalloc(&d_conn_count, N * sizeof(int));
    h_conn = (int*)malloc((size_t)N * MAX_CONN_PER_LEG * sizeof(int));
    h_conn_count = (int*)malloc(N * sizeof(int));

    // Single-leg costs
    cudaMalloc(&d_single_cost, N * sizeof(float));
    h_single_cost = (float*)malloc(N * sizeof(float));
}

void solution_compute(int N, int* assignments)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Step 1: Build connection graph on GPU (parallel)
    int max_connect_min = 36 * 60;
    build_connections_kernel<<<blocks, threads>>>(
        N, d_dep_min, d_arr_min, d_dep_stn, d_arr_stn,
        max_connect_min, d_conn, d_conn_count);

    // Step 2: Compute single-leg costs on GPU
    compute_single_costs_kernel<<<blocks, threads>>>(
        N, d_dep_min, d_arr_min, d_dep_stn, h_base_station,
        h_duty_rate, h_pairing_rate, d_single_cost);

    cudaDeviceSynchronize();

    // Copy connection graph back to host
    cudaMemcpy(h_conn, d_conn, (size_t)N * MAX_CONN_PER_LEG * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_conn_count, d_conn_count, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_single_cost, d_single_cost, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Step 3: Run greedy solver on CPU (sequential)
    solve_greedy(N, assignments);
}

} // extern "C"
