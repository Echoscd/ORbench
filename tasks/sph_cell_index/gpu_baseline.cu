// gpu_baseline.cu -- SPH cell-linked list construction (GPU baseline)
//
// Faithfully ported from DualSPHysics JCellDivGpu_ker.cu
//   KerCalcBeginEndCell + thrust::sort_by_key for sorting
//
// Steps: (1) compute cell_id per particle (kernel)
//        (2) sort by cell_id (thrust::sort_by_key)
//        (3) find begin/end per cell (KerCalcBeginEndCell kernel)
//
// Persistent GPU memory (cudaMalloc in init, reuse in compute).

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// ===== Module-level persistent state =====
static int    g_N;
static float  g_cell_size;
static int    g_grid_nx;
static int    g_grid_ny;
static int    g_grid_nz;

// Device arrays: inputs
static float* d_xs;
static float* d_ys;
static float* d_zs;

// Device arrays: intermediates and outputs
static int*   d_cell_ids;      // cell_id per particle
static int*   d_indices;       // particle indices (to be sorted alongside cell_ids)
static int*   d_cell_begin;
static int*   d_cell_end;

// ===== Kernel: compute cell index for each particle =====
__global__ void KerComputeCellIndex(int N,
                                    const float* __restrict__ xs,
                                    const float* __restrict__ ys,
                                    const float* __restrict__ zs,
                                    float cell_size, int nx, int ny,
                                    int* __restrict__ cell_ids,
                                    int* __restrict__ indices)
{
    unsigned int pt = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt < (unsigned int)N) {
        int cx = (int)floorf(xs[pt] / cell_size);
        int cy = (int)floorf(ys[pt] / cell_size);
        int cz = (int)floorf(zs[pt] / cell_size);
        // Clamp to valid range
        if (cx < 0) cx = 0;
        if (cy < 0) cy = 0;
        if (cz < 0) cz = 0;
        if (cx >= nx) cx = nx - 1;
        if (cy >= ny) cy = ny - 1;
        cell_ids[pt] = cx + cy * nx + cz * nx * ny;
        indices[pt] = pt;
    }
}

// ===== Kernel: KerCalcBeginEndCell =====
// Ported from DualSPHysics JCellDivGpu_ker.cu KerCalcBeginEndCell
// Uses shared memory to detect cell boundaries in the sorted cell_id array.
__global__ void KerCalcBeginEndCell(unsigned int n,
                                    const int* __restrict__ sorted_cell_ids,
                                    int* __restrict__ cell_begin,
                                    int* __restrict__ cell_end)
{
    extern __shared__ int scell[];  // [blockDim.x + 1]
    const unsigned int pt = blockIdx.x * blockDim.x + threadIdx.x;

    int cel;
    if (pt < n) {
        cel = sorted_cell_ids[pt];
        scell[threadIdx.x + 1] = cel;
        if (pt > 0 && threadIdx.x == 0)
            scell[0] = sorted_cell_ids[pt - 1];
    }
    __syncthreads();

    if (pt < n) {
        if (pt == 0 || cel != scell[threadIdx.x]) {
            cell_begin[cel] = (int)pt;
            if (pt > 0)
                cell_end[scell[threadIdx.x]] = (int)pt;
        }
        if (pt == n - 1)
            cell_end[cel] = (int)(pt + 1);
    }
}

// ===== Public interface (extern "C" for task_io linkage) =====

extern "C" {

static int g_num_cells = 0;

void solution_free(void)
{
    if (d_xs)         { cudaFree(d_xs);         d_xs         = nullptr; }
    if (d_ys)         { cudaFree(d_ys);         d_ys         = nullptr; }
    if (d_zs)         { cudaFree(d_zs);         d_zs         = nullptr; }
    if (d_cell_ids)   { cudaFree(d_cell_ids);   d_cell_ids   = nullptr; }
    if (d_indices)    { cudaFree(d_indices);    d_indices    = nullptr; }
    if (d_cell_begin) { cudaFree(d_cell_begin); d_cell_begin = nullptr; }
    if (d_cell_end)   { cudaFree(d_cell_end);   d_cell_end   = nullptr; }
    g_N = 0;
    g_num_cells = 0;
}

void solution_compute(int N,
                      const float* xs, const float* ys, const float* zs,
                      float cell_size,
                      int grid_nx, int grid_ny, int grid_nz,
                      int num_cells,
                      int* sorted_indices, int* cell_begin, int* cell_end)
{
    size_t sz_f = (size_t)N * sizeof(float);
    size_t sz_i = (size_t)N * sizeof(int);
    size_t sz_cells = (size_t)num_cells * sizeof(int);

    if (g_N != N || g_num_cells != num_cells) {
        solution_free();
        cudaMalloc(&d_xs, sz_f);
        cudaMalloc(&d_ys, sz_f);
        cudaMalloc(&d_zs, sz_f);
        cudaMalloc(&d_cell_ids, sz_i);
        cudaMalloc(&d_indices, sz_i);
        cudaMalloc(&d_cell_begin, sz_cells);
        cudaMalloc(&d_cell_end, sz_cells);
        g_N = N;
        g_num_cells = num_cells;
    }

    g_cell_size = cell_size;
    g_grid_nx = grid_nx;
    g_grid_ny = grid_ny;
    g_grid_nz = grid_nz;

    cudaMemcpy(d_xs, xs, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ys, ys, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_zs, zs, sz_f, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Step 1: Compute cell_id for each particle
    KerComputeCellIndex<<<grid, block>>>(
        N, d_xs, d_ys, d_zs,
        cell_size, grid_nx, grid_ny,
        d_cell_ids, d_indices);

    // Step 2: Sort by cell_id using thrust::sort_by_key
    // (matches DualSPHysics SortDataParticles)
    thrust::device_ptr<int> keys(d_cell_ids);
    thrust::device_ptr<int> vals(d_indices);
    thrust::sort_by_key(keys, keys + N, vals);

    // Step 3: Initialize cell_begin/cell_end to -1
    cudaMemset(d_cell_begin, 0xFF, (size_t)num_cells * sizeof(int));  // -1 for signed int
    cudaMemset(d_cell_end,   0xFF, (size_t)num_cells * sizeof(int));

    // Step 4: Find begin/end per cell (KerCalcBeginEndCell)
    size_t shared_mem = (BLOCK_SIZE + 1) * sizeof(int);
    KerCalcBeginEndCell<<<grid, block, shared_mem>>>(
        (unsigned int)N, d_cell_ids, d_cell_begin, d_cell_end);

    // Copy results back to host
    cudaMemcpy(sorted_indices, d_indices,    (size_t)N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cell_begin,     d_cell_begin, (size_t)num_cells * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cell_end,       d_cell_end,   (size_t)num_cells * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

} // extern "C"
