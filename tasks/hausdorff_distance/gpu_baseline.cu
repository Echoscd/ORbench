// gpu_baseline.cu — All-pairs directed Hausdorff distance (GPU baseline)
//
// Faithfully ported from cuSpatial (rapidsai/cuspatial,
// cpp/include/cuspatial/detail/distance/hausdorff.cuh::kernel_hausdorff).
// The decomposition is "one CUDA thread per LHS point": each thread walks
// over every RHS space, computes the per-(LHS-point, RHS-space) min distance
// in registers, then atomically updates the per-(LHS-space, RHS-space) max
// in the output via atomicMax. The output layout matches the original
// (`results[rhs_space_idx * num_spaces + lhs_space_idx]`).
//
// To stay simple, this baseline assumes 2D points (vec_2d<float>) — same as
// the typical cuSpatial use case for trajectory similarity.

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

struct vec_2d {
    float x;
    float y;
};

__device__ inline float magnitude_squared(float a, float b)
{
    return a * a + b * b;
}

// CUDA float atomicMax via integer CAS (atomicMax for float not supported pre-sm_90)
__device__ inline float atomicMaxFloat(float* addr, float value)
{
    int* addr_i = (int*)addr;
    int old = *addr_i, assumed;
    do {
        assumed = old;
        float old_f = __int_as_float(assumed);
        if (value <= old_f) break;
        old = atomicCAS(addr_i, assumed, __float_as_int(value));
    } while (assumed != old);
    return __int_as_float(old);
}

// ===== Kernel: faithful port of cuspatial::detail::kernel_hausdorff =====
__global__ void kernel_hausdorff(int             num_points,
                                 const vec_2d*  __restrict__ points,
                                 int             num_spaces,
                                 const int*     __restrict__ space_offsets,
                                 float*         __restrict__ results)
{
    int lhs_p_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (lhs_p_idx >= num_points) return;

    // Determine the LHS space this point belongs to via binary search
    // (mirrors cuspatial's thrust::upper_bound + thrust::prev).
    int lo = 0;
    int hi = num_spaces;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (space_offsets[mid] <= lhs_p_idx) lo = mid + 1;
        else                                  hi = mid;
    }
    int lhs_space_idx = lo - 1;

    vec_2d lhs_p = points[lhs_p_idx];

    // Loop over each RHS space.
    for (int rhs_space_idx = 0; rhs_space_idx < num_spaces; rhs_space_idx++) {
        int rhs_p_idx_begin = space_offsets[rhs_space_idx];
        int rhs_p_idx_end   = (rhs_space_idx + 1 == num_spaces)
                                ? num_points
                                : space_offsets[rhs_space_idx + 1];

        float min_distance_squared = INFINITY;

        for (int rhs_p_idx = rhs_p_idx_begin; rhs_p_idx < rhs_p_idx_end; rhs_p_idx++) {
            vec_2d rhs_p = points[rhs_p_idx];
            float dsq = magnitude_squared(rhs_p.x - lhs_p.x,
                                          rhs_p.y - lhs_p.y);
            if (dsq < min_distance_squared) min_distance_squared = dsq;
        }

        int output_idx = rhs_space_idx * num_spaces + lhs_space_idx;
        atomicMaxFloat(&results[output_idx], sqrtf(min_distance_squared));
    }
}

// ===== Persistent device state =====
static int     g_num_points = 0;
static int     g_num_spaces = 0;
static vec_2d* d_points = nullptr;
static int*    d_space_offsets = nullptr;
static float*  d_results = nullptr;

// Tiny init kernel (cuSpatial uses thrust::fill_n with -1 sentinel for atomicMax).
__global__ void hausdorff_init_results(int n, float* r)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = -1.0f;
}

extern "C" void solution_free(void)
{
    if (d_points)        { cudaFree(d_points);        d_points        = nullptr; }
    if (d_space_offsets) { cudaFree(d_space_offsets); d_space_offsets = nullptr; }
    if (d_results)       { cudaFree(d_results);       d_results       = nullptr; }
    g_num_points = 0;
    g_num_spaces = 0;
}

extern "C" void solution_compute(int          num_points,
                                 int          num_spaces,
                                 const float* points_xy,
                                 const int*   space_offsets,
                                 float*       results)
{
    size_t pts_bytes = (size_t)num_points * sizeof(vec_2d);
    size_t off_bytes = (size_t)num_spaces * sizeof(int);
    size_t res_bytes = (size_t)num_spaces * num_spaces * sizeof(float);

    if (g_num_points != num_points || g_num_spaces != num_spaces) {
        solution_free();
        cudaMalloc(&d_points,        pts_bytes);
        cudaMalloc(&d_space_offsets, off_bytes);
        cudaMalloc(&d_results,       res_bytes);
        g_num_points = num_points;
        g_num_spaces = num_spaces;
    }

    cudaMemcpy(d_points,        points_xy,     pts_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_space_offsets, space_offsets, off_bytes, cudaMemcpyHostToDevice);

    int n_results = num_spaces * num_spaces;

    int init_block = 256;
    int init_grid  = (n_results + init_block - 1) / init_block;
    hausdorff_init_results<<<init_grid, init_block>>>(n_results, d_results);

    int threads_per_block = 64;
    int num_tiles = (num_points + threads_per_block - 1) / threads_per_block;
    kernel_hausdorff<<<num_tiles, threads_per_block>>>(
        num_points, d_points, num_spaces, d_space_offsets, d_results);

    cudaMemcpy(results, d_results,
               (size_t)n_results * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
