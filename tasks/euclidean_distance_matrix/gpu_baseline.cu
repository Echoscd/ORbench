// gpu_baseline.cu — Pairwise squared Euclidean distance matrix (GPU baseline)
//
// Verbatim port of kNN-CUDA (vincentfpgarcia/kNN-CUDA, code/knncuda.cu) —
// the global-memory shared-memory tiled `compute_distances` kernel.
// All shared-memory layout, indexing arithmetic (begin_A/begin_B/step_A/
// step_B/end_A), tiling, and output writes are bit-for-bit identical to the
// original CUDA source. Only adjustment: pitches `ref_pitch` and
// `query_pitch` are passed equal to `ref_nb` and `query_nb` (no cuMemAlloc2D
// padding) since our input sizes are powers of two.
//
// Reference: Garcia et al., ICIP 2010.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 16

// ===== Verbatim from knncuda.cu::compute_distances (lines 20-91) =====
__global__ void compute_distances(float * ref,
                                  int     ref_width,
                                  int     ref_pitch,
                                  float * query,
                                  int     query_width,
                                  int     query_pitch,
                                  int     height,
                                  float * dist) {

    // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
    __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

    // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
    __shared__ int begin_A;
    __shared__ int begin_B;
    __shared__ int step_A;
    __shared__ int step_B;
    __shared__ int end_A;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Initializarion of the SSD for the current thread
    float ssd = 0.f;

    // Loop parameters
    begin_A = BLOCK_DIM * blockIdx.y;
    begin_B = BLOCK_DIM * blockIdx.x;
    step_A  = BLOCK_DIM * ref_pitch;
    step_B  = BLOCK_DIM * query_pitch;
    end_A   = begin_A + (height-1) * ref_pitch;

    // Conditions
    int cond0 = (begin_A + tx < ref_width); // used to write in shared memory
    int cond1 = (begin_B + tx < query_width); // used to write in shared memory & to computations and to write in output array
    int cond2 = (begin_A + ty < ref_width); // used to computations and to write in output matrix

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        if (a/ref_pitch + ty < height) {
            shared_A[ty][tx] = (cond0)? ref[a + ref_pitch * ty + tx] : 0;
            shared_B[ty][tx] = (cond1)? query[b + query_pitch * ty + tx] : 0;
        }
        else {
            shared_A[ty][tx] = 0;
            shared_B[ty][tx] = 0;
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
        if (cond2 && cond1) {
            for (int k = 0; k < BLOCK_DIM; ++k){
                float tmp = shared_A[k][ty] - shared_B[k][tx];
                ssd += tmp*tmp;
            }
        }

        // Synchronize to make sure that the preceeding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    if (cond2 && cond1) {
        dist[ (begin_A + ty) * query_pitch + begin_B + tx ] = ssd;
    }
}

// ===== Persistent device state =====
static int    g_ref_nb   = 0;
static int    g_query_nb = 0;
static int    g_dim      = 0;
static float* d_ref      = nullptr;
static float* d_query    = nullptr;
static float* d_dist     = nullptr;

extern "C" void solution_free(void)
{
    if (d_ref)   { cudaFree(d_ref);   d_ref   = nullptr; }
    if (d_query) { cudaFree(d_query); d_query = nullptr; }
    if (d_dist)  { cudaFree(d_dist);  d_dist  = nullptr; }
    g_ref_nb = 0;
    g_query_nb = 0;
    g_dim = 0;
}

extern "C" void solution_compute(int          ref_nb,
                                 int          query_nb,
                                 int          dim,
                                 const float* ref,
                                 const float* query,
                                 float*       dist)
{
    size_t ref_bytes   = (size_t)dim * ref_nb   * sizeof(float);
    size_t query_bytes = (size_t)dim * query_nb * sizeof(float);
    size_t dist_bytes  = (size_t)ref_nb * query_nb * sizeof(float);

    if (g_ref_nb != ref_nb || g_query_nb != query_nb || g_dim != dim) {
        solution_free();
        cudaMalloc(&d_ref,   ref_bytes);
        cudaMalloc(&d_query, query_bytes);
        cudaMalloc(&d_dist,  dist_bytes);
        g_ref_nb = ref_nb;
        g_query_nb = query_nb;
        g_dim = dim;
    }

    cudaMemcpy(d_ref,   ref,   ref_bytes,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, query_bytes, cudaMemcpyHostToDevice);

    // Launch matches kNN-CUDA's knn_cuda_global() (lines around 540) —
    //   block = (BLOCK_DIM, BLOCK_DIM)
    //   grid  = (ceil(query_nb/BLOCK_DIM), ceil(ref_nb/BLOCK_DIM))
    // pitches equal nb (no padding for power-of-two sizes).
    dim3 block(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid ((query_nb + BLOCK_DIM - 1) / BLOCK_DIM,
               (ref_nb   + BLOCK_DIM - 1) / BLOCK_DIM,
               1);

    compute_distances<<<grid, block>>>(d_ref, ref_nb, ref_nb,
                                       d_query, query_nb, query_nb,
                                       dim, d_dist);

    cudaMemcpy(dist, d_dist,
               (size_t)ref_nb * query_nb * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
