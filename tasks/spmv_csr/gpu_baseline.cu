// gpu_baseline.cu — Sparse Matrix-Vector Multiply: answer = A^T * vector
//
// Faithfully ported from Google OR-Tools PDLP:
//   ortools/pdlp/sharder.cc  lines 160–173
//   TransposedMatrixVectorProduct(matrix, vector, sharder)
//
// The original shards columns across CPU threads; each shard computes
//   shard(answer) = shard(matrix).transpose() * vector
// i.e. for each column j in the shard:
//   answer[j] = dot(column_j, vector) = sum values[k] * vector[row_indices[k]]
//
// GPU port: one CUDA thread per column (= one output element), directly
// matching the per-column decomposition of the Sharder. Each thread gathers
// from the dense vector using the CSC column's row indices — identical to the
// Eigen expression `shard(matrix).transpose() * vector`.
//
// CSC (ColMajor) storage:
//   col_ptrs[j]..col_ptrs[j+1]: nonzeros in column j
//   row_indices[k]: row index of nonzero k
//   values[k]: value of nonzero k

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ===== Kernel: one thread per column, column-gather dot product =====
// Directly maps to: shard(answer) = shard(matrix).transpose() * vector
__global__ void TransposedMatrixVectorProductKernel(
    int          num_cols,
    const int*   __restrict__ col_ptrs,
    const int*   __restrict__ row_indices,
    const float* __restrict__ values,
    const float* __restrict__ vector,
    float*       __restrict__ answer)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_cols) return;

    float sum = 0.0f;
    int start = col_ptrs[j];
    int end   = col_ptrs[j + 1];
    for (int k = start; k < end; k++) {
        sum += values[k] * __ldg(&vector[row_indices[k]]);
    }
    answer[j] = sum;
}

// ===== Persistent device state =====
static int    g_num_rows = 0;
static int    g_num_cols = 0;
static int    g_nnz = 0;
static int*   d_col_ptrs    = nullptr;
static int*   d_row_indices = nullptr;
static float* d_values      = nullptr;
static float* d_vector      = nullptr;
static float* d_answer      = nullptr;

extern "C" void solution_free(void)
{
    if (d_col_ptrs)    { cudaFree(d_col_ptrs);    d_col_ptrs    = nullptr; }
    if (d_row_indices) { cudaFree(d_row_indices); d_row_indices = nullptr; }
    if (d_values)      { cudaFree(d_values);      d_values      = nullptr; }
    if (d_vector)      { cudaFree(d_vector);      d_vector      = nullptr; }
    if (d_answer)      { cudaFree(d_answer);      d_answer      = nullptr; }
    g_num_rows = 0;
    g_num_cols = 0;
    g_nnz = 0;
}

extern "C" void solution_compute(int          num_rows,
                                 int          num_cols,
                                 const int*   col_ptrs,
                                 const int*   row_indices,
                                 const float* values,
                                 const float* vector,
                                 float*       answer)
{
    int nnz = col_ptrs[num_cols];

    if (g_num_rows != num_rows || g_num_cols != num_cols || g_nnz != nnz) {
        solution_free();
        cudaMalloc(&d_col_ptrs,    (num_cols + 1) * sizeof(int));
        cudaMalloc(&d_row_indices, nnz * sizeof(int));
        cudaMalloc(&d_values,      nnz * sizeof(float));
        cudaMalloc(&d_vector,      num_rows * sizeof(float));
        cudaMalloc(&d_answer,      num_cols * sizeof(float));
        g_num_rows = num_rows;
        g_num_cols = num_cols;
        g_nnz = nnz;
    }

    cudaMemcpy(d_col_ptrs,    col_ptrs,    (num_cols + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_indices, row_indices, nnz * sizeof(int),            cudaMemcpyHostToDevice);
    cudaMemcpy(d_values,      values,      nnz * sizeof(float),          cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector,      vector,      num_rows * sizeof(float),     cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (num_cols + threadsPerBlock - 1) / threadsPerBlock;

    TransposedMatrixVectorProductKernel<<<blocks, threadsPerBlock>>>(
        num_cols, d_col_ptrs, d_row_indices, d_values, d_vector, d_answer);

    cudaMemcpy(answer, d_answer, num_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
