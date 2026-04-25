// gpu_baseline.cu — PDHG LP Solver Iterations (GPU baseline)
//
// Faithfully derived from Google OR-Tools PDLP solver:
//   ortools/pdlp/primal_dual_hybrid_gradient.cc::TakeConstantSizeStep()
// The GPU baseline keeps the same PDHG iteration structure but offloads:
//   * SpMV (A*x and A^T*y) to CUDA kernels
//   * Primal/dual proximal projections to element-wise CUDA kernels
//   * Weighted-average update to CUDA kernel
//
// This is a baseline GPU implementation using custom SpMV kernels.
// An optimized version would use cuSPARSE csrmv/cscmv.

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ===== CUDA kernels =====

// SpMV: result = A^T * y (CSC natural direction)
// Matches TransposedMatrixVectorProduct() in sharder.cc:160-173.
// One thread per column of A.
__global__ void kernel_spmv_ATy(int num_vars,
                                const int*   col_ptrs,
                                const int*   row_indices,
                                const float* values,
                                const float* y,
                                float*       result)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_vars) return;
    float sum = 0.0f;
    for (int k = col_ptrs[j]; k < col_ptrs[j + 1]; k++) {
        sum += values[k] * y[row_indices[k]];
    }
    result[j] = sum;
}

// SpMV: result = A * x (CSC scattered, atomicAdd)
// Matches shard(TransposedConstraintMatrix).transpose() * x
__global__ void kernel_spmv_Ax(int num_vars,
                               const int*   col_ptrs,
                               const int*   row_indices,
                               const float* values,
                               const float* x,
                               float*       result)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_vars) return;
    float xj = x[j];
    for (int k = col_ptrs[j]; k < col_ptrs[j + 1]; k++) {
        atomicAdd(&result[row_indices[k]], values[k] * xj);
    }
}

// Zero a float array
__global__ void kernel_zero(int n, float* a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0.0f;
}

// Primal update + extrapolation (LP branch, lines 1867-1877)
// x_new = clip(x - τ*(c - dual_product), lb, ub)
// x_bar = 2*x_new - x_old
__global__ void kernel_primal_update(int num_vars,
                                     float primal_step_size,
                                     const float* obj,
                                     const float* dual_product,
                                     const float* var_lb,
                                     const float* var_ub,
                                     float* primal,
                                     float* x_bar)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_vars) return;
    float x_old = primal[j];
    float x_new = x_old - primal_step_size * (obj[j] - dual_product[j]);
    if (x_new > var_ub[j]) x_new = var_ub[j];
    if (x_new < var_lb[j]) x_new = var_lb[j];
    primal[j] = x_new;
    x_bar[j] = 2.0f * x_new - x_old;
}

// Dual update (lines 1912-1928)
// temp = y - σ * Ax_bar
// y_new = max(temp + σ*lb, min(0, temp + σ*ub))
__global__ void kernel_dual_update(int num_constraints,
                                   float dual_step_size,
                                   const float* Ax_bar,
                                   const float* con_lb,
                                   const float* con_ub,
                                   float* dual)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_constraints) return;
    float temp = dual[i] - dual_step_size * Ax_bar[i];
    float v_ub = temp + dual_step_size * con_ub[i];
    float v_lb = temp + dual_step_size * con_lb[i];
    float y_new = 0.0f;
    if (y_new > v_ub) y_new = v_ub;
    if (y_new < v_lb) y_new = v_lb;
    dual[i] = y_new;
}

// Weighted average update (M_14 algorithm, sharded_optimization_utils.cc:54-66)
__global__ void kernel_avg_update(int n, float ratio, const float* current,
                                  float* avg)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) avg[j] += ratio * (current[j] - avg[j]);
}

// ===== Persistent device state =====
static int    g_num_vars = 0;
static int    g_num_constraints = 0;
static int    g_num_iters = 0;
static float  g_step_size = 0;
static float  g_primal_weight = 0;

static float* d_obj = nullptr;
static float* d_var_lb = nullptr;
static float* d_var_ub = nullptr;
static float* d_con_lb = nullptr;
static float* d_con_ub = nullptr;
static int*   d_col_ptrs = nullptr;
static int*   d_row_indices = nullptr;
static float* d_values = nullptr;

static float* d_primal = nullptr;
static float* d_dual = nullptr;
static float* d_dual_product = nullptr;
static float* d_primal_avg = nullptr;
static float* d_x_bar = nullptr;
static float* d_Ax_bar = nullptr;

static int g_nnz = 0;

extern "C" void solution_free(void)
{
    if (d_obj)          { cudaFree(d_obj);          d_obj = nullptr; }
    if (d_var_lb)       { cudaFree(d_var_lb);       d_var_lb = nullptr; }
    if (d_var_ub)       { cudaFree(d_var_ub);       d_var_ub = nullptr; }
    if (d_con_lb)       { cudaFree(d_con_lb);       d_con_lb = nullptr; }
    if (d_con_ub)       { cudaFree(d_con_ub);       d_con_ub = nullptr; }
    if (d_col_ptrs)     { cudaFree(d_col_ptrs);     d_col_ptrs = nullptr; }
    if (d_row_indices)  { cudaFree(d_row_indices);  d_row_indices = nullptr; }
    if (d_values)       { cudaFree(d_values);       d_values = nullptr; }
    if (d_primal)       { cudaFree(d_primal);       d_primal = nullptr; }
    if (d_dual)         { cudaFree(d_dual);         d_dual = nullptr; }
    if (d_dual_product) { cudaFree(d_dual_product); d_dual_product = nullptr; }
    if (d_primal_avg)   { cudaFree(d_primal_avg);   d_primal_avg = nullptr; }
    if (d_x_bar)        { cudaFree(d_x_bar);        d_x_bar = nullptr; }
    if (d_Ax_bar)       { cudaFree(d_Ax_bar);       d_Ax_bar = nullptr; }
    g_num_vars = 0;
    g_num_constraints = 0;
    g_nnz = 0;
}

extern "C" void solution_compute(int          num_vars,
                                 int          num_constraints,
                                 int          nnz,
                                 int          num_iters,
                                 const float* obj,
                                 const float* var_lb,
                                 const float* var_ub,
                                 const float* con_lb,
                                 const float* con_ub,
                                 const int*   col_ptrs,
                                 const int*   row_indices,
                                 const float* values,
                                 float        step_size,
                                 float        primal_weight,
                                 float*       primal_out)
{
    g_num_iters       = num_iters;
    g_step_size       = step_size;
    g_primal_weight   = primal_weight;

    if (g_num_vars != num_vars || g_num_constraints != num_constraints || g_nnz != nnz) {
        solution_free();
        cudaMalloc(&d_obj,         num_vars * sizeof(float));
        cudaMalloc(&d_var_lb,      num_vars * sizeof(float));
        cudaMalloc(&d_var_ub,      num_vars * sizeof(float));
        cudaMalloc(&d_con_lb,      num_constraints * sizeof(float));
        cudaMalloc(&d_con_ub,      num_constraints * sizeof(float));
        cudaMalloc(&d_col_ptrs,    (num_vars + 1) * sizeof(int));
        cudaMalloc(&d_row_indices, nnz * sizeof(int));
        cudaMalloc(&d_values,      nnz * sizeof(float));
        cudaMalloc(&d_primal,       num_vars * sizeof(float));
        cudaMalloc(&d_dual,         num_constraints * sizeof(float));
        cudaMalloc(&d_dual_product, num_vars * sizeof(float));
        cudaMalloc(&d_primal_avg,   num_vars * sizeof(float));
        cudaMalloc(&d_x_bar,        num_vars * sizeof(float));
        cudaMalloc(&d_Ax_bar,       num_constraints * sizeof(float));
        g_num_vars = num_vars;
        g_num_constraints = num_constraints;
        g_nnz = nnz;
    }

    cudaMemcpy(d_obj,         obj,         num_vars * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_lb,      var_lb,      num_vars * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_ub,      var_ub,      num_vars * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_con_lb,      con_lb,      num_constraints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_con_ub,      con_ub,      num_constraints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ptrs,    col_ptrs,    (num_vars + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_indices, row_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values,      values,      nnz * sizeof(float), cudaMemcpyHostToDevice);

    int bv = 256;
    int gv = (num_vars + bv - 1) / bv;
    int gc = (num_constraints + bv - 1) / bv;

    // Zero all state (idempotent)
    cudaMemset(d_primal,       0, num_vars * sizeof(float));
    cudaMemset(d_dual,         0, num_constraints * sizeof(float));
    cudaMemset(d_dual_product, 0, num_vars * sizeof(float));
    cudaMemset(d_primal_avg,   0, num_vars * sizeof(float));

    float avg_weight_sum = 0.0f;
    float primal_step_size = step_size / primal_weight;
    float dual_step_size   = step_size * primal_weight;

    for (int iter = 0; iter < num_iters; iter++) {
        // 1. Primal update + extrapolation
        kernel_primal_update<<<gv, bv>>>(num_vars, primal_step_size,
            d_obj, d_dual_product, d_var_lb, d_var_ub, d_primal, d_x_bar);

        // 2. SpMV: A * x_bar → d_Ax_bar
        kernel_zero<<<gc, bv>>>(num_constraints, d_Ax_bar);
        kernel_spmv_Ax<<<gv, bv>>>(num_vars, d_col_ptrs, d_row_indices,
                                   d_values, d_x_bar, d_Ax_bar);

        // 3. Dual update
        kernel_dual_update<<<gc, bv>>>(num_constraints, dual_step_size,
            d_Ax_bar, d_con_lb, d_con_ub, d_dual);

        // 4. SpMV: A^T * y → d_dual_product
        kernel_spmv_ATy<<<gv, bv>>>(num_vars, d_col_ptrs, d_row_indices,
                                    d_values, d_dual, d_dual_product);

        // 5. Weighted average update
        float ratio = step_size / (avg_weight_sum + step_size);
        kernel_avg_update<<<gv, bv>>>(num_vars, ratio, d_primal, d_primal_avg);
        avg_weight_sum += step_size;
    }

    cudaMemcpy(primal_out, d_primal_avg, num_vars * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
