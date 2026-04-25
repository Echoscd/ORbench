// gpu_baseline.cu -- SPH force computation (GPU baseline)
//
// Faithfully ported from DualSPHysics JSphGpu_ker.cu
//   KerInteractionForcesFluidBox (simplified: Wendland kernel only,
//   no floating bodies, no periodic boundaries, no shifting)
//
// One thread per particle, iterates over 3x3x3 neighbor cells using
// precomputed cell_begin/cell_end arrays.
// Persistent GPU memory (cudaMalloc in init, reuse in compute).
//
// Reference: Crespo et al. "DualSPHysics: Open-source parallel CFD solver
// based on SPH", Computer Physics Communications, 2015.

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
#define SPH_GAMMA 7.0f
#define ALMOSTZERO 1e-18f

// ===== Module-level persistent state =====
static int    g_N;
static int    g_grid_nx, g_grid_ny, g_grid_nz;
static float  g_cell_size;

// Precomputed constants
static float  g_h, g_bwenh, g_kernelsize2;
static float  g_cs0, g_alpha_visc;
static float  g_cteb, g_ovrhopzero;

// Device arrays: inputs
static float* d_xs;
static float* d_ys;
static float* d_zs;
static float* d_vxs;
static float* d_vys;
static float* d_vzs;
static float* d_rhos;
static float* d_masses;
static int*   d_cell_begin;
static int*   d_cell_end;
static int*   d_sorted_idx;

// Device arrays: outputs
static float* d_ax;
static float* d_ay;
static float* d_az;
static float* d_drhodt;

// ===== Device function: Wendland C2 kernel gradient factor =====
// Ported from DualSPHysics FunSphKernel_iker.h GetKernelWendland_Fac
__device__ __forceinline__
float GetKernelWendland_Fac(float rr2, float h, float bwenh)
{
    float rad = sqrtf(rr2);
    float qq = rad / h;
    float wqq1 = 1.0f - 0.5f * qq;
    return bwenh * wqq1 * wqq1 * wqq1;
}

// ===== Device function: Tait equation of state (Monaghan 1994) =====
// Ported from DualSPHysics FunSphEos_iker.h ComputePressMonaghan
__device__ __forceinline__
float ComputePressMonaghan(float rho, float ovrhopzero, float cteb)
{
    return cteb * (powf(rho * ovrhopzero, SPH_GAMMA) - 1.0f);
}

// ===== Device function: InteractionForcesFluidBox =====
// Ported from DualSPHysics KerInteractionForcesFluidBox (simplified)
// Iterates over neighbor particles in one cell [pini, pfin), accumulating forces.
__device__ void InteractionForcesFluidBox(
    int p1, int pini, int pfin,
    float p1x, float p1y, float p1z,
    float p1vx, float p1vy, float p1vz,
    float p1rho, float pressp1, float massp1,
    float h, float bwenh, float kernelsize2,
    float cs0, float alpha_visc, float ovrhopzero, float cteb,
    const float* __restrict__ xs,
    const float* __restrict__ ys,
    const float* __restrict__ zs,
    const float* __restrict__ vxs,
    const float* __restrict__ vys,
    const float* __restrict__ vzs,
    const float* __restrict__ rhos,
    const float* __restrict__ masses,
    const int* __restrict__ sorted_idx,
    float& acex, float& acey, float& acez, float& arp1)
{
    for (int p2 = pini; p2 < pfin; p2++) {
        int idx2 = sorted_idx[p2];
        if (idx2 == p1) continue;

        float p2x = xs[idx2];
        float p2y = ys[idx2];
        float p2z = zs[idx2];

        // Distance computation (matches DualSPHysics drx/dry/drz)
        float drx = p1x - p2x;
        float dry = p1y - p2y;
        float drz = p1z - p2z;
        float rr2 = drx * drx + dry * dry + drz * drz;

        if (rr2 <= kernelsize2 && rr2 >= ALMOSTZERO) {
            // Compute kernel gradient factor
            float fac = GetKernelWendland_Fac(rr2, h, bwenh);
            float frx = fac * drx;
            float fry = fac * dry;
            float frz = fac * drz;

            float p2rho = rhos[idx2];
            float massp2 = masses[idx2];

            // Pressure force (momentum equation)
            float pressp2 = ComputePressMonaghan(p2rho, ovrhopzero, cteb);
            float prs = (pressp1 + pressp2) / (p1rho * p2rho);
            float p_vpm = -prs * massp2;
            acex += p_vpm * frx;
            acey += p_vpm * fry;
            acez += p_vpm * frz;

            // Artificial viscosity (Monaghan 1992)
            float dvx = p1vx - vxs[idx2];
            float dvy = p1vy - vys[idx2];
            float dvz = p1vz - vzs[idx2];
            float dot_rv = drx * dvx + dry * dvy + drz * dvz;

            if (dot_rv < 0.0f) {
                float rhobar = 0.5f * (p1rho + p2rho);
                float mu = h * dot_rv / (rr2 + 0.01f * h * h);
                float pi_visc = -alpha_visc * cs0 * mu / rhobar;
                float visc_force = -pi_visc * massp2;
                acex += visc_force * frx;
                acey += visc_force * fry;
                acez += visc_force * frz;
            }

            // Density derivative (continuity equation)
            arp1 += massp2 * (dvx * frx + dvy * fry + dvz * frz) * (p1rho / p2rho);
        }
    }
}

// ===== Main kernel: one thread per particle, 3x3x3 neighbor cell iteration =====
__launch_bounds__(256)
__global__ void InteractionForcesFluidKernel(
    int N,
    const float* __restrict__ xs,
    const float* __restrict__ ys,
    const float* __restrict__ zs,
    const float* __restrict__ vxs,
    const float* __restrict__ vys,
    const float* __restrict__ vzs,
    const float* __restrict__ rhos,
    const float* __restrict__ masses,
    const int* __restrict__ cell_begin,
    const int* __restrict__ cell_end,
    const int* __restrict__ sorted_idx,
    int grid_nx, int grid_ny, int grid_nz,
    float cell_size,
    float h, float bwenh, float kernelsize2,
    float cs0, float alpha_visc, float ovrhopzero, float cteb,
    float* __restrict__ ax,
    float* __restrict__ ay,
    float* __restrict__ az,
    float* __restrict__ drhodt)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned int)N) return;

    int num_cells_xy = grid_nx * grid_ny;

    // Load current particle data into registers (shared memory equivalent for single particle)
    float p1x = xs[i];
    float p1y = ys[i];
    float p1z = zs[i];
    float p1vx = vxs[i];
    float p1vy = vys[i];
    float p1vz = vzs[i];
    float p1rho = rhos[i];
    float massp1 = masses[i];

    float pressp1 = ComputePressMonaghan(p1rho, ovrhopzero, cteb);

    float acex = 0.0f, acey = 0.0f, acez = 0.0f;
    float arp1 = 0.0f;

    // Determine cell of particle i
    int cx = (int)floorf(p1x / cell_size);
    int cy = (int)floorf(p1y / cell_size);
    int cz = (int)floorf(p1z / cell_size);

    // Iterate over 3x3x3 neighbor cells
    for (int dcz = -1; dcz <= 1; dcz++) {
        int ncz = cz + dcz;
        if (ncz < 0 || ncz >= grid_nz) continue;
        for (int dcy = -1; dcy <= 1; dcy++) {
            int ncy = cy + dcy;
            if (ncy < 0 || ncy >= grid_ny) continue;
            for (int dcx = -1; dcx <= 1; dcx++) {
                int ncx = cx + dcx;
                if (ncx < 0 || ncx >= grid_nx) continue;

                int cell_id = ncx + ncy * grid_nx + ncz * num_cells_xy;
                int pini = cell_begin[cell_id];
                int pfin = cell_end[cell_id];
                if (pini < 0 || pfin < 0) continue;

                InteractionForcesFluidBox(
                    (int)i, pini, pfin,
                    p1x, p1y, p1z,
                    p1vx, p1vy, p1vz,
                    p1rho, pressp1, massp1,
                    h, bwenh, kernelsize2,
                    cs0, alpha_visc, ovrhopzero, cteb,
                    xs, ys, zs, vxs, vys, vzs, rhos, masses, sorted_idx,
                    acex, acey, acez, arp1);
            }
        }
    }

    ax[i] = acex;
    ay[i] = acey;
    az[i] = acez;
    drhodt[i] = arp1;
}

// ===== Public interface (extern "C" for task_io linkage) =====

extern "C" {

static int g_num_cells = 0;

void solution_free(void)
{
    if (d_xs)         { cudaFree(d_xs);         d_xs         = nullptr; }
    if (d_ys)         { cudaFree(d_ys);         d_ys         = nullptr; }
    if (d_zs)         { cudaFree(d_zs);         d_zs         = nullptr; }
    if (d_vxs)        { cudaFree(d_vxs);        d_vxs        = nullptr; }
    if (d_vys)        { cudaFree(d_vys);        d_vys        = nullptr; }
    if (d_vzs)        { cudaFree(d_vzs);        d_vzs        = nullptr; }
    if (d_rhos)       { cudaFree(d_rhos);       d_rhos       = nullptr; }
    if (d_masses)     { cudaFree(d_masses);     d_masses     = nullptr; }
    if (d_cell_begin) { cudaFree(d_cell_begin); d_cell_begin = nullptr; }
    if (d_cell_end)   { cudaFree(d_cell_end);   d_cell_end   = nullptr; }
    if (d_sorted_idx) { cudaFree(d_sorted_idx); d_sorted_idx = nullptr; }
    if (d_ax)         { cudaFree(d_ax);         d_ax         = nullptr; }
    if (d_ay)         { cudaFree(d_ay);         d_ay         = nullptr; }
    if (d_az)         { cudaFree(d_az);         d_az         = nullptr; }
    if (d_drhodt)     { cudaFree(d_drhodt);     d_drhodt     = nullptr; }
    g_N = 0;
    g_num_cells = 0;
}

void solution_compute(int N,
                      const float* xs, const float* ys, const float* zs,
                      const float* vxs, const float* vys, const float* vzs,
                      const float* rhos, const float* masses,
                      float h, float cs0, float rhop0, float alpha_visc,
                      const int* cell_begin, const int* cell_end,
                      const int* sorted_idx,
                      int grid_nx, int grid_ny, int grid_nz,
                      float cell_size,
                      float* ax, float* ay, float* az, float* drhodt)
{
    int num_cells = grid_nx * grid_ny * grid_nz;

    // Precompute Wendland kernel constants (3D)
    float awen = 495.0f / (32.0f * 3.14159265358979f * h * h * h);
    g_bwenh = -10.0f * awen / h;
    g_kernelsize2 = 4.0f * h * h;
    g_cteb = cs0 * cs0 * rhop0 / SPH_GAMMA;
    g_ovrhopzero = 1.0f / rhop0;

    g_grid_nx = grid_nx;
    g_grid_ny = grid_ny;
    g_grid_nz = grid_nz;
    g_cell_size = cell_size;
    g_h = h;
    g_cs0 = cs0;
    g_alpha_visc = alpha_visc;

    size_t sz_f = (size_t)N * sizeof(float);
    size_t sz_i_n = (size_t)N * sizeof(int);
    size_t sz_i_c = (size_t)num_cells * sizeof(int);

    if (g_N != N || g_num_cells != num_cells) {
        solution_free();
        cudaMalloc(&d_xs, sz_f);
        cudaMalloc(&d_ys, sz_f);
        cudaMalloc(&d_zs, sz_f);
        cudaMalloc(&d_vxs, sz_f);
        cudaMalloc(&d_vys, sz_f);
        cudaMalloc(&d_vzs, sz_f);
        cudaMalloc(&d_rhos, sz_f);
        cudaMalloc(&d_masses, sz_f);
        cudaMalloc(&d_cell_begin, sz_i_c);
        cudaMalloc(&d_cell_end, sz_i_c);
        cudaMalloc(&d_sorted_idx, sz_i_n);
        cudaMalloc(&d_ax, sz_f);
        cudaMalloc(&d_ay, sz_f);
        cudaMalloc(&d_az, sz_f);
        cudaMalloc(&d_drhodt, sz_f);
        g_N = N;
        g_num_cells = num_cells;
    }

    cudaMemcpy(d_xs, xs, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ys, ys, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_zs, zs, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vxs, vxs, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vys, vys, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vzs, vzs, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rhos, rhos, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_masses, masses, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cell_begin, cell_begin, sz_i_c, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cell_end, cell_end, sz_i_c, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sorted_idx, sorted_idx, sz_i_n, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    InteractionForcesFluidKernel<<<grid, block>>>(
        N,
        d_xs, d_ys, d_zs,
        d_vxs, d_vys, d_vzs,
        d_rhos, d_masses,
        d_cell_begin, d_cell_end, d_sorted_idx,
        grid_nx, grid_ny, grid_nz,
        cell_size,
        h, g_bwenh, g_kernelsize2,
        cs0, alpha_visc, g_ovrhopzero, g_cteb,
        d_ax, d_ay, d_az, d_drhodt);

    cudaMemcpy(ax,     d_ax,     sz_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(ay,     d_ay,     sz_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(az,     d_az,     sz_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(drhodt, d_drhodt, sz_f, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

} // extern "C"
