// cpu_reference.c -- N-body gravitational simulation CPU baseline
//
// Computes gravitational forces on all N particles via direct summation.
// F_i = sum_{j!=i} G * m_i * m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
// G = 1.0 (natural units).
//
// NO file I/O. All I/O handled by task_io_cpu.c.
//
// Build (via task_io + harness):
//   gcc -O2 -I framework/
//       framework/harness_cpu.c tasks/nbody_simulation/task_io_cpu.c
//       tasks/nbody_simulation/cpu_reference.c -o solution_cpu -lm

#include <stdlib.h>
#include <math.h>

// ===== Module-level state =====
static int g_N;
static float g_eps2;  // softening^2
static const float* g_px;
static const float* g_py;
static const float* g_pz;
static const float* g_mass;

void solution_init(int N, float softening,
                   const float* pos_x, const float* pos_y, const float* pos_z,
                   const float* mass) {
    g_N = N;
    g_eps2 = softening * softening;
    g_px = pos_x;
    g_py = pos_y;
    g_pz = pos_z;
    g_mass = mass;
}

void solution_compute(int N, float* fx, float* fy, float* fz) {
    for (int i = 0; i < N; i++) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        float xi = g_px[i], yi = g_py[i], zi = g_pz[i];
        float mi = g_mass[i];

        for (int j = 0; j < N; j++) {
            if (j == i) continue;

            float dx = g_px[j] - xi;
            float dy = g_py[j] - yi;
            float dz = g_pz[j] - zi;

            float dist2 = dx * dx + dy * dy + dz * dz + g_eps2;
            float inv_dist = 1.0f / sqrtf(dist2);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;

            float f = g_mass[j] * inv_dist3;  // G=1, m_i factored out then multiplied
            ax += f * dx;
            ay += f * dy;
            az += f * dz;
        }

        // F_i = m_i * acceleration
        fx[i] = mi * ax;
        fy[i] = mi * ay;
        fz[i] = mi * az;
    }
}
