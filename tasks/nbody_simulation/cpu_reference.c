// cpu_reference.c -- N-body CPU baseline (compute_only interface)
//
// Computes gravitational forces by direct summation.
// No file I/O; task_io_cpu.c handles all I/O.

#include <math.h>

void solution_compute(
    int N,
    float softening,
    const float* pos_x,
    const float* pos_y,
    const float* pos_z,
    const float* mass,
    float* fx,
    float* fy,
    float* fz
) {
    const float eps2 = softening * softening;

    for (int i = 0; i < N; i++) {
        const float xi = pos_x[i];
        const float yi = pos_y[i];
        const float zi = pos_z[i];
        const float mi = mass[i];

        float acc_x = 0.0f;
        float acc_y = 0.0f;
        float acc_z = 0.0f;

        for (int j = 0; j < N; j++) {
            if (j == i) continue;

            const float dx = pos_x[j] - xi;
            const float dy = pos_y[j] - yi;
            const float dz = pos_z[j] - zi;
            const float dist2 = dx * dx + dy * dy + dz * dz + eps2;
            const float inv_dist = 1.0f / sqrtf(dist2);
            const float inv_dist3 = inv_dist * inv_dist * inv_dist;
            const float coeff = mi * mass[j] * inv_dist3;

            acc_x += coeff * dx;
            acc_y += coeff * dy;
            acc_z += coeff * dz;
        }

        fx[i] = acc_x;
        fy[i] = acc_y;
        fz[i] = acc_z;
    }
}

void solution_free(void) {
    // no persistent CPU state
}
