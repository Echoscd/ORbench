// cpu_reference.c -- MRI-Q CPU baseline / reference implementation
// Pure C implementation, no external dependencies.

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int g_num_k = 0;
static int g_num_x = 0;
static const float* g_kx = NULL;
static const float* g_ky = NULL;
static const float* g_kz = NULL;
static const float* g_phi_r = NULL;
static const float* g_phi_i = NULL;
static const float* g_x = NULL;
static const float* g_y = NULL;
static const float* g_z = NULL;

void solution_init(
    int num_k,
    int num_x,
    const float* h_kx,
    const float* h_ky,
    const float* h_kz,
    const float* h_phi_r,
    const float* h_phi_i,
    const float* h_x,
    const float* h_y,
    const float* h_z
) {
    g_num_k = num_k;
    g_num_x = num_x;
    g_kx = h_kx;
    g_ky = h_ky;
    g_kz = h_kz;
    g_phi_r = h_phi_r;
    g_phi_i = h_phi_i;
    g_x = h_x;
    g_y = h_y;
    g_z = h_z;
}

void solution_compute(float* h_qr, float* h_qi) {
    const float two_pi = (float)(2.0 * M_PI);
    for (int i = 0; i < g_num_x; i++) {
        float xi = g_x[i];
        float yi = g_y[i];
        float zi = g_z[i];
        double qr = 0.0;
        double qi = 0.0;
        for (int k = 0; k < g_num_k; k++) {
            float arg = two_pi * (g_kx[k] * xi + g_ky[k] * yi + g_kz[k] * zi);
            float c = cosf(arg);
            float s = sinf(arg);
            float pr = g_phi_r[k];
            float pi = g_phi_i[k];
            qr += (double)(pr * c - pi * s);
            qi += (double)(pr * s + pi * c);
        }
        h_qr[i] = (float)qr;
        h_qi[i] = (float)qi;
    }
}

void solution_free(void) {
    g_num_k = 0;
    g_num_x = 0;
    g_kx = g_ky = g_kz = NULL;
    g_phi_r = g_phi_i = NULL;
    g_x = g_y = g_z = NULL;
}
