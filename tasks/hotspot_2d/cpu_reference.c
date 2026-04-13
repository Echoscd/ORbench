#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_PD 3.0f
#define MAX_PD_SCALE 1000000.0f
#define PRECISION 0.001f
#define SPEC_HEAT_SI 1.75e6f
#define K_SI 100.0f
#define FACTOR_CHIP 0.5f

static int g_rows = 0;
static int g_cols = 0;
static int g_iterations = 0;
static int g_numel = 0;
static float* g_temp_init = NULL;
static float* g_power = NULL;
static float* g_buf_a = NULL;
static float* g_buf_b = NULL;

static const float g_t_chip = 0.0005f;
static const float g_chip_height = 0.016f;
static const float g_chip_width = 0.016f;
static const float g_amb_temp = 80.0f;

static void free_all(void) {
    if (g_temp_init) { free(g_temp_init); g_temp_init = NULL; }
    if (g_power) { free(g_power); g_power = NULL; }
    if (g_buf_a) { free(g_buf_a); g_buf_a = NULL; }
    if (g_buf_b) { free(g_buf_b); g_buf_b = NULL; }
    g_rows = g_cols = g_iterations = g_numel = 0;
}

void solution_init(
    int rows,
    int cols,
    int iterations,
    const float* h_temp_init,
    const float* h_power
) {
    int numel;
    free_all();
    if (rows <= 0 || cols <= 0 || iterations < 0) {
        fprintf(stderr, "invalid hotspot dimensions\n");
        return;
    }
    g_rows = rows;
    g_cols = cols;
    g_iterations = iterations;
    numel = rows * cols;
    g_numel = numel;

    g_temp_init = (float*)malloc((size_t)numel * sizeof(float));
    g_power = (float*)malloc((size_t)numel * sizeof(float));
    g_buf_a = (float*)malloc((size_t)numel * sizeof(float));
    g_buf_b = (float*)malloc((size_t)numel * sizeof(float));
    if (!g_temp_init || !g_power || !g_buf_a || !g_buf_b) {
        fprintf(stderr, "allocation failed in solution_init\n");
        free_all();
        return;
    }
    memcpy(g_temp_init, h_temp_init, (size_t)numel * sizeof(float));
    memcpy(g_power, h_power, (size_t)numel * sizeof(float));
}

static void run_hotspot(float* cur, float* nxt) {
    int rows = g_rows;
    int cols = g_cols;
    float grid_height = g_chip_height / (float)rows;
    float grid_width  = g_chip_width / (float)cols;
    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * g_t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0f * K_SI * g_t_chip * grid_height);
    float Ry = grid_height / (2.0f * K_SI * g_t_chip * grid_width);
    float Rz = g_t_chip / (K_SI * grid_height * grid_width);
    float max_slope = (MAX_PD * MAX_PD_SCALE) / (FACTOR_CHIP * g_t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;
    int it, r, c;

    for (it = 0; it < g_iterations; ++it) {
        float* src = (it % 2 == 0) ? cur : nxt;
        float* dst = (it % 2 == 0) ? nxt : cur;

        for (r = 1; r < rows - 1; ++r) {
            for (c = 1; c < cols - 1; ++c) {
                int idx = r * cols + c;
                float center = src[idx];
                float delta = (step / Cap) * (
                    g_power[idx] +
                    (src[idx + cols] + src[idx - cols] - 2.0f * center) / Ry +
                    (src[idx + 1] + src[idx - 1] - 2.0f * center) / Rx +
                    (g_amb_temp - center) / Rz
                );
                dst[idx] = center + delta;
            }
        }

        {
            int idx = 0;
            float center = src[idx];
            float delta = (step / Cap) * (
                g_power[idx] +
                (src[1] - center) / Rx +
                (src[cols] - center) / Ry +
                (g_amb_temp - center) / Rz
            );
            dst[idx] = center + delta;
        }
        {
            int idx = cols - 1;
            float center = src[idx];
            float delta = (step / Cap) * (
                g_power[idx] +
                (src[idx - 1] - center) / Rx +
                (src[idx + cols] - center) / Ry +
                (g_amb_temp - center) / Rz
            );
            dst[idx] = center + delta;
        }
        {
            int idx = (rows - 1) * cols + (cols - 1);
            float center = src[idx];
            float delta = (step / Cap) * (
                g_power[idx] +
                (src[idx - 1] - center) / Rx +
                (src[idx - cols] - center) / Ry +
                (g_amb_temp - center) / Rz
            );
            dst[idx] = center + delta;
        }
        {
            int idx = (rows - 1) * cols;
            float center = src[idx];
            float delta = (step / Cap) * (
                g_power[idx] +
                (src[idx + 1] - center) / Rx +
                (src[idx - cols] - center) / Ry +
                (g_amb_temp - center) / Rz
            );
            dst[idx] = center + delta;
        }

        for (c = 1; c < cols - 1; ++c) {
            int top = c;
            int bot = (rows - 1) * cols + c;
            float center_top = src[top];
            float center_bot = src[bot];
            float delta_top = (step / Cap) * (
                g_power[top] +
                (src[top + 1] + src[top - 1] - 2.0f * center_top) / Rx +
                (src[top + cols] - center_top) / Ry +
                (g_amb_temp - center_top) / Rz
            );
            float delta_bot = (step / Cap) * (
                g_power[bot] +
                (src[bot + 1] + src[bot - 1] - 2.0f * center_bot) / Rx +
                (src[bot - cols] - center_bot) / Ry +
                (g_amb_temp - center_bot) / Rz
            );
            dst[top] = center_top + delta_top;
            dst[bot] = center_bot + delta_bot;
        }

        for (r = 1; r < rows - 1; ++r) {
            int left = r * cols;
            int right = r * cols + (cols - 1);
            float center_left = src[left];
            float center_right = src[right];
            float delta_right = (step / Cap) * (
                g_power[right] +
                (src[right + cols] + src[right - cols] - 2.0f * center_right) / Ry +
                (src[right - 1] - center_right) / Rx +
                (g_amb_temp - center_right) / Rz
            );
            float delta_left = (step / Cap) * (
                g_power[left] +
                (src[left + cols] + src[left - cols] - 2.0f * center_left) / Ry +
                (src[left + 1] - center_left) / Rx +
                (g_amb_temp - center_left) / Rz
            );
            dst[right] = center_right + delta_right;
            dst[left] = center_left + delta_left;
        }
    }
}

void solution_compute(int output_stride, float* h_sampled_out) {
    int k;
    float* final_grid;
    if (!g_temp_init || !g_power || !g_buf_a || !g_buf_b) {
        fprintf(stderr, "solution_compute called before successful init\n");
        return;
    }
    memcpy(g_buf_a, g_temp_init, (size_t)g_numel * sizeof(float));
    memset(g_buf_b, 0, (size_t)g_numel * sizeof(float));

    if (g_iterations == 0) {
        final_grid = g_buf_a;
    } else {
        run_hotspot(g_buf_a, g_buf_b);
        final_grid = (g_iterations % 2 == 0) ? g_buf_a : g_buf_b;
    }

    for (k = 0; k * output_stride < g_numel; ++k) {
        h_sampled_out[k] = final_grid[k * output_stride];
    }
}

void solution_free(void) {
    free_all();
}
