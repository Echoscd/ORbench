// cpu_reference.c — thompson_sampling CPU baseline (compute_only interface)
//
// Monte Carlo simulation of Thompson Sampling for Bernoulli bandits.
// NO file I/O. All I/O handled by task_io_cpu.c.
//
// Build: gcc -O2 -DORBENCH_COMPUTE_ONLY -I framework/
//        framework/harness_cpu.c tasks/thompson_sampling/task_io_cpu.c
//        tasks/thompson_sampling/cpu_reference.c -o solution_cpu -lm

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define MAX_ARMS 1024

// ===== RNG: SplitMix64 for seeding, Xoshiro128** for generation =====

typedef struct {
    uint32_t s[4];
} rng_state_t;

static uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void rng_init(rng_state_t* rng, uint64_t seed, uint64_t stream) {
    uint64_t s = seed ^ (stream * 0x517cc1b727220a95ULL);
    uint64_t a = splitmix64(&s);
    uint64_t b = splitmix64(&s);
    rng->s[0] = (uint32_t)a;
    rng->s[1] = (uint32_t)(a >> 32);
    rng->s[2] = (uint32_t)b;
    rng->s[3] = (uint32_t)(b >> 32);
}

static uint32_t rotl32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

static uint32_t xoshiro128ss(rng_state_t* rng) {
    uint32_t result = rotl32(rng->s[1] * 5, 7) * 9;
    uint32_t t = rng->s[1] << 9;
    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3] = rotl32(rng->s[3], 11);
    return result;
}

static float rng_uniform(rng_state_t* rng) {
    return (float)(xoshiro128ss(rng) >> 8) * (1.0f / 16777216.0f);
}

static float rng_normal(rng_state_t* rng) {
    // Box-Muller transform
    float u1 = rng_uniform(rng);
    float u2 = rng_uniform(rng);
    while (u1 < 1e-30f) u1 = rng_uniform(rng);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

// Marsaglia-Tsang method for Gamma(alpha, 1) with alpha >= 1
static float rng_gamma(float alpha, rng_state_t* rng) {
    if (alpha < 1.0f) {
        // Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
        float g = rng_gamma(alpha + 1.0f, rng);
        return g * powf(rng_uniform(rng), 1.0f / alpha);
    }
    if (alpha == 1.0f) {
        // Gamma(1) = Exponential(1)
        float u = rng_uniform(rng);
        while (u < 1e-30f) u = rng_uniform(rng);
        return -logf(u);
    }
    // Marsaglia-Tsang for alpha >= 1
    float d = alpha - 1.0f / 3.0f;
    float c = 1.0f / sqrtf(9.0f * d);
    while (1) {
        float x, v;
        do {
            x = rng_normal(rng);
            v = 1.0f + c * x;
        } while (v <= 0.0f);
        v = v * v * v;
        float u = rng_uniform(rng);
        if (u < 1.0f - 0.0331f * (x * x) * (x * x))
            return d * v;
        if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v)))
            return d * v;
    }
}

static float rng_beta(float alpha, float beta, rng_state_t* rng) {
    float x = rng_gamma(alpha, rng);
    float y = rng_gamma(beta, rng);
    return x / (x + y);
}

// ===== Thompson Sampling simulation =====

void solution_compute(
    int N, int T, int M,
    const float* arm_means,
    uint64_t seed,
    float* avg_regret_out,
    float* avg_counts_out
) {
    // Find optimal arm
    float mu_star = arm_means[0];
    for (int i = 1; i < N; i++)
        if (arm_means[i] > mu_star) mu_star = arm_means[i];

    double total_regret = 0.0;
    double* total_counts = (double*)calloc(N, sizeof(double));

    for (int m = 0; m < M; m++) {
        rng_state_t rng;
        rng_init(&rng, seed, (uint64_t)m);

        int S[MAX_ARMS];
        int F[MAX_ARMS];
        memset(S, 0, N * sizeof(int));
        memset(F, 0, N * sizeof(int));
        double regret_m = 0.0;

        for (int t = 0; t < T; t++) {
            // Beta sampling + argmax
            int best_arm = 0;
            float best_theta = -1.0f;
            for (int i = 0; i < N; i++) {
                float theta = rng_beta((float)(S[i] + 1), (float)(F[i] + 1), &rng);
                if (theta > best_theta) {
                    best_theta = theta;
                    best_arm = i;
                }
            }

            // Bernoulli reward
            float u = rng_uniform(&rng);
            int reward = (u < arm_means[best_arm]) ? 1 : 0;

            S[best_arm] += reward;
            F[best_arm] += (1 - reward);
            regret_m += (double)(mu_star - arm_means[best_arm]);
        }

        total_regret += regret_m;
        for (int i = 0; i < N; i++)
            total_counts[i] += (double)(S[i] + F[i]);
    }

    *avg_regret_out = (float)(total_regret / M);
    for (int i = 0; i < N; i++)
        avg_counts_out[i] = (float)(total_counts[i] / M);

    free(total_counts);
}

void solution_free(void) {}
