// cpu_reference.c -- asian_option_pricing_mc CPU baseline
//
// Computes arithmetic-average Asian option prices under a geometric Brownian motion
// model using Monte Carlo simulation with pre-generated standard-normal shocks.

#include <math.h>
#include <stdlib.h>

static inline float maxf_local(float a, float b) {
    return (a > b) ? a : b;
}

void solution_compute(
    int N,
    int num_paths,
    int num_steps,
    const float* s0,
    const float* strike,
    const float* rate,
    const float* sigma,
    const float* maturity,
    const int* option_type,
    const float* shocks,
    float* prices
) {
    for (int i = 0; i < N; i++) {
        float S0 = s0[i];
        float K = strike[i];
        float r = rate[i];
        float vol = sigma[i];
        float T = maturity[i];
        int is_put = option_type[i];

        float dt = T / (float)num_steps;
        float drift = (r - 0.5f * vol * vol) * dt;
        float vol_term = vol * sqrtf(dt);
        float discount = expf(-r * T);

        double payoff_sum = 0.0;
        for (int p = 0; p < num_paths; p++) {
            float S = S0;
            double path_sum = 0.0;
            const float* z = shocks + (size_t)p * (size_t)num_steps;
            for (int s = 0; s < num_steps; s++) {
                S *= expf(drift + vol_term * z[s]);
                path_sum += (double)S;
            }
            float avg_S = (float)(path_sum / (double)num_steps);
            float payoff = is_put ? maxf_local(K - avg_S, 0.0f) : maxf_local(avg_S - K, 0.0f);
            payoff_sum += (double)payoff;
        }

        prices[i] = discount * (float)(payoff_sum / (double)num_paths);
    }
}

void solution_free(void) {
    // No persistent state.
}
