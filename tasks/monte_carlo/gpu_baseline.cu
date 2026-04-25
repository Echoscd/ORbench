// gpu_baseline.cu — Monte Carlo option pricing (GPU baseline)
//
// Faithfully ported from FinanceBench/Monte-Carlo/CUDA/monteCarloKernels.cu
// Preserves original function names, structure, and computation flow.
//
// Key changes from original:
//   - Uses xorshift32 RNG instead of curand for determinism (matches cpu_reference.c)
//   - Uses ORBench solution_init/solution_compute/solution_free interface
//   - Persistent GPU memory allocation

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

// ===== Constants from monteCarloConstants.h =====
#define DEFAULT_SEQ_VAL    1.0f
#define DEFAULT_SEQ_WEIGHT 1.0f
#define START_PATH_VAL     1.0f

// ===== Inverse normal distribution coefficients (from monteCarloKernels.cu) =====
#define A_1 -39.696830286653757
#define A_2  220.94609842452050
#define A_3 -275.92851044696869
#define A_4  138.35775186726900
#define A_5 -30.664798066147160
#define A_6  2.5066282774592392
#define B_1 -54.476098798224058
#define B_2  161.58583685804089
#define B_3 -155.69897985988661
#define B_4  66.801311887719720
#define B_5 -13.280681552885721

typedef float dataType;

// ===== Option struct (from monteCarloStructs.h) =====
struct monteCarloOptionStruct {
    dataType riskVal;
    dataType divVal;
    dataType voltVal;
    dataType underlyingVal;
    dataType strikeVal;
    dataType discountVal;
};

// ===== Device functions — names match original monteCarloKernels.cu =====

__device__ unsigned int xorshift32_dev(unsigned int state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// function to compute the inverse normal distribution
__device__ dataType compInverseNormDist(dataType x)
{
    dataType z;
    z = x - 0.5;
    dataType r = z * z;
    z = (((((A_1*r+A_2)*r+A_3)*r+A_4)*r+A_5)*r+A_6)*z /
                (((((B_1*r+B_2)*r+B_3)*r+B_4)*r+B_5)*r+1.0);
    return z;
}

__device__ dataType interestRateCompoundFact(dataType t, dataType rate)
{
    // assuming "continuous" option
    return exp(rate * t);
}

__device__ dataType interestRateDiscountFact(dataType t, dataType rate)
{
    return 1.0 / interestRateCompoundFact(t, rate);
}

__device__ dataType flatForwardDiscountImpl(dataType t, dataType rate)
{
    return interestRateDiscountFact(t, rate);
}

__device__ dataType yieldTermStructDiscount(dataType t, dataType rate)
{
    return flatForwardDiscountImpl(t, rate);
}

__device__ dataType interestRateImpliedRate(dataType compound, dataType t)
{
    dataType r = log(compound) / t;
    return r;
}

__device__ dataType yieldTermStructForwardRate(dataType t1, dataType t2, dataType rate)
{
    dataType compound = interestRateDiscountFact(t1, rate) / interestRateDiscountFact(t2, rate);
    return interestRateImpliedRate(compound, t2 - t1);
}

__device__ dataType localVoltLocVol(dataType t, dataType underlyingLevel, monteCarloOptionStruct optionStruct)
{
    return optionStruct.voltVal;
}

__device__ dataType processDiff(dataType t, dataType x, monteCarloOptionStruct optionStruct)
{
    return localVoltLocVol(t, x, optionStruct);
}

__device__ dataType processDrift(dataType t, dataType x, monteCarloOptionStruct optionStruct)
{
    dataType sigma = processDiff(t, x, optionStruct);
    dataType t1 = t + 0.0001;

    return yieldTermStructForwardRate(t, t1, optionStruct.riskVal)
         - yieldTermStructForwardRate(t, t1, optionStruct.divVal)
         - (0.5 * sigma * sigma);
}

__device__ dataType discDiff(dataType t0, dataType x0, dataType dt, monteCarloOptionStruct optionStruct)
{
    return processDiff(t0, x0, optionStruct) * sqrt(dt);
}

__device__ dataType stdDeviation(dataType t0, dataType x0, dataType dt, monteCarloOptionStruct optionStruct)
{
    return discDiff(t0, x0, dt, optionStruct);
}

__device__ dataType apply_fn(dataType x0, dataType dx)
{
    return (x0 * exp(dx));
}

__device__ dataType discDrift(dataType t0, dataType x0, dataType dt, monteCarloOptionStruct optionStruct)
{
    return processDrift(t0, x0, optionStruct) * dt;
}

__device__ dataType processEvolve(dataType t0, dataType x0, dataType dt, dataType dw, monteCarloOptionStruct optionStruct)
{
    return apply_fn(x0, discDrift(t0, x0, dt, optionStruct) + stdDeviation(t0, x0, dt, optionStruct) * dw);
}

__device__ dataType getProcessValX0(monteCarloOptionStruct optionStruct)
{
    return optionStruct.underlyingVal;
}

__device__ dataType getPrice(dataType val, dataType strikeVal, dataType discountVal)
{
    dataType diff = strikeVal - val;
    return (diff > 0.0 ? diff : 0.0) * discountVal;
}

// ===== Main kernel — matches structure of original monteCarloGpuKernel =====
__launch_bounds__(256)
__global__ void monteCarloGpuKernel(
    dataType* __restrict__ samplePrices,
    int numSamples,
    dataType dt,
    monteCarloOptionStruct optionStruct,
    int seqLen,
    unsigned int baseSeed
) {
    // retrieve the thread number
    size_t numThread = blockIdx.x * blockDim.x + threadIdx.x;

    // retrieve the number of sample
    int numSample = numThread;

    if (numSample < numSamples)
    {
        // Path simulation in registers (no shared memory needed)
        dataType pathVal = getProcessValX0(optionStruct);

        // Deterministic per-sample seed (same formula as cpu_reference.c)
        unsigned int rngState = baseSeed ^ ((unsigned int)numSample * 2654435761u);
        if (rngState == 0) rngState = 1;

        for (int i = 1; i < seqLen; i++)
        {
            dataType t = i * dt;
            // Generate uniform random in (0, 1) via xorshift32
            rngState = xorshift32_dev(rngState);
            dataType randVal = (dataType)(rngState) / 4294967296.0;
            if (randVal < 0.0001) randVal = 0.0001;
            if (randVal > 0.9999) randVal = 0.9999;
            dataType inverseCumRandVal = compInverseNormDist(randVal);
            pathVal = processEvolve(t, pathVal, dt, inverseCumRandVal, optionStruct);
        }

        dataType price = getPrice(pathVal, optionStruct.strikeVal, optionStruct.discountVal);
        samplePrices[numSample] = price;
    }
}

// ===== Host interface =====

static float*  d_samplePrices = nullptr;
static int     g_N = 0;
static int     g_seqLen = 0;
static float   g_dt = 0.0f;
static unsigned int g_baseSeed = 0;
static monteCarloOptionStruct g_optionStruct;

extern "C" void solution_free(void)
{
    if (d_samplePrices) { cudaFree(d_samplePrices); d_samplePrices = nullptr; }
    g_N = 0;
}

extern "C" void solution_compute(int N, int num_steps, float risk_free, float volatility,
                                  float strike, float spot, float time_to_maturity,
                                  unsigned int base_seed,
                                  float* samplePrices)
{
    g_seqLen = num_steps;
    g_dt = 1.0f / (float)num_steps;
    g_baseSeed = base_seed;

    g_optionStruct.riskVal = risk_free;
    g_optionStruct.divVal = 0.0f;
    g_optionStruct.voltVal = volatility;
    g_optionStruct.underlyingVal = spot;
    g_optionStruct.strikeVal = strike;
    g_optionStruct.discountVal = expf(-risk_free * time_to_maturity);

    if (g_N != N) {
        solution_free();
        cudaMalloc(&d_samplePrices, N * sizeof(float));
        g_N = N;
    }

    // Launch: one thread per sample
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    monteCarloGpuKernel<<<numBlocks, threadsPerBlock>>>(
        d_samplePrices, N, g_dt, g_optionStruct, g_seqLen, g_baseSeed
    );

    // Download results
    cudaMemcpy(samplePrices, d_samplePrices, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
