// gpu_baseline.cu — Black-Scholes option pricing (GPU baseline)
//
// Faithfully ported from FinanceBench/Black-Scholes/CUDA/
//   blackScholesAnalyticEngineKernels.cu
//   blackScholesAnalyticEngineStructs.cuh
//   errorFunctConsts.cuh
//
// Preserves original __device__ function names (without Cpu suffix, matching
// the CUDA version), struct definitions, and computation flow.
// Optimization: uses erff() built-in instead of the manual polynomial for
// the error function — this maps to hardware on sm_80+.

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

// ===== Option type constants =====
#define CALL 0
#define PUT  1

// ===== Structs from blackScholesAnalyticEngineStructs.cuh =====

struct yieldTermStruct {
    float timeYearFraction;
    float forward;
    float compounding;
    float frequency;
    float intRate;
};

struct blackVolStruct {
    float timeYearFraction;
    float following;
    float volatility;
};

struct blackScholesMertStruct {
    float x0;
    yieldTermStruct dividendTS;
    yieldTermStruct riskFreeTS;
    blackVolStruct blackVolTS;
};

struct payoffStruct {
    int type;
    float strike;
};

struct optionStruct {
    payoffStruct payoff;
    float yearFractionTime;
    blackScholesMertStruct pricingEngine;
};

struct blackCalcStruct {
    float strike;
    float forward;
    float stdDev;
    float discount;
    float variance;
    float d1;
    float d2;
    float alpha;
    float beta;
    float DalphaDd1;
    float DbetaDd2;
    float n_d1;
    float cum_d1;
    float n_d2;
    float cum_d2;
    float x;
    float DxDs;
    float DxDstrike;
};

struct normalDistStruct {
    float average;
    float sigma;
    float denominator;
    float derNormalizationFactor;
    float normalizationFactor;
};

struct optionInputStruct {
    int type;
    float strike;
    float spot;
    float q;
    float r;
    float t;
    float vol;
    float value;
    float tol;
};

// ===== __device__ functions — names match original CUDA version =====

__device__ float interestRateCompoundFactor(float t, yieldTermStruct currYieldTermStruct)
{
    return (expf((currYieldTermStruct.forward) * t));
}

__device__ float interestRateDiscountFactor(float t, yieldTermStruct currYieldTermStruct)
{
    return 1.0f / interestRateCompoundFactor(t, currYieldTermStruct);
}

__device__ float getBlackVolBlackVar(blackVolStruct volTS)
{
    float vol = volTS.volatility;
    return vol * vol * volTS.timeYearFraction;
}

__device__ float getDiscountOnDividendYield(float yearFraction, yieldTermStruct dividendYieldTermStruct)
{
    float intDiscountFactor = interestRateDiscountFactor(yearFraction, dividendYieldTermStruct);
    return intDiscountFactor;
}

__device__ float getDiscountOnRiskFreeRate(float yearFraction, yieldTermStruct riskFreeRateYieldTermStruct)
{
    return interestRateDiscountFactor(yearFraction, riskFreeRateYieldTermStruct);
}

// GPU optimization: use erff() built-in instead of manual polynomial
// erff() maps to hardware instructions on sm_80+
__device__ float cumNormDistOp(float z)
{
    return 0.5f * (1.0f + erff(z * 0.7071067811865475f));  // z / sqrt(2)
}

__device__ float gaussianFunctNormDist(float x)
{
    float exponent = -(x * x) / 2.0f;
    return exponent <= -690.0f ? 0.0f :
        (0.7978845608028654f) * expf(exponent);  // 1/sqrt(2*pi) * normFactor
}

__device__ float cumNormDistDeriv(float x)
{
    return gaussianFunctNormDist(x);
}

__device__ void initBlackCalcVars(blackCalcStruct& blackCalculator, payoffStruct payoff)
{
    blackCalculator.d1 = logf(blackCalculator.forward / blackCalculator.strike) / blackCalculator.stdDev + 0.5f * blackCalculator.stdDev;
    blackCalculator.d2 = blackCalculator.d1 - blackCalculator.stdDev;

    blackCalculator.cum_d1 = cumNormDistOp(blackCalculator.d1);
    blackCalculator.cum_d2 = cumNormDistOp(blackCalculator.d2);
    blackCalculator.n_d1 = cumNormDistDeriv(blackCalculator.d1);
    blackCalculator.n_d2 = cumNormDistDeriv(blackCalculator.d2);

    blackCalculator.x = payoff.strike;
    blackCalculator.DxDstrike = 1.0f;
    blackCalculator.DxDs = 0.0f;

    switch (payoff.type) {
    case CALL:
        blackCalculator.alpha     =  blackCalculator.cum_d1;
        blackCalculator.DalphaDd1 =  blackCalculator.n_d1;
        blackCalculator.beta      = -1.0f * blackCalculator.cum_d2;
        blackCalculator.DbetaDd2  = -1.0f * blackCalculator.n_d2;
        break;
    case PUT:
        blackCalculator.alpha     = -1.0f + blackCalculator.cum_d1;
        blackCalculator.DalphaDd1 =         blackCalculator.n_d1;
        blackCalculator.beta      =  1.0f - blackCalculator.cum_d2;
        blackCalculator.DbetaDd2  = -1.0f * blackCalculator.n_d2;
        break;
    }
}

__device__ void initBlackCalculator(blackCalcStruct& blackCalc, payoffStruct payoff,
                                    float forwardPrice, float stdDev, float riskFreeDiscount)
{
    blackCalc.strike = payoff.strike;
    blackCalc.forward = forwardPrice;
    blackCalc.stdDev = stdDev;
    blackCalc.discount = riskFreeDiscount;
    blackCalc.variance = stdDev * stdDev;

    initBlackCalcVars(blackCalc, payoff);
}

__device__ float getResultVal(blackCalcStruct blackCalculator)
{
    float result = blackCalculator.discount * (blackCalculator.forward *
                    blackCalculator.alpha + blackCalculator.x * blackCalculator.beta);
    return result;
}

// ===== Main kernel — matches original getOutValOption structure =====
__launch_bounds__(256)
__global__ void getOutValOptionKernel(
    int numVals,
    const int* __restrict__   types,
    const float* __restrict__ strikes,
    const float* __restrict__ spots,
    const float* __restrict__ qs,
    const float* __restrict__ rs,
    const float* __restrict__ ts,
    const float* __restrict__ vols,
    float* __restrict__ outputVals
) {
    int optionNum = blockIdx.x * blockDim.x + threadIdx.x;
    if (optionNum >= numVals) return;

    // Build option structs (matches original getOutValOption)
    optionInputStruct threadOption;
    threadOption.type   = types[optionNum];
    threadOption.strike = strikes[optionNum];
    threadOption.spot   = spots[optionNum];
    threadOption.q      = qs[optionNum];
    threadOption.r      = rs[optionNum];
    threadOption.t      = ts[optionNum];
    threadOption.vol    = vols[optionNum];

    payoffStruct currPayoff;
    currPayoff.type = threadOption.type;
    currPayoff.strike = threadOption.strike;

    yieldTermStruct qTS;
    qTS.timeYearFraction = threadOption.t;
    qTS.forward = threadOption.q;

    yieldTermStruct rTS;
    rTS.timeYearFraction = threadOption.t;
    rTS.forward = threadOption.r;

    blackVolStruct volTS;
    volTS.timeYearFraction = threadOption.t;
    volTS.volatility = threadOption.vol;

    blackScholesMertStruct stochProcess;
    stochProcess.x0 = threadOption.spot;
    stochProcess.dividendTS = qTS;
    stochProcess.riskFreeTS = rTS;
    stochProcess.blackVolTS = volTS;

    optionStruct currOption;
    currOption.payoff = currPayoff;
    currOption.yearFractionTime = threadOption.t;
    currOption.pricingEngine = stochProcess;

    float variance = getBlackVolBlackVar(currOption.pricingEngine.blackVolTS);
    float dividendDiscount = getDiscountOnDividendYield(currOption.yearFractionTime, currOption.pricingEngine.dividendTS);
    float riskFreeDiscount = getDiscountOnRiskFreeRate(currOption.yearFractionTime, currOption.pricingEngine.riskFreeTS);
    float spot = currOption.pricingEngine.x0;

    float forwardPrice = spot * dividendDiscount / riskFreeDiscount;

    blackCalcStruct blackCalc;
    initBlackCalculator(blackCalc, currOption.payoff, forwardPrice, sqrtf(variance), riskFreeDiscount);

    float resultVal = getResultVal(blackCalc);
    outputVals[optionNum] = resultVal;
}

// ===== Host interface (compute_only) =====

static int    g_N = 0;
static int*   d_types   = nullptr;
static float* d_strikes = nullptr;
static float* d_spots   = nullptr;
static float* d_qs      = nullptr;
static float* d_rs      = nullptr;
static float* d_ts      = nullptr;
static float* d_vols    = nullptr;
static float* d_prices  = nullptr;

extern "C" void solution_free(void)
{
    if (d_types)   { cudaFree(d_types);   d_types   = nullptr; }
    if (d_strikes) { cudaFree(d_strikes); d_strikes = nullptr; }
    if (d_spots)   { cudaFree(d_spots);   d_spots   = nullptr; }
    if (d_qs)      { cudaFree(d_qs);      d_qs      = nullptr; }
    if (d_rs)      { cudaFree(d_rs);      d_rs      = nullptr; }
    if (d_ts)      { cudaFree(d_ts);      d_ts      = nullptr; }
    if (d_vols)    { cudaFree(d_vols);    d_vols    = nullptr; }
    if (d_prices)  { cudaFree(d_prices);  d_prices  = nullptr; }
    g_N = 0;
}

extern "C" void solution_compute(int N,
                                  const int* types, const float* strikes, const float* spots,
                                  const float* qs, const float* rs, const float* ts,
                                  const float* vols,
                                  float* prices)
{
    size_t szi = (size_t)N * sizeof(int);
    size_t szf = (size_t)N * sizeof(float);

    if (g_N != N) {
        solution_free();
        cudaMalloc(&d_types,   szi);
        cudaMalloc(&d_strikes, szf);
        cudaMalloc(&d_spots,   szf);
        cudaMalloc(&d_qs,      szf);
        cudaMalloc(&d_rs,      szf);
        cudaMalloc(&d_ts,      szf);
        cudaMalloc(&d_vols,    szf);
        cudaMalloc(&d_prices,  szf);
        g_N = N;
    }

    cudaMemcpy(d_types,   types,   szi, cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, strikes, szf, cudaMemcpyHostToDevice);
    cudaMemcpy(d_spots,   spots,   szf, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qs,      qs,      szf, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rs,      rs,      szf, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ts,      ts,      szf, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vols,    vols,    szf, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    getOutValOptionKernel<<<blocks, threadsPerBlock>>>(
        N, d_types, d_strikes, d_spots, d_qs, d_rs, d_ts, d_vols, d_prices);

    cudaMemcpy(prices, d_prices, szf, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
