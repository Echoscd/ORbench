#include <cuda_runtime.h>
#include <math.h>

#define BS_DBL_MIN 1e-30f
#define M_SQRT_2 0.7071067811865475244008443621048490392848359376887f

#define ERROR_FUNCT_tiny   0.000000000000000000001f
#define ERROR_FUNCT_one    1.00000000000000000000e+00f
#define ERROR_FUNCT_erx    8.45062911510467529297e-01f
#define ERROR_FUNCT_efx    1.28379167095512586316e-01f
#define ERROR_FUNCT_efx8   1.02703333676410069053e+00f
#define ERROR_FUNCT_pp0    1.28379167095512558561e-01f
#define ERROR_FUNCT_pp1   -3.25042107247001499370e-01f
#define ERROR_FUNCT_pp2   -2.84817495755985104766e-02f
#define ERROR_FUNCT_pp3   -5.77027029648944159157e-03f
#define ERROR_FUNCT_pp4   -2.37630166566501626084e-05f
#define ERROR_FUNCT_qq1    3.97917223959155352819e-01f
#define ERROR_FUNCT_qq2    6.50222499887672944485e-02f
#define ERROR_FUNCT_qq3    5.08130628187576562776e-03f
#define ERROR_FUNCT_qq4    1.32494738004321644526e-04f
#define ERROR_FUNCT_qq5   -3.96022827877536812320e-06f
#define ERROR_FUNCT_pa0   -2.36211856075265944077e-03f
#define ERROR_FUNCT_pa1    4.14856118683748331666e-01f
#define ERROR_FUNCT_pa2   -3.72207876035701323847e-01f
#define ERROR_FUNCT_pa3    3.18346619901161753674e-01f
#define ERROR_FUNCT_pa4   -1.10894694282396677476e-01f
#define ERROR_FUNCT_pa5    3.54783043256182359371e-02f
#define ERROR_FUNCT_pa6   -2.16637559486879084300e-03f
#define ERROR_FUNCT_qa1    1.06420880400844228286e-01f
#define ERROR_FUNCT_qa2    5.40397917702171048937e-01f
#define ERROR_FUNCT_qa3    7.18286544141962662868e-02f
#define ERROR_FUNCT_qa4    1.26171219808761642112e-01f
#define ERROR_FUNCT_qa5    1.36370839120290507362e-02f
#define ERROR_FUNCT_qa6    1.19844998467991074170e-02f
#define ERROR_FUNCT_ra0   -9.86494403484714822705e-03f
#define ERROR_FUNCT_ra1   -6.93858572707181764372e-01f
#define ERROR_FUNCT_ra2   -1.05586262253232909814e+01f
#define ERROR_FUNCT_ra3   -6.23753324503260060396e+01f
#define ERROR_FUNCT_ra4   -1.62396669462573470355e+02f
#define ERROR_FUNCT_ra5   -1.84605092906711035994e+02f
#define ERROR_FUNCT_ra6   -8.12874355063065934246e+01f
#define ERROR_FUNCT_ra7   -9.81432934416914548592e+00f
#define ERROR_FUNCT_sa1    1.96512716674392571292e+01f
#define ERROR_FUNCT_sa2    1.37657754143519042600e+02f
#define ERROR_FUNCT_sa3    4.34565877475229228821e+02f
#define ERROR_FUNCT_sa4    6.45387271733267880336e+02f
#define ERROR_FUNCT_sa5    4.29008140027567833386e+02f
#define ERROR_FUNCT_sa6    1.08635005541779435134e+02f
#define ERROR_FUNCT_sa7    6.57024977031928170135e+00f
#define ERROR_FUNCT_sa8   -6.04244152148580987438e-02f
#define ERROR_FUNCT_rb0   -9.86494292470009928597e-03f
#define ERROR_FUNCT_rb1   -7.99283237680523006574e-01f
#define ERROR_FUNCT_rb2   -1.77579549177547519889e+01f
#define ERROR_FUNCT_rb3   -1.60636384855821916062e+02f
#define ERROR_FUNCT_rb4   -6.37566443368389627722e+02f
#define ERROR_FUNCT_rb5   -1.02509513161107724954e+03f
#define ERROR_FUNCT_rb6   -4.83519191608651397019e+02f
#define ERROR_FUNCT_sb1    3.03380607434824582924e+01f
#define ERROR_FUNCT_sb2    3.25792512996573918826e+02f
#define ERROR_FUNCT_sb3    1.53672958608443695994e+03f
#define ERROR_FUNCT_sb4    3.19985821950859553908e+03f
#define ERROR_FUNCT_sb5    2.55305040643316442583e+03f
#define ERROR_FUNCT_sb6    4.74528541206955367215e+02f
#define ERROR_FUNCT_sb7   -2.24409524465858183362e+01f

__device__ float errorFunctGpu(float x)
{
    float R, S, P, Q, s, y, z, r, ax;

    ax = fabsf(x);

    if (ax < 0.84375f) {
        if (ax < 3.7252902984e-09f) {
            if (ax < BS_DBL_MIN * 16.0f)
                return 0.125f * (8.0f * x + (ERROR_FUNCT_efx8) * x);
            return x + (ERROR_FUNCT_efx) * x;
        }
        z = x * x;
        r = ERROR_FUNCT_pp0 + z * (ERROR_FUNCT_pp1 + z * (ERROR_FUNCT_pp2 + z * (ERROR_FUNCT_pp3 + z * ERROR_FUNCT_pp4)));
        s = ERROR_FUNCT_one + z * (ERROR_FUNCT_qq1 + z * (ERROR_FUNCT_qq2 + z * (ERROR_FUNCT_qq3 + z * (ERROR_FUNCT_qq4 + z * ERROR_FUNCT_qq5))));
        y = r / s;
        return x + x * y;
    }
    if (ax < 1.25f) {
        s = ax - ERROR_FUNCT_one;
        P = ERROR_FUNCT_pa0 + s * (ERROR_FUNCT_pa1 + s * (ERROR_FUNCT_pa2 + s * (ERROR_FUNCT_pa3 + s * (ERROR_FUNCT_pa4 + s * (ERROR_FUNCT_pa5 + s * ERROR_FUNCT_pa6)))));
        Q = ERROR_FUNCT_one + s * (ERROR_FUNCT_qa1 + s * (ERROR_FUNCT_qa2 + s * (ERROR_FUNCT_qa3 + s * (ERROR_FUNCT_qa4 + s * (ERROR_FUNCT_qa5 + s * ERROR_FUNCT_qa6)))));
        if (x >= 0.0f) return ERROR_FUNCT_erx + P / Q;
        else return -1.0f * ERROR_FUNCT_erx - P / Q;
    }
    if (ax >= 6.0f) {
        if (x >= 0.0f) return ERROR_FUNCT_one - ERROR_FUNCT_tiny;
        else return ERROR_FUNCT_tiny - ERROR_FUNCT_one;
    }

    s = ERROR_FUNCT_one / (ax * ax);

    if (ax < 2.85714285714285f) {
        R = ERROR_FUNCT_ra0 + s * (ERROR_FUNCT_ra1 + s * (ERROR_FUNCT_ra2 + s * (ERROR_FUNCT_ra3 + s * (ERROR_FUNCT_ra4 + s * (ERROR_FUNCT_ra5 + s * (ERROR_FUNCT_ra6 + s * ERROR_FUNCT_ra7))))));
        S = ERROR_FUNCT_one + s * (ERROR_FUNCT_sa1 + s * (ERROR_FUNCT_sa2 + s * (ERROR_FUNCT_sa3 + s * (ERROR_FUNCT_sa4 + s * (ERROR_FUNCT_sa5 + s * (ERROR_FUNCT_sa6 + s * (ERROR_FUNCT_sa7 + s * ERROR_FUNCT_sa8)))))));
    } else {
        R = ERROR_FUNCT_rb0 + s * (ERROR_FUNCT_rb1 + s * (ERROR_FUNCT_rb2 + s * (ERROR_FUNCT_rb3 + s * (ERROR_FUNCT_rb4 + s * (ERROR_FUNCT_rb5 + s * ERROR_FUNCT_rb6)))));
        S = ERROR_FUNCT_one + s * (ERROR_FUNCT_sb1 + s * (ERROR_FUNCT_sb2 + s * (ERROR_FUNCT_sb3 + s * (ERROR_FUNCT_sb4 + s * (ERROR_FUNCT_sb5 + s * (ERROR_FUNCT_sb6 + s * ERROR_FUNCT_sb7))))));
    }

    r = expf(-ax * ax - 0.5625f + R / S);
    if (x >= 0.0f) return ERROR_FUNCT_one - r / ax;
    else return r / ax - ERROR_FUNCT_one;
}

__device__ float cumNormDistOpGpu(float z)
{
    return 0.5f * (1.0f + errorFunctGpu(z * M_SQRT_2));
}

__global__ void blackScholesKernel(
    int N,
    const int* __restrict__ types,
    const float* __restrict__ strikes,
    const float* __restrict__ spots,
    const float* __restrict__ qs,
    const float* __restrict__ rs,
    const float* __restrict__ ts,
    const float* __restrict__ vols,
    float* __restrict__ prices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int type = types[i];
        float strike = strikes[i];
        float spot = spots[i];
        float q = qs[i];
        float r = rs[i];
        float t = ts[i];
        float vol = vols[i];

        float riskFreeDiscount = 1.0f / expf(r * t);
        float dividendDiscount = 1.0f / expf(q * t);
        
        float forwardPrice = spot * dividendDiscount / riskFreeDiscount;

        float stdDev = vol * sqrtf(t);
        
        float d1 = (logf(forwardPrice / strike) / stdDev) + 0.5f * stdDev;
        float d2 = d1 - stdDev;

        float cum_d1 = cumNormDistOpGpu(d1);
        float cum_d2 = cumNormDistOpGpu(d2);

        float alpha, beta;
        if (type == 0) { // CALL
            alpha = cum_d1;
            beta = -cum_d2;
        } else { // PUT
            alpha = cum_d1 - 1.0f;
            beta = 1.0f - cum_d2;
        }

        float price = riskFreeDiscount * (forwardPrice * alpha + strike * beta);
        
        if (isnan(price) || isinf(price)) {
            price = 1e30f;
        }
        
        prices[i] = price;
    }
}

extern "C" void solution_compute(
    int N,
    const int* types,
    const float* strikes,
    const float* spots,
    const float* qs,
    const float* rs,
    const float* ts,
    const float* vols,
    float* prices
) {
    if (N <= 0) return;

    // Allocate a single contiguous block of device memory for all arrays
    size_t bytes_int = N * sizeof(int);
    size_t bytes_float = N * sizeof(float);
    size_t total_bytes = bytes_int + 7 * bytes_float;

    void* d_buffer;
    cudaMalloc(&d_buffer, total_bytes);

    int* d_types = (int*)d_buffer;
    float* d_strikes = (float*)(d_types + N);
    float* d_spots = d_strikes + N;
    float* d_qs = d_spots + N;
    float* d_rs = d_qs + N;
    float* d_ts = d_rs + N;
    float* d_vols = d_ts + N;
    float* d_prices = d_vols + N;

    // Copy data to device asynchronously
    cudaMemcpyAsync(d_types, types, bytes_int, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_strikes, strikes, bytes_float, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_spots, spots, bytes_float, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_qs, qs, bytes_float, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_rs, rs, bytes_float, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_ts, ts, bytes_float, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_vols, vols, bytes_float, cudaMemcpyHostToDevice);

    // Launch configuration
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Kernel execution
    blackScholesKernel<<<numBlocks, blockSize>>>(
        N, d_types, d_strikes, d_spots, d_qs, d_rs, d_ts, d_vols, d_prices
    );

    // Synchronous copy back blocks until all previous operations in the default stream complete
    cudaMemcpy(prices, d_prices, bytes_float, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_buffer);
}