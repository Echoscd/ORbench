// harness_gpu.cu - ORBench v2 GPU harness (CUDA Events timing)

#include <cuda_runtime.h>

#define WARMUP     3
#define NUM_TRIALS 10
#define SYNC()     cudaDeviceSynchronize()

static cudaEvent_t _ev_start, _ev_stop;
#define TIMER_START() do { \
    cudaEventCreate(&_ev_start); cudaEventCreate(&_ev_stop); \
    cudaEventRecord(_ev_start); } while(0)
#define TIMER_STOP() do { \
    cudaEventRecord(_ev_stop); cudaEventSynchronize(_ev_stop); } while(0)
#define TIMER_ELAPSED_MS() ({ float _ms; \
    cudaEventElapsedTime(&_ms, _ev_start, _ev_stop); \
    cudaEventDestroy(_ev_start); cudaEventDestroy(_ev_stop); _ms; })

#include "harness_common.h"

int main(int argc, char** argv) { return harness_main(argc, argv); }







