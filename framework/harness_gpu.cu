// harness_gpu.cu - AccelEval v2 GPU harness (CUDA Events timing)

#include <cuda_runtime.h>

#define WARMUP     3
#define NUM_TRIALS 10
#define SYNC()     cudaDeviceSynchronize()
// Wipe device state (frees all model-private persistent device allocations,
// destroys streams/events) so that any timed call must allocate from scratch.
// No-op on the CPU harness via the matching macro in harness_cpu.c.
#define RESET_DEVICE_STATE() do { cudaDeviceSynchronize(); cudaDeviceReset(); } while(0)

// Establish the CUDA context outside any timed region so that context
// creation / module loading is never charged to the solution (relevant for
// cold runs with --warmup 0 and for re-init after RESET_DEVICE_STATE).
#define CTX_PREINIT() (void)cudaFree(0)

// Audit-round error probe: surface any synchronous or sticky asynchronous
// CUDA error produced by the preceding task_run (S1 signal). NULL == clean.
static const char* audit_check_err(void) {
    cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess) e = cudaGetLastError();
    return (e == cudaSuccess) ? (const char*)0 : cudaGetErrorString(e);
}
#define AUDIT_CHECK_ERR() audit_check_err()

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









