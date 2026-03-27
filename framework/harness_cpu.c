// harness_cpu.c - ORBench v2 CPU harness (clock_gettime timing)

#include <time.h>

#define WARMUP     3
#define NUM_TRIALS 10
#define SYNC()     ((void)0)

static struct timespec _ts0, _ts1;
#define TIMER_START()      clock_gettime(CLOCK_MONOTONIC, &_ts0)
#define TIMER_STOP()       clock_gettime(CLOCK_MONOTONIC, &_ts1)
#define TIMER_ELAPSED_MS() (float)((_ts1.tv_sec - _ts0.tv_sec) * 1000.0 + \
                           (_ts1.tv_nsec - _ts0.tv_nsec) * 1e-6)

#include "harness_common.h"

int main(int argc, char** argv) { return harness_main(argc, argv); }















