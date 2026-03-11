// harness_common.h - ORBench v2.1 request-based benchmark harness skeleton (C-only)
//
// Included by both framework/harness_gpu.cu and framework/harness_cpu.c
// so this file MUST be valid C (not C++).
//
// Three-layer architecture:
//   harness (this file, generic)
//     → task_io (task-specific I/O adapter, provided per task)
//       → solution (LLM-written, pure computation, no I/O)

#ifndef ORBENCH_HARNESS_COMMON_H
#define ORBENCH_HARNESS_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "orbench_io.h"

// Implemented by task_io layer (task_io.cu / task_io_cpu.c)
// Framework-agnostic: harness knows nothing about task-specific data
#ifdef __cplusplus
extern "C" {
#endif
// task_setup: Parse task-specific inputs (requests.txt etc.), call solution_init, return ctx
extern void* task_setup(const TaskData* data, const char* data_dir);

// task_run: Call solution_compute (timed region)
extern void  task_run(void* ctx);

// task_write_output: Write results to output.txt (format controlled by task_io)
extern void  task_write_output(void* ctx, const char* output_path);

// task_cleanup: Call solution_free, free task_io resources
extern void  task_cleanup(void* ctx);
#ifdef __cplusplus
}
#endif

static int harness_main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_dir> [--validate]\n", argv[0]);
        return 1;
    }
    const char* data_dir = argv[1];
    int do_validate = (argc >= 3 && strcmp(argv[2], "--validate") == 0);

    // 1. Load input.bin
    char path[512];
    snprintf(path, sizeof(path), "%s/input.bin", data_dir);
    TaskData data = load_input_bin(path);

    // 2. Setup: task_io parses requests, calls solution_init (not timed)
    void* ctx = task_setup(&data, data_dir);
    if (!ctx) {
        fprintf(stderr, "task_setup failed\n");
        free_task_data(&data);
        return 1;
    }

    // 3. Warmup (not timed)
    for (int w = 0; w < WARMUP; w++) {
        task_run(ctx);
        SYNC();
    }

    // 4. Timed trials
    float total_ms = 0.0f, min_ms = 1e9f, max_ms = 0.0f;
    for (int t = 0; t < NUM_TRIALS; t++) {
        TIMER_START();
        task_run(ctx);
        SYNC();
        TIMER_STOP();

        float ms = TIMER_ELAPSED_MS();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }

    float mean_ms = total_ms / (float)NUM_TRIALS;
    printf("TIME_MS: %.3f\n", mean_ms);
    fprintf(stderr, "Timing: mean=%.3f ms, min=%.3f ms, max=%.3f ms (%d trials)\n",
            mean_ms, min_ms, max_ms, NUM_TRIALS);

    // 5. Validate: run once and write output.txt
    if (do_validate) {
        task_run(ctx);
        SYNC();
        snprintf(path, sizeof(path), "%s/output.txt", data_dir);
        task_write_output(ctx, path);
    }

    // 6. Cleanup
    task_cleanup(ctx);
    free_task_data(&data);
    return 0;
}

#endif // ORBENCH_HARNESS_COMMON_H
