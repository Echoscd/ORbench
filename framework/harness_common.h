// harness_common.h - ORBench v2.2 benchmark harness skeleton (C-only)
//
// Included by both framework/harness_gpu.cu and framework/harness_cpu.c
// so this file MUST be valid C (not C++).
//
// Three-layer architecture:
//   harness (this file, generic)
//     -> task_io (task-specific I/O adapter, provided per task)
//       -> solution (LLM-written, pure computation, no I/O)
//
// Timing output (timing.json):
//   init_ms:  time for task_setup (includes solution_init)
//   solve_ms: mean time per task_run call (solution_compute)
//   Both are reported so downstream can compute:
//     solve_speedup = cpu_solve / gpu_solve
//     total_speedup = (cpu_init + cpu_solve) / (gpu_init + gpu_solve)

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
        fprintf(stderr, "Usage: %s <data_dir> [--validate] [--warmup N] [--trials N]\n", argv[0]);
        return 1;
    }
    const char* data_dir = argv[1];

    // Parse optional flags (order-independent)
    int do_validate = 0;
    int warmup     = WARMUP;      // default from harness_gpu.cu / harness_cpu.c
    int num_trials = NUM_TRIALS;
    {
        int i;
        for (i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--validate") == 0) {
                do_validate = 1;
            } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
                warmup = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc) {
                num_trials = atoi(argv[++i]);
            }
        }
    }

    // 1. Load input.bin
    char path[512];
    snprintf(path, sizeof(path), "%s/input.bin", data_dir);
    TaskData data = load_input_bin(path);

#ifdef ORBENCH_COMPUTE_ONLY
    // ============ compute_only mode ============
    // Every trial does setup + run + cleanup.
    // This prevents hiding computation in init (e.g. CUDA Graph recording).
    float init_ms = 0.0f;

    // Warmup (full cycle each time)
    for (int w = 0; w < warmup; w++) {
        void* ctx = task_setup(&data, data_dir);
        SYNC();
        task_run(ctx);
        SYNC();
        task_cleanup(ctx);
    }

    // Timed trials: setup + run together
    float total_ms = 0.0f, min_ms = 1e9f, max_ms = 0.0f;
    void* last_ctx = NULL;

    for (int t = 0; t < num_trials; t++) {
        TIMER_START();
        void* ctx = task_setup(&data, data_dir);
        SYNC();
        task_run(ctx);
        SYNC();
        TIMER_STOP();

        float ms = TIMER_ELAPSED_MS();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;

        // Keep last ctx for validate, cleanup previous
        if (last_ctx) task_cleanup(last_ctx);
        last_ctx = ctx;
    }

    float solve_mean_ms = (num_trials > 0) ? total_ms / (float)num_trials : 0.0f;

    // Validate: last_ctx already has results from last trial
    if (do_validate && last_ctx) {
        snprintf(path, sizeof(path), "%s/output.txt", data_dir);
        task_write_output(last_ctx, path);
    }
    if (last_ctx) task_cleanup(last_ctx);

#else
    // ============ init_compute mode (default, unchanged) ============

    // 2. Setup: task_io parses requests, calls solution_init (TIMED)
    TIMER_START();
    void* ctx = task_setup(&data, data_dir);
    SYNC();
    TIMER_STOP();
    float init_ms = TIMER_ELAPSED_MS();

    if (!ctx) {
        fprintf(stderr, "task_setup failed\n");
        free_task_data(&data);
        return 1;
    }

    // 3. Warmup (not timed)
    for (int w = 0; w < warmup; w++) {
        task_run(ctx);
        SYNC();
    }

    // 4. Timed trials (solve only)
    float total_ms = 0.0f, min_ms = 1e9f, max_ms = 0.0f;
    for (int t = 0; t < num_trials; t++) {
        TIMER_START();
        task_run(ctx);
        SYNC();
        TIMER_STOP();

        float ms = TIMER_ELAPSED_MS();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }

    float solve_mean_ms = (num_trials > 0) ? total_ms / (float)num_trials : 0.0f;

    // 5. Validate: run once and write output.txt
    if (do_validate) {
        task_run(ctx);
        SYNC();
        snprintf(path, sizeof(path), "%s/output.txt", data_dir);
        task_write_output(ctx, path);
    }

    // 6. Cleanup
    task_cleanup(ctx);
#endif

    // Output (shared by both modes)
    printf("TIME_MS: %.3f\n", solve_mean_ms);
    printf("INIT_MS: %.3f\n", init_ms);
    fprintf(stderr, "Init:  %.3f ms\n", init_ms);
    fprintf(stderr, "Solve: mean=%.3f ms, min=%.3f ms, max=%.3f ms (%d trials)\n",
            solve_mean_ms, min_ms, max_ms, num_trials);

    // Write detailed timing to timing.json (machine-readable)
    snprintf(path, sizeof(path), "%s/timing.json", data_dir);
    {
        FILE* tf = fopen(path, "w");
        if (tf) {
            fprintf(tf, "{"
                    "\"interface_mode\":\"%s\","
                    "\"init_ms\":%.3f,"
                    "\"mean_ms\":%.3f,\"min_ms\":%.3f,\"max_ms\":%.3f,"
                    "\"num_trials\":%d"
                    "}\n",
#ifdef ORBENCH_COMPUTE_ONLY
                    "compute_only",
#else
                    "init_compute",
#endif
                    init_ms, solve_mean_ms, min_ms, max_ms, num_trials);
            fclose(tf);
        }
    }

    free_task_data(&data);
    return 0;
}

#endif // ORBENCH_HARNESS_COMMON_H
