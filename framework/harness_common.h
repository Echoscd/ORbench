// harness_common.h - AccelEval v2.1 request-based benchmark harness skeleton (C-only)
//
// Included by both framework/harness_gpu.cu and framework/harness_cpu.c
// so this file MUST be valid C (not C++).
//
// Three-layer architecture:
//   harness (this file, generic)
//     → task_io (task-specific I/O adapter, provided per task)
//       → solution (LLM-written, pure computation, no I/O)
//
// Timing-audit support (see rebuttal/AUDIT_PLAN.md):
//   S0  every call (warmup AND timed) is individually timed and recorded in
//       timing.json, so cost hidden in an early call is visible in the data.
//   S1  --audit-rounds N appends N reset-and-revalidate rounds: device state
//       is wiped between rounds (RESET_DEVICE_STATE), so any solution that
//       depends on cross-call device state errors out or produces a wrong
//       output_audit_<r>.txt. Compliant solutions are unaffected.
//   Context creation / module loading is established via CTX_PREINIT()
//   outside every timed region, so cold runs (--warmup 0 --trials 1) charge
//   the solution only for its own per-call work (S2 uses this).

#ifndef ACCELEVAL_HARNESS_COMMON_H
#define ACCELEVAL_HARNESS_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "acceleval_io.h"

// Max per-call samples recorded into timing.json (timing itself is not capped)
#define HARNESS_REC_MAX 64

// Implemented by task_io layer (task_io.cu / task_io_cpu.c)
// Framework-agnostic: harness knows nothing about task-specific data
#ifdef __cplusplus
extern "C" {
#endif
// task_setup: Parse task-specific inputs (requests.txt etc.), return ctx
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
        fprintf(stderr,
            "Usage: %s <data_dir> [--validate] [--warmup N] [--trials N] [--audit-rounds N]\n",
            argv[0]);
        return 1;
    }
    const char* data_dir = argv[1];

    // Parse optional flags (order-independent)
    int do_validate  = 0;
    int warmup       = WARMUP;      // default from harness_gpu.cu / harness_cpu.c
    int num_trials   = NUM_TRIALS;
    int audit_rounds = 0;
    {
        int i;
        for (i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--validate") == 0) {
                do_validate = 1;
            } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
                warmup = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc) {
                num_trials = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--audit-rounds") == 0 && i + 1 < argc) {
                audit_rounds = atoi(argv[++i]);
            }
        }
    }

    // 1. Load input.bin
    char path[512];
    snprintf(path, sizeof(path), "%s/input.bin", data_dir);
    TaskData data = load_input_bin(path);

    // Establish context / load modules outside any timed region.
    CTX_PREINIT();

    // 2. Setup: task_io parses requests (not timed)
    void* ctx = task_setup(&data, data_dir);
    if (!ctx) {
        fprintf(stderr, "task_setup failed\n");
        free_task_data(&data);
        return 1;
    }

    // 3. Warmup — individually timed (S0), excluded from reported statistics
    float warmup_ms[HARNESS_REC_MAX];
    int   warmup_rec = 0;
    for (int w = 0; w < warmup; w++) {
        TIMER_START();
        task_run(ctx);
        SYNC();
        TIMER_STOP();
        {
            float wms = TIMER_ELAPSED_MS();
            if (warmup_rec < HARNESS_REC_MAX) warmup_ms[warmup_rec++] = wms;
        }
    }

    // 4. Timed trials — individually recorded
    float trial_ms[HARNESS_REC_MAX];
    int   trial_rec = 0;
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
        if (trial_rec < HARNESS_REC_MAX) trial_ms[trial_rec++] = ms;
    }

    float mean_ms = (num_trials > 0) ? total_ms / (float)num_trials : 0.0f;
    if (num_trials > 0) {
        printf("TIME_MS: %.3f\n", mean_ms);
        fprintf(stderr, "Timing: mean=%.3f ms, min=%.3f ms, max=%.3f ms (%d trials)\n",
                mean_ms, min_ms, max_ms, num_trials);
    }

    // 4b. Write detailed timing to timing.json (machine-readable).
    // Skipped when num_trials == 0 (e.g. a fresh-process validation run) so a
    // validation-only invocation never clobbers the timing run's results.
    if (num_trials > 0) {
        snprintf(path, sizeof(path), "%s/timing.json", data_dir);
        FILE* tf = fopen(path, "w");
        if (tf) {
            int i;
            fprintf(tf, "{\"mean_ms\":%.3f,\"min_ms\":%.3f,\"max_ms\":%.3f,\"num_trials\":%d",
                    mean_ms, min_ms, max_ms, num_trials);
            fprintf(tf, ",\"warmup\":%d,\"warmup_ms\":[", warmup);
            for (i = 0; i < warmup_rec; i++)
                fprintf(tf, "%s%.3f", i ? "," : "", warmup_ms[i]);
            fprintf(tf, "],\"trial_ms\":[");
            for (i = 0; i < trial_rec; i++)
                fprintf(tf, "%s%.3f", i ? "," : "", trial_ms[i]);
            fprintf(tf, "]}\n");
            fclose(tf);
        }
    }

    // 5. Validate: run once and write output.txt
    if (do_validate) {
        task_run(ctx);
        SYNC();
        snprintf(path, sizeof(path), "%s/output.txt", data_dir);
        task_write_output(ctx, path);
    }

    // 6. Audit rounds (S1: reset-and-revalidate).
    // Device state is wiped between rounds; ctx is rebuilt from scratch each
    // round (task_io state must not survive the reset either). The solution's
    // process-level static state — the thing being audited — DOES survive, so
    // a solution that caches device pointers across calls now dereferences
    // dangling pointers and fails the per-round error probe / output check.
    if (audit_rounds > 0) {
        int r;
        task_cleanup(ctx);      // clean teardown of the warm ctx BEFORE the first reset
        ctx = NULL;
        for (r = 0; r < audit_rounds; r++) {
            RESET_DEVICE_STATE();
            CTX_PREINIT();
            ctx = task_setup(&data, data_dir);
            if (!ctx) {
                printf("AUDIT_ROUND %d: SETUP_FAIL\n", r);
                continue;
            }
            task_run(ctx);
            {
                const char* err = AUDIT_CHECK_ERR();
                snprintf(path, sizeof(path), "%s/output_audit_%d.txt", data_dir, r);
                task_write_output(ctx, path);
                task_cleanup(ctx);
                ctx = NULL;
                printf("AUDIT_ROUND %d: %s\n", r, err ? err : "OK");
            }
        }
        free_task_data(&data);
        return 0;
    }

    // 7. Cleanup
    task_cleanup(ctx);
    free_task_data(&data);
    return 0;
}

#endif // ACCELEVAL_HARNESS_COMMON_H
