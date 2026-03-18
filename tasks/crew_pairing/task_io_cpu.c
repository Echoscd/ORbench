// task_io_cpu.c -- crew_pairing CPU I/O adapter layer
//
// Three-layer architecture: harness -> task_io -> solution
// This file bridges harness (generic) and solution (LLM-written pure computation).
//
// Build: gcc -O2 -I framework/
//        framework/harness_cpu.c tasks/crew_pairing/task_io_cpu.c
//        tasks/crew_pairing/cpu_reference.c -o solution_cpu -lm

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ===== LLM / CPU-baseline interface (two functions) =====
extern void solution_init(int N, int num_stations, int base_station,
                          const int* dep_minutes, const int* arr_minutes,
                          const int* dep_stations, const int* arr_stations,
                          float duty_cost_per_hour, float pairing_cost_per_hour,
                          int max_duty_min, int max_block_min,
                          int max_legs_duty, int min_rest_min);

extern void solution_compute(int N, int* assignments);

// ===== task_io internal state =====
typedef struct {
    int N;
    int num_stations;
    int base_station;
    int max_duty_min;
    int max_block_min;
    int max_legs_duty;
    int min_rest_min;
    int pos_fee_x100;
    float duty_rate;
    float pairing_rate;
    const int* dep_min;
    const int* arr_min;
    const int* dep_stn;
    const int* arr_stn;
    int* assignments;    // output buffer
} TaskIOContext;

// ===== harness calls these four functions =====
void* task_setup(const TaskData* data, const char* data_dir) {
    int N             = (int)get_param(data, "N");
    int num_stations  = (int)get_param(data, "num_stations");
    int base_station  = (int)get_param(data, "base_station");
    int duty_rate_x100   = (int)get_param(data, "duty_rate_x100");
    int pairing_rate_x100 = (int)get_param(data, "pairing_rate_x100");
    int max_duty_min  = (int)get_param(data, "max_duty_min");
    int max_block_min = (int)get_param(data, "max_block_min");
    int max_legs_duty = (int)get_param(data, "max_legs_duty");
    int min_rest_min  = (int)get_param(data, "min_rest_min");
    int pos_fee_x100  = (int)get_param(data, "pos_fee_x100");

    float duty_rate    = (float)duty_rate_x100 / 100.0f;
    float pairing_rate = (float)pairing_rate_x100 / 100.0f;

    const int* dep_minutes  = get_tensor_int(data, "dep_minutes");
    const int* arr_minutes  = get_tensor_int(data, "arr_minutes");
    const int* dep_stations = get_tensor_int(data, "dep_stations");
    const int* arr_stations = get_tensor_int(data, "arr_stations");

    if (!dep_minutes || !arr_minutes || !dep_stations || !arr_stations) {
        fprintf(stderr, "[task_io] Missing tensor data\n");
        return NULL;
    }

    solution_init(N, num_stations, base_station,
                  dep_minutes, arr_minutes, dep_stations, arr_stations,
                  duty_rate, pairing_rate,
                  max_duty_min, max_block_min, max_legs_duty, min_rest_min);

    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    ctx->N = N;
    ctx->num_stations = num_stations;
    ctx->base_station = base_station;
    ctx->max_duty_min = max_duty_min;
    ctx->max_block_min = max_block_min;
    ctx->max_legs_duty = max_legs_duty;
    ctx->min_rest_min = min_rest_min;
    ctx->pos_fee_x100 = pos_fee_x100;
    ctx->duty_rate = duty_rate;
    ctx->pairing_rate = pairing_rate;
    ctx->dep_min = dep_minutes;
    ctx->arr_min = arr_minutes;
    ctx->dep_stn = dep_stations;
    ctx->arr_stn = arr_stations;
    ctx->assignments = (int*)calloc((size_t)N, sizeof(int));

    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(ctx->N, ctx->assignments);
}

// Compare function for qsort: sort leg indices by departure time
static const int* _sort_dep_min;
static int _cmp_by_dep(const void* a, const void* b) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    return (_sort_dep_min[ia] > _sort_dep_min[ib]) - (_sort_dep_min[ia] < _sort_dep_min[ib]);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    int N = ctx->N;
    const int* assignments = ctx->assignments;

    // Find max pairing ID
    int max_p = -1;
    for (int i = 0; i < N; i++) {
        if (assignments[i] > max_p) max_p = assignments[i];
    }
    int num_pairings = max_p + 1;

    // Group legs by pairing
    // pairing_legs[p][0] = count, pairing_legs[p][1..] = leg indices
    int** pairing_legs = (int**)calloc((size_t)num_pairings, sizeof(int*));
    int* pairing_count = (int*)calloc((size_t)num_pairings, sizeof(int));

    for (int i = 0; i < N; i++) {
        pairing_count[assignments[i]]++;
    }
    for (int p = 0; p < num_pairings; p++) {
        pairing_legs[p] = (int*)malloc((size_t)pairing_count[p] * sizeof(int));
        pairing_count[p] = 0;  // reuse as index
    }
    for (int i = 0; i < N; i++) {
        int p = assignments[i];
        pairing_legs[p][pairing_count[p]++] = i;
    }

    // Sort each pairing's legs by departure time
    _sort_dep_min = ctx->dep_min;
    for (int p = 0; p < num_pairings; p++) {
        qsort(pairing_legs[p], (size_t)pairing_count[p], sizeof(int), _cmp_by_dep);
    }

    // Compute total cost
    double total_cost = 0.0;
    for (int p = 0; p < num_pairings; p++) {
        int cnt = pairing_count[p];
        int* legs = pairing_legs[p];

        // Block hours: sum of airborne time
        double block_hours = 0.0;
        for (int i = 0; i < cnt; i++) {
            block_hours += (double)(ctx->arr_min[legs[i]] - ctx->dep_min[legs[i]]) / 60.0;
        }

        // Duty hours: partition by rest >= min_rest
        double duty_hours = 0.0;
        int duty_start_min = ctx->dep_min[legs[0]];
        int prev_arr_min = ctx->arr_min[legs[0]];

        for (int i = 1; i < cnt; i++) {
            int rest = ctx->dep_min[legs[i]] - prev_arr_min;
            if (rest >= ctx->min_rest_min) {
                // Close previous duty
                duty_hours += (double)(prev_arr_min - duty_start_min) / 60.0;
                duty_start_min = ctx->dep_min[legs[i]];
            }
            prev_arr_min = ctx->arr_min[legs[i]];
        }
        // Close final duty
        duty_hours += (double)(prev_arr_min - duty_start_min) / 60.0;

        // Positioning fee
        double pos_fee = (ctx->dep_stn[legs[0]] != ctx->base_station)
                         ? (double)ctx->pos_fee_x100 / 100.0 : 0.0;

        total_cost += duty_hours * (double)ctx->duty_rate
                    + block_hours * (double)ctx->pairing_rate
                    + pos_fee;
    }

    // Write output: single line with cost
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    fprintf(f, "%.2f\n", total_cost);
    fclose(f);

    fprintf(stderr, "[task_io] Total cost: %.2f (%d pairings for %d legs)\n",
            total_cost, num_pairings, N);

    // Cleanup grouping
    for (int p = 0; p < num_pairings; p++) free(pairing_legs[p]);
    free(pairing_legs);
    free(pairing_count);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    free(ctx->assignments);
    free(ctx);
}
