#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
extern void solution_init(int n, const int* h_profit);
extern void solution_compute(long long* out_total_profit);
extern void solution_free(void);
#ifdef __cplusplus
}
#endif

typedef struct {
    long long total_profit;
} TaskIOContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    (void)data_dir;
    int n = (int)get_param(data, "n");
    const int* profit = get_tensor_int(data, "profit");
    if (!profit) {
        fprintf(stderr, "[task_io_cpu] Missing profit tensor\n");
        return NULL;
    }
    solution_init(n, profit);
    TaskIOContext* ctx = (TaskIOContext*)calloc(1, sizeof(TaskIOContext));
    return ctx;
}

void task_run(void* test_data) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    solution_compute(&ctx->total_profit);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskIOContext* ctx = (TaskIOContext*)test_data;
    FILE* f = fopen(output_path, "w");
    if (!f) return;
    fprintf(f, "%lld\n", ctx->total_profit);
    fclose(f);
}

void task_cleanup(void* test_data) {
    if (!test_data) return;
    solution_free();
    free(test_data);
}
