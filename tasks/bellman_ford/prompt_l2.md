# Bellman-Ford 单源最短路 — CUDA 优化

## 任务

给定一个有向加权图（CSR 格式）和一批 `(source, target)` 查询，对每个查询计算从 `source` 到 `target` 的最短距离。

请实现以下三个 `extern "C"` 函数（在一个 `.cu` 文件中）。**不需要** 写 `main()`、不读/写任何文件。

## 你需要实现的函数

```c
// 1. 初始化：接收 host 端图数据指针，做 cudaMalloc + H2D 等
//    只调用一次，不计时
extern "C" void solution_init(
    int V, int E,
    const int*   h_row_offsets,   // host, (V+1) 个 int, CSR 行偏移
    const int*   h_col_indices,   // host, E 个 int, CSR 列索引
    const float* h_weights        // host, E 个 float, 边权（正数）
);

// 2. 计算：处理一批查询，将结果写入 distances
//    会被多次调用（warmup + 计时），必须幂等
//    distances[i] = 从 sources[i] 到 targets[i] 的最短距离
//    不可达时填 1e30f
extern "C" void solution_compute(
    int num_requests,
    const int*   h_sources,       // host, num_requests 个 int
    const int*   h_targets,       // host, num_requests 个 int
    float*       h_distances      // host, num_requests 个 float (输出)
);

// 3. 释放 GPU 资源
//    只调用一次，不计时
extern "C" void solution_free(void);
```

## 关键约束

1. 三个函数都用 `extern "C"` 链接
2. `solution_compute` 的所有参数都是 **host 指针**
3. 你自己决定何时 H2D / D2H，是否用 pinned memory、streams 等
4. **不要** 在 `solution_compute` 里做 `cudaMalloc`（它会被反复调用）
5. 文件里 **不要** 出现 `fopen` / `fread` / `fwrite` / `fprintf` / `printf` 等 I/O 调用

## 输入规模

| 规模 | V | E | 查询数 |
|------|---|---|--------|
| small | 1,000 | 5,000 | 100 |
| medium | 100,000 | 500,000 | 100 |
| large | 500,000 | 2,500,000 | 100 |

查询中 `source` 分为 10 组（每组 10 个不同的 `source`，每个 `source` 配 10 个 `target`），共 100 条。
同一个 `source` 的查询可以共享一次 Bellman-Ford 计算。

## CPU 参考实现

```c
#define INF_VAL 1e30f

void solution_compute(int num_requests,
                      const int* sources, const int* targets,
                      float* distances) {
    float* dist_buf = (float*)malloc(V * sizeof(float));
    for (int r = 0; r < num_requests; r++) {
        int src = sources[r], tgt = targets[r];
        for (int i = 0; i < V; i++) dist_buf[i] = INF_VAL;
        dist_buf[src] = 0.0f;

        for (int round = 0; round < V - 1; round++) {
            int updated = 0;
            for (int u = 0; u < V; u++) {
                if (dist_buf[u] >= INF_VAL) continue;
                for (int idx = row_offsets[u]; idx < row_offsets[u+1]; idx++) {
                    float nd = dist_buf[u] + weights[idx];
                    if (nd < dist_buf[col_indices[idx]]) {
                        dist_buf[col_indices[idx]] = nd;
                        updated = 1;
                    }
                }
            }
            if (!updated) break;
        }
        distances[r] = (tgt >= 0 && tgt < V) ? dist_buf[tgt] : INF_VAL;
    }
    free(dist_buf);
}
```

## 优化提示

- 将图数据 **一次性** 拷贝到 GPU（在 `solution_init` 里），后续查询复用
- 同一个 `source` 的 10 个查询只需一次 Bellman-Ford，可在 `solution_compute` 里去重
- 可用 edge-parallel 或 vertex-parallel 方案加速 relaxation
- 利用 shared memory / warp-level primitives 加速
- 用 early-exit 检测收敛
