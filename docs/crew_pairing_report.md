# 航空机组配对问题 (Crew Pairing) 的 GPU 加速实验报告

## 1. 问题背景

### 1.1 什么是机组配对问题

航空公司需要为每一条计划航段（flight leg）分配机组人员。一个**配对（pairing）**是一组航段的序列，由同一组机组执飞。机组配对问题（Crew Pairing Problem, CPP）的目标是：**找到一组配对，使得每条航段恰好被覆盖一次，且总成本最小。**

这是航空运筹优化中最核心的问题之一，通常建模为**集合划分问题（Set Partitioning Problem）**，属于 NP-Hard。

### 1.2 约束条件

每个配对内部的航段按时间排列，相邻航段之间的间隔决定了它们属于同一个**执勤段（duty）**还是不同的执勤段：

| 约束 | 参数 | 说明 |
|------|------|------|
| 最大执勤时长 | H_d_max = 14 小时 | 一个 duty 从第一条航段出发到最后一条航段到达的总时间 |
| 最大飞行时长 | H_b_max = 10 小时 | 一个 duty 内所有航段的实际飞行（block）时间之和 |
| 最大航段数 | L_max = 6 | 一个 duty 最多包含 6 条航段 |
| 最小休息时间 | R_min = 9 小时 | 两个 duty 之间的最小间隔 |
| 基地 | NKX | 机组的驻地 |
| 定位费 | C_pos = $10,000 | 若配对首航段不从基地出发，需额外支付定位费 |

### 1.3 成本模型

每个配对 p 的成本为：

```
cost(p) = duty_hours × DutyCostPerHour + block_hours × ParingCostPerHour + pos_fee
```

其中：
- **duty_hours** = 各执勤段时长之和（每段 = 末航段到达时间 - 首航段出发时间）
- **block_hours** = 所有航段实际飞行时间之和
- **pos_fee** = 首航段从非基地出发时收取 $10,000

### 1.4 数学规划模型

```
min  Σ_p cost(p) × x_p
s.t. Σ_{p∋f} x_p = 1,   ∀ flight leg f    （每条航段恰好被覆盖一次）
     x_p ∈ {0, 1}
```

这是一个标准的**集合划分整数规划**，候选配对集 P 的规模随航段数呈指数增长。

---

## 2. 输入输出格式

### 2.1 原始数据（CSV 格式）

```csv
FltNum,DptrDate,DptrTime,DptrStn,ArrvDate,ArrvTime,ArrvStn,Comp,DutyCostPerHour,ParingCostPerHour
FA680,8/11/2021,8:00,NKX,8/11/2021,9:30,PGX,C1F1,680.0,20.0
FA681,8/11/2021,10:10,PGX,8/11/2021,11:40,NKX,C1F1,,
...
```

### 2.2 ORBench 编码格式

CSV 数据被转换为 ORBench 标准二进制格式 `input.bin`，包含以下张量和参数：

**张量（Tensors）：**
| 名称 | 类型 | 大小 | 说明 |
|------|------|------|------|
| dep_minutes | int32 | N | 各航段出发时间（距 epoch 的分钟数） |
| arr_minutes | int32 | N | 各航段到达时间 |
| dep_stations | int32 | N | 出发机场 ID（0-indexed） |
| arr_stations | int32 | N | 到达机场 ID |

**参数（Params）：**
| 名称 | 说明 |
|------|------|
| N | 航段总数 |
| num_stations | 机场数量 |
| base_station | 基地机场 ID |
| duty_rate_x100 | DutyCostPerHour × 100 |
| pairing_rate_x100 | ParingCostPerHour × 100 |

### 2.3 输出

LLM 生成的求解器需要实现两个 C 函数：

```c
// 初始化：接收航段数据（仅调用一次，不计入求解时间）
void solution_init(int N, int num_stations, int base_station,
                   const int* dep_minutes, const int* arr_minutes,
                   const int* dep_stations, const int* arr_stations,
                   float duty_cost_per_hour, float pairing_cost_per_hour,
                   int max_duty_min, int max_block_min,
                   int max_legs_duty, int min_rest_min);

// 求解：输出每条航段的配对分组 ID（多次调用，计时）
void solution_compute(int N, int* assignments);
```

`assignments[i]` = 航段 i 所属的配对编号。框架根据分组计算总成本。

### 2.4 数据规模

| 规模 | 航段数 | 机场数 | 数据来源 |
|------|--------|--------|----------|
| small | 37 | 7 | DataA_chunk3_id1（3 天排班） |
| medium | 206 | 7 | DataA_chunk15_id1（15 天排班） |
| large | 2,239 | 39 | DataB_chunk5_id1（5 天排班） |

---

## 3. CPU Baseline 设计

CPU baseline 采用**两阶段混合策略**，取两种算法的较优解：

### 3.1 策略 A：SPPRC + 贪心集合覆盖

**阶段 1 — 候选配对生成（SPPRC）：**

以每条航段为起点，运行**带资源约束的最短路算法（Shortest Path Problem with Resource Constraints）**，在连接图上搜索可行配对：

- **连接图**：若航段 j 的出发时间晚于航段 i 的到达时间，且 `arr_station[i] == dep_station[j]`（站点连续），则 i→j 有一条连接边
- **标签（Label）**：每个标签记录 `{cost, duty_start, duty_block, duty_legs}`
- **支配关系剪枝**：若标签 A 在所有资源维度上都优于标签 B，则 B 被淘汰
- **每个起点保留 top-20 最优配对**

复杂度约 O(N × 搜索深度 × labels/节点)。

**阶段 2 — 贪心集合覆盖：**

对所有候选配对计算**节约值（savings）**：

```
savings(p) = Σ_{f∈p} single_leg_cost(f) - pairing_cost(p)
```

即：将这些航段合并为一个配对，比各自单独飞能省多少钱。按 savings 降序排列，贪心选择不重叠的配对。

### 3.2 策略 B：顺序贪心

按出发时间遍历每条航段，尝试追加到已有配对（first-fit）：
- 检查站点连续性（上一航段到达 == 当前航段出发）
- 检查执勤约束（duty hours / block hours / legs）
- 若无法追加，新建配对

### 3.3 混合选择

两种策略独立运行，用 `compute_cost()` 评估总成本，取更优解：

```c
if (cost_spprc <= cost_greedy)
    memcpy(assignments, assign_spprc, ...);
else
    memcpy(assignments, assign_greedy, ...);
```

**实验结果：**

| 规模 | SPPRC 成本 | 贪心成本 | 最终选择 | CPU 求解时间 |
|------|-----------|---------|----------|-------------|
| small (37 legs) | $100,015 | **$97,495** | 贪心 | 0.08 ms |
| medium (206 legs) | **$381,838** | $387,855 | SPPRC | 2.7 ms |
| large (2,239 legs) | $7,204,218 | **$4,182,805** | 贪心 | 1,660 ms |

SPPRC 在中等规模上更好，但在大规模上因搜索空间爆炸导致候选质量下降。

---

## 4. Gemini 3.1 Pro 生成的 GPU 加速方案

### 4.1 整体架构

Gemini 生成的 CUDA 代码采用了与 CPU baseline 完全相同的**两阶段混合策略**，但将阶段 1 的 SPPRC 搬到 GPU 上并行执行：

```
solution_init():
  ├── 航段数据 → GPU constant memory
  ├── 连接图构建（CPU，上传到 GPU）
  └── 预分配候选输出缓冲区

solution_compute():
  ├── Phase 1: GPU SPPRC kernel  ←── GPU 并行
  │   └── 每个 CUDA 线程 = 一个起始航段的 SPPRC 搜索
  ├── Phase 2: Host 端贪心集合覆盖  ←── CPU 串行
  ├── Fallback: Host 端顺序贪心  ←── CPU 串行
  └── 取两者较优解
```

### 4.2 GPU Kernel 设计

```cuda
__global__ void spprc_kernel(...) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;  // 每个线程处理一个起始航段
    if (s >= c_N) return;

    Path pq[20];        // 线程局部优先队列（local memory）
    Candidate cands[20]; // 线程局部候选列表

    // 初始化起始航段的标签
    // 主循环：提取最优、扩展邻居、支配检查
    while (pq_size > 0 && num_candidates < 20) {
        // 线性扫描找最优（PQ size 仅 20，可接受）
        // 沿连接图扩展：同 duty / 新 duty 两个分支
        // 支配关系剪枝
    }

    // 结果写回 global memory
    num_cands[s] = num_candidates;
    for (...) all_candidates[s * 20 + i] = cands[i];
}
```

**关键设计决策：**

| 设计点 | 方案 | 原因 |
|--------|------|------|
| 并行粒度 | 一个线程 = 一个起始航段 | 天然独立，无通信开销 |
| 优先队列 | 线程局部数组（size=20） | 避免 global memory 竞争 |
| 约束参数 | `__constant__` memory | 所有线程共享，硬件广播 |
| 航段数据 | `__restrict__` global memory | 只读，GPU 可缓存 |
| 连接图 | 预构建 CSR，GPU global | init 阶段一次性上传 |

### 4.3 Host 端后处理

GPU kernel 完成后，候选配对被下载到 host 内存，后续处理与 CPU baseline 完全相同：

1. 展平候选列表，计算每个配对的 savings
2. `qsort` 按 savings 降序
3. 贪心覆盖：依次选择不重叠的最优配对
4. 未覆盖航段 → 单航段配对
5. 与顺序贪心的结果比较，取更优解

### 4.4 数据流

```
         ┌─────────────────┐
         │   solution_init  │
         │  (NOT timed)     │
         └────────┬────────┘
                  │ 航段数据 → GPU
                  │ 连接图 → GPU
                  ▼
         ┌─────────────────┐
         │ solution_compute │
         │   (TIMED)       │
         └────────┬────────┘
                  │
    ┌─────────────┴─────────────┐
    ▼                           ▼
┌────────┐              ┌──────────┐
│GPU SPPRC│              │CPU Greedy│
│N threads│              │Sequential│
└────┬───┘              └────┬─────┘
     │ D2H copy              │
     ▼                       │
┌──────────┐                 │
│CPU Set   │                 │
│Cover     │                 │
└────┬─────┘                 │
     │                       │
     ▼                       ▼
   cost_A                  cost_B
     │                       │
     └───────┬───────────────┘
             ▼
        min(A, B)
```

---

## 5. 实验结果

### 5.1 硬件环境

- GPU: NVIDIA L20X (Compute Capability 8.9)
- CPU: 单核性能（harness 不使用多线程）

### 5.2 Medium 规模（206 航段）详细结果

| 指标 | CPU Baseline | Gemini GPU | 加速比 |
|------|-------------|-----------|--------|
| Init 时间 | ~0 ms | 0.80 ms | — |
| Solve 时间 | 2.68 ms | 1.27 ms | **2.1x** |
| 总时间 | 2.68 ms | 2.07 ms | **1.3x** |
| Kernel 时间 | — | 0.80 ms | — |
| GPU 利用率 | — | 63.5% | — |
| Kernel 启动次数 | — | 9 | — |
| 解的正确性 | ✅ | ✅ | — |

### 5.3 跨任务对比（Gemini 3.1 Pro，Medium 规模）

| 任务 | 问题类型 | CPU 时间 | GPU 时间 | Solve 加速比 | GPU 利用率 |
|------|---------|---------|---------|-------------|-----------|
| Bellman-Ford | 图最短路 | 2,489 ms | 1.72 ms | **1,451x** | 58.4% |
| Regex Match | NFA 匹配 | 12,159 ms | 3.82 ms | **3,186x** | 98.6% |
| Crew Pairing | 组合优化 | 2.68 ms | 1.27 ms | **2.1x** | 63.5% |

### 5.4 分析

**Crew Pairing 的 GPU 加速比远低于其他任务，原因：**

1. **Medium 规模太小**：仅 206 个航段，SPPRC 的搜索空间有限，CPU 仅需 2.7ms，GPU 的 kernel launch + D2H copy 开销已占主导

2. **算法的串行瓶颈**：阶段 2（贪心集合覆盖）和 Fallback（顺序贪心）都在 CPU 上串行执行，限制了总加速比

3. **GPU 并行度不足**：只有 206 个线程（一个起始航段一个线程），远低于 GPU 的数千个 CUDA core

4. **数据传输开销**：候选配对需从 GPU 下载到 CPU 进行后处理

**对比之下：**
- Bellman-Ford 有 100K 节点的图遍历并行度
- Regex Match 有 100K 独立字符串的天然并行度
- Crew Pairing 的并行度仅在候选生成阶段，且受限于航段数

---

## 6. 关键发现与思考

### 6.1 组合优化问题的 GPU 加速困境

Crew Pairing 代表了一类**组合优化问题**的 GPU 加速挑战：

- **候选生成**（SPPRC）可以并行化 → GPU 有优势
- **候选选择**（集合覆盖）具有天然串行依赖 → GPU 无优势
- **问题规模**较小时，GPU 的启动开销 > 计算收益

这与计算密集型问题（最短路、NFA 匹配）形成鲜明对比：后者的核心计算是大规模的规则化迭代，完美适配 GPU 的 SIMT 执行模型。

### 6.2 LLM 生成 CUDA 代码的能力

Gemini 3.1 Pro 的表现值得关注：

- **正确理解了问题结构**：准确识别出两阶段策略，将 SPPRC 并行化到 GPU
- **合理的 CUDA 设计**：使用 `__constant__` memory、线程局部优先队列、`__restrict__` 指针
- **混合 CPU/GPU 架构**：保留 CPU 端的贪心作为 fallback，体现了工程判断
- **代码可编译、结果正确**：一次生成即通过编译和正确性验证

但也有明显不足：
- 未将 Phase 2 的集合覆盖并行化（理论上可用并行前缀和加速）
- 连接图在 CPU 构建后上传，未利用 GPU 并行构建
- 未考虑 shared memory 优化

### 6.3 对 OR 研究的启示

1. **GPU 加速不是万能的**：对于组合优化中串行决策占主导的阶段（如贪心选择），GPU 的优势有限
2. **问题规模是关键**：同样的算法，在 large（2,239 航段）上的 GPU 加速比远高于 medium（206 航段）
3. **LLM 可以作为 GPU 加速的"初始方案生成器"**：生成可工作的 CUDA 代码，再由人类专家调优
4. **混合 CPU/GPU 架构是实际中的最佳实践**：把适合并行的部分放 GPU，串行部分留 CPU

---

## 附录：文件结构

```
tasks/crew_pairing/
├── task.json              # 任务元数据和参数
├── gen_data.py            # CSV → ORBench 二进制格式
├── cpu_reference.c        # CPU baseline（SPPRC + 贪心混合）
├── task_io_cpu.c          # CPU I/O 适配层
├── task_io.cu             # GPU I/O 适配层
├── prompt_template.yaml   # LLM 提示模板（L1/L2）
├── solution.cu            # 手写 GPU 参考实现
└── data/
    ├── small/             # 37 航段，7 机场
    ├── medium/            # 206 航段，7 机场
    └── large/             # 2,239 航段，39 机场
```
