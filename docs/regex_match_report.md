# NFA 正则表达式匹配的 GPU 加速实验报告

## 1. 问题背景

### 1.1 什么是 NFA 正则匹配

给定一个**非确定性有限自动机（NFA）**和 N 个输入字符串，判断每个字符串是否被该 NFA 接受（full match）。这是正则表达式引擎的核心计算。

与常见的回溯式正则引擎（如 Python `re`、PCRE）不同，本任务使用 **Thompson NFA 模拟算法**：维护一个"活跃状态集合"，每读入一个字符就做一次状态转移。该算法保证 O(n×m) 的时间复杂度（n = 字符串长度，m = NFA 状态数），不会出现回溯引擎的指数级病态行为。

### 1.2 为什么适合 GPU 加速

NFA 匹配具有两个层次的并行性：

| 并行层次 | 说明 | 并行度 |
|----------|------|--------|
| **外层：字符串并行** | N 个字符串完全独立，可同时匹配 | N = 100K ~ 1M |
| **内层：状态集合并行** | 活跃状态集合可用 bitmask 表示，转移变为位运算 | states / 64 |

这与 Bellman-Ford 图算法形成了有趣的对比：

```
Bellman-Ford:  图的节点上传播"距离"值，每轮松弛边
NFA 匹配:     自动机的状态上传播"活跃"标记，每个字符做状态转移
```

两者都是"图上的迭代传播"，但 NFA 匹配的外层并行度（字符串数 N）远高于 Bellman-Ford 的节点并行度，理论加速比更大。

### 1.3 与 OR 的联系

NFA 匹配在运筹优化领域有广泛应用：
- **约束编程**中的正则语言约束（regular constraint）
- **序列模式挖掘**中的模式匹配
- **网络流量分析**中的深度包检测（DPI）
- **文本分析**中的大规模日志过滤

当需要对海量字符串做同一模式的匹配时，GPU 加速具有重要的实际价值。

---

## 2. 输入输出格式

### 2.1 NFA 表示

NFA 使用 **CSR（Compressed Sparse Row）格式**的转移表，与图算法的邻接表同构：

| 数据结构 | 类型 | 大小 | 说明 |
|----------|------|------|------|
| trans_offsets | int32 | num_states × num_symbols + 1 | CSR 偏移：(state, symbol) → 转移目标范围 |
| trans_targets | int32 | num_trans | CSR 目标：转移目标状态列表 |
| eps_offsets | int32 | num_states + 1 | ε-转移 CSR 偏移 |
| eps_targets | int32 | num_eps | ε-转移目标状态列表 |
| is_accept | int32 | num_states | 1 = 接受状态，0 = 非接受 |

**与图 CSR 的类比：**
```
Bellman-Ford CSR:  row_offsets[V+1], col_indices[E], weights[E]
NFA 转移 CSR:      trans_offsets[S×A+1], trans_targets[T]
```

### 2.2 输入字符串

N 个字符串被打包为两个数组：

| 数据 | 类型 | 大小 | 说明 |
|------|------|------|------|
| str_offsets | int32 | N + 1 | 每个字符串在 str_data 中的起始位置 |
| str_data | int32 | total_chars | 所有字符串拼接，值为 symbol ID（0 ~ num_symbols-1） |

### 2.3 C 函数接口

```c
// 初始化：接收 NFA 和字符串数据（仅调用一次）
void solution_init(
    int num_states, int num_symbols, int start_state,
    int num_strings, int total_chars,
    const int* trans_offsets, const int* trans_targets,
    const int* eps_offsets, const int* eps_targets,
    const int* is_accept,
    const int* str_offsets, const int* str_data);

// 匹配：对所有字符串进行 NFA 匹配（多次调用，计时）
void solution_compute(int num_strings, int* results);
// results[i] = 1 表示字符串 i 被接受，0 表示拒绝
```

### 2.4 数据规模

| 规模 | 字符串数 | 平均长度 | NFA 状态数 | 字母表 | 总字符数 |
|------|---------|---------|-----------|--------|---------|
| small | 1,000 | 100 | 32 | 26 (a-z) | ~73K |
| medium | 100,000 | 200 | 64 | 26 | ~15.5M |
| large | 1,000,000 | 500 | 128 | 26 | ~估计 500M |

NFA 由随机生成器构建，控制转移密度（~1.5 条/state/symbol）和 ε-转移密度（~0.2 条/state），匹配比例约 40-75%。

---

## 3. CPU Baseline 设计

### 3.1 Thompson NFA 模拟

CPU baseline 采用经典的 **Thompson NFA 模拟算法**，对每个字符串串行执行：

```
对每个字符串 s[0..n-1]:
  1. current_states = ε-closure({start_state})
  2. 对每个字符 c = s[i]:
     a. next_states = ∅
     b. 对 current_states 中的每个状态 q:
        next_states ∪= transitions[q][c]
     c. current_states = ε-closure(next_states)
     d. 若 current_states 为空 → 提前终止（不匹配）
  3. 若 current_states ∩ accept_states ≠ ∅ → 匹配
```

### 3.2 数据结构

```c
// 活跃状态集合用 char 数组表示（MAX_STATES = 256）
char current[MAX_STATES];  // current[s] = 1 表示状态 s 活跃
char next[MAX_STATES];
```

### 3.3 ε-闭包计算

使用 BFS 从当前活跃状态出发，沿 ε-边扩展：

```c
static void epsilon_closure(const int* eps_offsets, const int* eps_targets,
                            int num_states, char* active) {
    int queue[MAX_STATES];
    // 将所有 active[s]==1 的状态入队
    // BFS 沿 eps_offsets/eps_targets 扩展
    // 直到队列为空
}
```

### 3.4 主循环

```c
void solution_compute(int num_strings, int* results) {
    for (int i = 0; i < num_strings; i++) {
        int offset = str_offsets[i];
        int len = str_offsets[i + 1] - offset;
        results[i] = nfa_match_one(&str_data[offset], len);
    }
}
```

**复杂度**：O(N × L × S²)，其中 N = 字符串数，L = 平均长度，S = NFA 状态数。对于 medium 规模（100K × 200 × 64），约 1.28 × 10⁹ 次操作，实测 CPU 时间 **12.16 秒**。

---

## 4. Gemini 3.1 Pro 生成的 GPU 加速方案

### 4.1 核心创新：Bitmask 预计算 + 模板化 Kernel

Gemini 的方案展现了对 NFA 匹配 GPU 加速的**深度理解**，包含三个关键优化：

#### 优化 1：ε-闭包预计算

在 `solution_init` 中，对每个状态预计算其 ε-闭包，存为 bitmask：

```c
// eps_closure[s] = 从状态 s 出发经 ε-边可达的所有状态（bitmask）
uint64_t* eps_closure = calloc(num_states * num_words, sizeof(uint64_t));
// 对每个状态 s 做 BFS，结果编码为 bitmask
```

#### 优化 2：转移表 bitmask 化

将 CSR 转移表转换为 **bitmask 转移表**，并融合 ε-闭包：

```c
// trans_mask[symbol][state] = 从状态 state 读入 symbol 后
//                             经转移+ε-闭包可达的所有状态（bitmask）
for (int c = 0; c < num_symbols; c++) {
    for (int s = 0; s < num_states; s++) {
        for each target t in transitions[s][c]:
            trans_mask[c][s] |= eps_closure[t];  // 融合 ε-闭包！
    }
}
```

**这是最关键的优化**：将运行时的"转移 + ε-闭包"两步操作，预计算为单次 bitmask OR 查表，消除了运行时的 BFS 开销。

#### 优化 3：模板化 Kernel 按状态数分派

根据 NFA 状态数，在编译期确定 bitmask 宽度：

```cuda
template <int WORDS>  // WORDS = ceil(num_states / 64)
__global__ void match_kernel(...) {
    uint64_t active[WORDS];  // 编译期确定大小 → 寄存器分配

    for (每个字符 c) {
        uint64_t new_active[WORDS] = {0};
        for (int w = 0; w < WORDS; w++) {
            uint64_t bits = active[w];
            while (bits) {
                int s = __ffsll(bits) - 1;  // 找最低位
                bits &= bits - 1;            // 清除最低位
                // 查预计算的 bitmask 转移表
                new_active[k] |= trans_mask[(c * S + s) * WORDS + k];
            }
        }
        active = new_active;
    }
}
```

运行时根据 `num_words` 分派到正确的模板实例：

```c
if      (g_num_words == 1) match_kernel<1><<<...>>>(...);
else if (g_num_words == 2) match_kernel<2><<<...>>>(...);
// ... 支持到 WORDS=8（512 个状态）
```

### 4.2 整体架构

```
solution_init()（不计时）:
  ├── 1. 对每个状态预计算 ε-闭包（CPU BFS → bitmask）
  ├── 2. 构建 bitmask 转移表：trans_mask[symbol][state] = 目标 bitmask
  │      （融合了转移 + ε-闭包，消除运行时 BFS）
  ├── 3. 预计算 initial_mask = ε-closure({start_state})
  ├── 4. 预计算 accept_mask
  ├── 5. 字符串数据 int32 → uint16 转换（GPU kernel）
  └── 6. 所有预计算表上传 GPU

solution_compute()（计时）:
  ├── 1. 根据 num_words 分派模板 kernel
  │      └── N 个线程，每线程处理一个字符串
  │          ├── active = initial_mask
  │          ├── for each char: active = Σ trans_mask[char][s] for s ∈ active
  │          └── result = (active & accept_mask) != 0
  └── 2. D2H 拷贝 results
```

### 4.3 内存布局

| 数据 | 位置 | 大小 (medium, 64 states) | 说明 |
|------|------|--------------------------|------|
| trans_mask | Global memory | 26 × 64 × 8 = 13.3 KB | bitmask 转移表 |
| initial_mask | Global memory | 8 B | 起始状态 bitmask |
| accept_mask | Global memory | 8 B | 接受状态 bitmask |
| str_data | Global memory | 15.5 MB (uint16) | 字符串数据 |
| str_offsets | Global memory | 400 KB | 字符串偏移 |
| active[WORDS] | 寄存器 | 8 B / thread | 线程局部活跃状态 |

### 4.4 设计决策分析

| 设计点 | Gemini 的选择 | 分析 |
|--------|--------------|------|
| 并行粒度 | 一个线程 = 一个字符串 | 最大化外层并行度（100K+ 线程） |
| 状态集合表示 | uint64_t[WORDS] bitmask | 集合运算 → 位运算，O(1) per word |
| ε-闭包 | Init 阶段预计算，融入转移表 | 消除运行时 BFS，空间换时间 |
| 模板分派 | 编译期确定 WORDS | 循环可展开，bitmask 存寄存器 |
| 字符串存储 | int32 → uint16 压缩 | 减少显存带宽，GPU kernel 预处理 |
| `#pragma unroll` | 内层循环全部展开 | 减少分支，最大化 ILP |

---

## 5. 实验结果

### 5.1 Medium 规模详细结果

| 指标 | CPU Baseline | Gemini GPU | 加速比 |
|------|-------------|-----------|--------|
| **Init 时间** | ~0 ms | 17.46 ms | — |
| **Solve 时间** | 12,159 ms | 3.82 ms | **3,186x** |
| **总时间** | 12,159 ms | 21.28 ms | **571x** |
| Kernel 时间 | — | 3.76 ms | — |
| GPU 利用率 | — | 98.6% | — |
| Kernel 启动次数 | — | 10 | — |
| 正确性 | ✅ | ✅ | — |

### 5.2 时间分解

```
CPU (12,159 ms total):
  ├── Init: ~0 ms（只是存指针）
  └── Solve: 12,159 ms（串行遍历 100K 字符串 × Thompson 模拟）

GPU (21.28 ms total):
  ├── Init: 17.46 ms
  │   ├── ε-闭包预计算（CPU BFS × 64 states）
  │   ├── bitmask 转移表构建
  │   ├── int32→uint16 转换 kernel
  │   └── H2D 上传（~15.5 MB 字符串 + 转移表）
  └── Solve: 3.82 ms
      ├── match_kernel<1>: 3.76 ms（100K 线程并行匹配）
      └── D2H: 0.06 ms（100K × 4B results）
```

**关键观察**：
- Solve 阶段加速 **3,186 倍**，几乎达到理论上限
- GPU 利用率 **98.6%** — 几乎无空闲
- Init 开销（17.46 ms）远大于 Solve（3.82 ms），但只需执行一次
- 如果算总时间（Init + Solve），加速比降为 571 倍

### 5.3 跨任务对比

| 任务 | 问题类型 | 并行维度 | CPU 时间 | GPU Solve | **Solve 加速比** | GPU 利用率 |
|------|---------|---------|---------|-----------|-----------------|-----------|
| **Regex Match** | 自动机模拟 | 100K 字符串 | 12.16 s | 3.82 ms | **3,186x** | 98.6% |
| Bellman-Ford | 图最短路 | 100K 节点 | 2.49 s | 1.72 ms | **1,451x** | 58.4% |
| Crew Pairing | 组合优化 | 206 航段 | 2.68 ms | 1.27 ms | **2.1x** | 63.5% |

### 5.4 为什么 Regex Match 加速比最高

1. **外层并行度极高**：100K 个完全独立的字符串，每个线程零通信
2. **计算规则化**：每个线程执行相同的 bitmask 操作序列，无分支发散（warp divergence 最小）
3. **Bitmask 化消除了串行瓶颈**：集合运算（并集、交集、空集检测）变为 O(1) 位运算
4. **ε-闭包预计算**：将运行时的 BFS 转化为 Init 阶段的一次性开销
5. **数据量适中**：转移表仅 13 KB，大部分可缓存在 L1/L2

---

## 6. 算法对比：CPU vs GPU

### 6.1 核心差异

| 维度 | CPU Baseline | Gemini GPU |
|------|-------------|-----------|
| 状态集合表示 | `char active[256]`（数组） | `uint64_t active[WORDS]`（bitmask） |
| 集合并集 | `for` 循环逐元素 OR | `\|=` 位运算（1 条指令/word） |
| ε-闭包 | 运行时 BFS（每个字符执行一次） | 预计算融入转移表（运行时零开销） |
| 转移查表 | CSR 间接寻址（cache-unfriendly） | bitmask 直接查表（连续内存） |
| 并行方式 | 串行遍历 N 个字符串 | N 个 CUDA 线程同时执行 |
| 字符串遍历 | 顺序读取 | Global memory coalesced read |

### 6.2 算法复杂度

```
CPU:  O(N × L × S)    每字符：遍历活跃状态 × 查 CSR 转移 × BFS ε-闭包
GPU:  O(N × L × W)    每字符：遍历 W 个 uint64 word × 位扫描 + bitmask OR
                       其中 W = ceil(S/64)，且 N 个字符串并行

对 medium (N=100K, L=200, S=64, W=1):
  CPU: 100K × 200 × 64 = 1.28 × 10⁹ operations (串行)
  GPU: 100K × 200 × 1  = 2 × 10⁷ operations (并行，实际受带宽限制)
```

---

## 7. 关键发现

### 7.1 Bitmask 是 NFA 匹配 GPU 加速的核心

从 `char[256]` 到 `uint64_t[WORDS]` 的转换不仅仅是数据压缩，而是**算法范式的改变**：

| 操作 | 数组实现 | Bitmask 实现 |
|------|---------|-------------|
| 集合并集 | O(S) 循环 | O(W) 位运算 |
| 空集检测 | O(S) 循环 | O(W) 比较 |
| 成员检测 | O(1) 查表 | O(1) 位测试 |
| 遍历活跃状态 | O(S) 全扫描 | `__ffsll` + 位清除 |

当 S ≤ 64 时（如 medium），W=1，所有集合运算都是**单条机器指令**。

### 7.2 预计算是空间-时间权衡的典范

Gemini 的 ε-闭包融合策略：

```
运行时成本:  消除（从 O(S²) BFS/字符 → 0）
预计算成本:  O(S² × A) 一次性（Init 阶段）
空间成本:    S × A × W × 8 bytes（转移表大小）
```

对 medium（S=64, A=26, W=1）：转移表仅 13 KB，轻松放入 GPU 缓存。这是典型的**用 Init 阶段的预计算换取 Solve 阶段的零开销**。

### 7.3 模板元编程提升了硬件利用率

`template <int WORDS>` 使得编译器在编译期知道循环次数：
- `WORDS=1`：内层循环完全消除，直接展开为标量操作
- `WORDS=2`：展开为两组独立的 64-bit 操作
- 寄存器分配确定性：`active[WORDS]` 分配到寄存器而非 local memory

这解释了 98.6% 的 GPU 利用率——几乎没有指令级别的浪费。

### 7.4 LLM 的算法理解能力

Gemini 3.1 Pro 在这个任务上展现了令人印象深刻的能力：

1. **正确识别了 bitmask 优化**：从 prompt 中的提示出发，生成了完整的 bitmask 转移表预计算
2. **ε-闭包融合**：自主发现将 ε-闭包预计算并融入转移表的优化——这超出了 prompt 的直接提示
3. **模板分派**：根据状态数自动选择合适的 bitmask 宽度
4. **数据类型优化**：主动将 int32 字符压缩为 uint16，减少带宽
5. **代码质量**：一次生成即通过编译和正确性验证，3,186 倍加速

---

## 附录 A：文件结构

```
tasks/regex_match/
├── task.json              # 任务元数据
├── gen_data.py            # 随机 NFA + 字符串生成 → input.bin
├── cpu_reference.c        # CPU baseline（Thompson NFA 模拟）
├── task_io_cpu.c          # CPU I/O 适配层
├── task_io.cu             # GPU I/O 适配层
├── prompt_template.yaml   # LLM 提示模板（含 bitmask 提示）
└── data/
    ├── small/             # 1K 字符串，32 states
    └── medium/            # 100K 字符串，64 states
```

## 附录 B：NFA CSR 与图 CSR 的结构对比

```
图 (Bellman-Ford):
  row_offsets[V+1]     ←→  trans_offsets[S×A+1]     NFA 转移
  col_indices[E]       ←→  trans_targets[T]          转移目标
  weights[E]           ←→  (无权重)                  NFA 无权

  传播: dist[v] = min(dist[u] + w)     ←→  states |= trans_mask[s][c]
  收敛: 全局 flag (迭代)                ←→  到达字符串末尾 (固定步数)
```

两者使用了相同的 CSR 存储格式，但计算模式不同：Bellman-Ford 是 min-plus 半环上的迭代松弛，NFA 匹配是 Boolean 半环上的状态传播。这一结构同源性使得两个任务的 GPU 优化策略有很多共通之处（持久化图数据、边并行/状态并行、提前终止等）。
