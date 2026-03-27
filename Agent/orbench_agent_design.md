# ORBench Auto-Add-Task Agent — 架构设计文档

## 1. 概览

本系统是一个两阶段 Agent 流水线，用于将 OR（运筹学）领域的论文和代码自动转化为 ORBench 的 benchmark 任务。

```
input_folder/          Agent 1                Agent 2              tasks/{task_id}/
├── paper.pdf    →  Feasibility  →  suitable?  →  Task      →   ├── task.json
├── code.py         Checker         ↓ yes         Assembler       ├── prompt_template.yaml
└── ...                             ↓ no                          ├── cpu_reference.c
                                  报告原因                         ├── gen_data.py
                                  并退出                           ├── task_io.cu
                                                                  └── task_io_cpu.c
```

## 2. Agent 1 — 可行性检查器 (Feasibility Checker)

### 2.1 职责

读取论文和代码，判断该 OR 问题是否适合加入 ORBench。

### 2.2 五项准入标准

| # | 标准 | 说明 | 典型正例 | 典型反例 |
|---|------|------|----------|----------|
| 1 | **充足的并行度** | 核心计算有数据并行或任务并行，可映射到数千个GPU线程 | 逐边松弛、逐元素矩阵运算、逐字符串匹配、逐状态DP | 顺序单纯形法pivot、纯串行分支定界 |
| 2 | **充足的规模** | large 尺寸在 CPU 上至少运行数毫秒 | V=500K, E=2.5M 的图算法 | 100个变量的小LP |
| 3 | **清晰的I/O** | 输入输出可表达为 typed tensors + 整数参数 | CSR图、矩阵、字符串数组 | 复杂对象图、需要动态数据结构 |
| 4 | **确定性CPU参考实现** | 纯C实现，50-300行，无外部库依赖 | Bellman-Ford、SAT碰撞检测 | 需要CPLEX/Gurobi的MIP求解 |
| 5 | **有意义的加速潜力** | 良好的CUDA实现可达 5-10x+ 加速 | 大规模并行DP、批量图遍历 | I/O密集型、计算量极小 |

### 2.3 常见拒绝原因

- **并行度不好** (`low_parallelism`): 算法有长依赖链，无法有效并行化
- **规模太小** (`too_small`): 即使最大的合理实例也在CPU上 <1ms 完成
- **需要外部求解器** (`external_solver`): CPU参考需要商业求解器
- **非确定性** (`non_deterministic`): 结果取决于线程调度
- **I/O密集** (`io_bound`): 计算量相对数据加载微不足道

### 2.4 输出

```python
@dataclass
class FeasibilityResult:
    suitable: bool                        # 是否适合
    task_id: str                          # snake_case 标识符
    task_name: str                        # 人类可读名称
    category: str                         # 分类
    problem_summary: str                  # 问题摘要
    parallelism_analysis: str             # 并行度分析
    scale_analysis: str                   # 规模分析
    rejection_reasons: list[str]          # 拒绝原因（适合时为空）
    gpu_optimization_points: list[str]    # GPU优化切入点
    suggested_sizes: dict                 # small/medium/large 参数
    interface_mode: str                   # "init_compute" or "compute_only"
    difficulty: int                       # 1-4
    tags: list[str]                       # 标签
    algorithm_description: str            # 算法详述
    input_data_description: str           # 输入数据描述
    output_data_description: str          # 输出数据描述
    reference_code_snippet: str           # 论文中的参考代码
```

## 3. Agent 2 — 任务组装器 (Task Assembler)

### 3.1 职责

接收 Agent 1 的分析结果和原始文件，生成完整的 ORBench 任务文件夹。

### 3.2 生成的 6 个文件

#### 3.2.1 `task.json` — 任务元数据

定义任务ID、名称、分类、难度、输入尺寸、正确性检验配置和计时参数。

```json
{
  "task_id": "example",
  "name": "Example Task",
  "category": "graph",
  "difficulty": 2,
  "tags": ["CSR_graph", "sssp"],
  "interface_mode": "init_compute",
  "input_sizes": {
    "small":  {"V": 1000,   "E": 5000,   "seed": 42},
    "medium": {"V": 100000, "E": 500000, "seed": 42},
    "large":  {"V": 500000, "E": 2500000,"seed": 42}
  },
  "correctness": {"mode": "numerical", "atol": 0.01, "rtol": 0.01},
  "timing": {"warmup": 3, "trials": 10, "timeout": 180},
  "gpu_optimization_points": ["persistent_data", "coalesced_access"]
}
```

#### 3.2.2 `prompt_template.yaml` — LLM 提示模板

YAML 格式，包含以下字段：

| 字段 | 用途 | 包含级别 |
|------|------|----------|
| `task_description` | 问题描述 | L1, L2, L3 |
| `interface` | 函数签名 | L1, L2, L3 |
| `input_size_notes` | 尺寸补充说明 | L1, L2, L3 |
| `output_contract` | 输出格式约定 | L1, L2, L3 |
| `algorithm_background` | 算法背景 | L1 only |
| `hints_l2` | 简要提示 | L2 only |
| `hints_l1` | 详细优化指南 | L1 only |

框架自动注入：通用约束、输入尺寸表（来自task.json）、cpu_reference.c。

#### 3.2.3 `cpu_reference.c` — CPU 基线实现

```c
// cpu_reference.c — {task_name} CPU baseline
// 实现 solution_init / solution_compute
// 无文件I/O，所有I/O由 task_io 处理

#include <stdlib.h>
#include <string.h>

// init_compute 模式：
void solution_init(/* 接收 host 数据，分配工作缓冲区 */);
void solution_compute(/* 核心计算，可被重复调用，幂等 */);

// 或 compute_only 模式：
void solution_compute(/* 接收所有数据+输出缓冲区 */);
void solution_free(void);  // compute_only 需要
```

#### 3.2.4 `gen_data.py` — 数据生成脚本

```python
#!/usr/bin/env python3
"""gen_data.py — 生成 input.bin + requests.txt + expected_output.txt"""

from framework.orbench_io_py import write_input_bin

SIZES = {
    "small":  { ... },
    "medium": { ... },
    "large":  { ... },
}

def main():
    # 1. 生成数据（numpy）
    # 2. write_input_bin(path, tensors=[...], params={...})
    # 3. 写 requests.txt
    # 4. 如果 --with-expected：编译运行 CPU baseline，写 expected_output.txt
```

#### 3.2.5 `task_io.cu` — GPU I/O 适配层

```
harness_gpu.cu → task_io.cu → solution.cu (LLM 生成)
                     ↓
            task_setup()     : 解析 TaskData → 调用 solution_init
            task_run()       : 调用 solution_compute
            task_write_output(): 写 output.txt
            task_cleanup()   : 释放资源
```

#### 3.2.6 `task_io_cpu.c` — CPU I/O 适配层

与 task_io.cu 逻辑相同，但纯 C（无 cuda_runtime.h）。

## 4. 数据流与二进制格式

### 4.1 input.bin 格式

```
FileHeader    (32 bytes)  : magic="ORBENCH\0", version=1, num_tensors, num_params, data_offset
TensorDesc[]  (64 bytes each) : name, dtype, count, offset, size_bytes
ParamEntry[]  (48 bytes each) : key, value (int64)
[padding to 64B alignment]
Raw tensor data
```

支持的 dtype: INT32 (0), FLOAT32 (1), FLOAT64 (2)

### 4.2 Python 写入 API

```python
from framework.orbench_io_py import write_input_bin

write_input_bin(
    "output/input.bin",
    tensors=[
        ("row_offsets", "int32",   row_offsets_array),
        ("weights",     "float32", weights_array),
    ],
    params={"V": 1000, "E": 5000},
)
```

## 5. 使用方法

### 5.1 基本用法

```bash
# 准备输入文件夹
mkdir paper_folder
cp my_paper.pdf paper_folder/
cp reference_code.py paper_folder/

# 运行 Agent 流水线
python orbench_add_task_agent.py paper_folder/ --orbench-root /path/to/orbench
```

### 5.2 命令行参数

| 参数 | 说明 |
|------|------|
| `input_folder` | 包含论文和代码的文件夹 |
| `--orbench-root` | ORBench 根目录（默认: 当前目录） |
| `--model` | Anthropic 模型名（默认: claude-sonnet-4-20250514） |
| `--skip-feasibility` | 跳过 Agent 1，直接生成任务 |
| `--force` | 即使 Agent 1 判定不适合也强制生成 |
| `--dry-run` | 只运行分析，不写文件 |

### 5.3 生成后的检查步骤

```bash
cd tasks/{task_id}

# 1. 生成测试数据
python gen_data.py small data/small --with-expected

# 2. 编译 CPU baseline
gcc -O2 -I ../../framework/ \
    ../../framework/harness_cpu.c task_io_cpu.c cpu_reference.c \
    -o solution_cpu -lm

# 3. 运行验证
./solution_cpu data/small --validate

# 4. 检查输出
diff data/small/output.txt data/small/expected_output.txt
```

## 6. 设计决策与考量

### 6.1 为什么分两个 Agent？

- **关注点分离**: Agent 1 专注于"能不能做"的判断，Agent 2 专注于"怎么做"的执行
- **提前终止**: 不合适的问题在 Agent 1 就被过滤，节省 Agent 2 的 token 开销
- **可调试性**: 每个阶段有独立的输出，便于定位问题

### 6.2 interface_mode 选择

- **init_compute** (默认): 适合大多数任务。init 做一次性的数据传输，compute 只做计算。这允许 LLM 在 init 中做 cudaMalloc + H2D。
- **compute_only**: 用于防止将计算藏在 init 中的场景（如 CUDA Graph 录制）。每次 trial 都重新 setup + compute + cleanup。

### 6.3 正确性检验模式

- **exact**: 整数结果，必须完全一致（如 collision counts, regex match results）
- **numerical**: 浮点结果，允许容差（atol + rtol）

## 7. 扩展性

### 7.1 添加新的拒绝原因

在 Agent 1 的 system prompt 中添加新的不适合条件即可。

### 7.2 支持新的数据类型

1. 在 `orbench_io_py.py` 中添加新的 dtype 常量
2. 在 `orbench_io.h` 中更新 `_orbench_dtype_size`
3. 在 Agent 2 的 system prompt 中说明新类型

### 7.3 批量处理

```bash
# 批量处理多个论文文件夹
for folder in papers/*/; do
    python orbench_add_task_agent.py "$folder" --orbench-root . 2>&1 | tee "${folder}/agent.log"
done
```
