# ORBench

**通用 CPU→CUDA 加速 Benchmark：评测大语言模型的 GPU 编程能力**

ORBench 评测大语言模型能否将通用 CPU 代码自动改写为高效的 CUDA kernel。与现有 benchmark 不同——KernelBench 只关注 ML 算子，ComputeEval 只考察 CUDA 语法正确性——ORBench 覆盖经典算法，包括图遍历、动态规划、计算几何、组合优化等。

## 当前任务

| Task ID | 领域 | 难度 | 并行维度 | 接口模式 | GPU 瓶颈 |
|---------|------|------|----------|----------|----------|
| bellman_ford | 图算法 | ★★ | V=500K 节点 | init_compute | 同步开销 |
| collision_detection | 计算几何 | ★★★ | C=30M 对 | init_compute | 负载均衡 |
| network_rm_dp | 运筹/动态定价 | ★★★★ | S=2.8M 状态 | compute_only | 状态空间大小 |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 列出所有可用任务和模型
python run.py list

# 3. 生成测试数据
python tasks/bellman_ford/gen_data.py small tasks/bellman_ford/data/small --with-expected
python tasks/network_rm_dp/gen_data.py small tasks/network_rm_dp/data/small --with-expected

# 4. 用 LLM 生成 CUDA 代码（需要 API key，run name 自动带日期）
export LLM_API_KEY="your-api-key"
python run.py generate --task bellman_ford --model claude-sonnet-4-20250514 --level 2 --samples 3

# 5. 批量多模型生成
python run.py generate-batch --models claude-sonnet-4 gpt-4o --tasks bellman_ford --levels 2 --samples 5

# 6. 评测（需要 NVIDIA GPU）
python run.py eval --run claude-sonnet-4_l2_20260319 --arch sm_89

# 7. 多轮 Agent 实验（生成→评测→反馈→改进）
python run.py agent-multiturn --model gemini-3.1-pro-preview --task network_rm_dp --level 2 --turns 10

# 8. 查看结果
python run.py analyze --run claude-sonnet-4_l2_20260319

# 9. 跨模型对比
python run.py compare --runs claude-sonnet-4_l2_20260319 gpt-4o_l2_20260319
```

## 项目结构

```
ORBench/
├── run.py                      # 统一 CLI 入口
├── models.yaml                 # LLM 提供商 & 模型配置
├── config.yaml                 # 全局配置（GPU 架构、超时等）
├── requirements.txt
│
├── framework/                  # 评测框架
│   ├── task.py                 # 任务加载、TaskConfig（含 interface_mode）
│   ├── generate.py             # 调 LLM API 生成 .cu 代码
│   ├── generate_prompt.py      # 从 prompt_template.yaml 组装分级 prompt
│   ├── compile.py              # nvcc 编译（自动传 -DORBENCH_COMPUTE_ONLY）
│   ├── validate.py             # 正确性验证（对比 expected_output.txt）
│   ├── benchmark.py            # CUDA Event 计时 + nsys 集成
│   ├── profile.py              # nsys 自动化：profile → CSV → 分析
│   ├── batch_eval.py           # 多 GPU 批量调度（spawn 进程隔离）
│   ├── analyze.py              # 结果汇总
│   ├── harness_common.h        # 通用 benchmark 骨架（支持双接口模式）
│   ├── harness_gpu.cu          # GPU 计时封装（CUDA Event）
│   ├── harness_cpu.c           # CPU 计时封装（clock_gettime）
│   ├── orbench_io.h            # 二进制 input.bin 读取（C）
│   ├── orbench_io_py.py        # 二进制 input.bin 写入（Python）
│   ├── llm/
│   │   ├── registry.py         # 多 LLM 提供商注册、客户端管理
│   │   └── scheduler.py        # 批量生成调度（并发控制、断点续跑）
│   └── agent/
│       ├── multiturn.py        # 多轮 Agent 流水线
│       ├── prompts.py          # 反馈 prompt 构建
│       └── plot_metrics.py     # Agent 指标可视化（CSV + PNG）
│
├── tasks/                      # 任务定义
│   └── <task_id>/
│       ├── task.json           # 元信息（难度、规模、容差、interface_mode）
│       ├── prompt_template.yaml# Prompt 模板（L1/L2/L3 分级 hints）
│       ├── cpu_reference.c     # CPU 基准（纯计算，无 I/O）
│       ├── task_io.cu          # GPU I/O 适配层（harness ↔ solution 桥接）
│       ├── task_io_cpu.c       # CPU I/O 适配层
│       ├── gen_data.py         # 数据生成 + CPU 求解 expected output
│       └── data/{small,medium,large}/
│           ├── input.bin               # 二进制输入（ORBench 格式）
│           ├── requests.txt            # 请求描述
│           ├── expected_output.txt     # CPU 参考答案
│           ├── cpu_time_ms.txt         # CPU 基准耗时
│           └── timing.json             # 最近一次运行的计时数据
│
├── runs/                       # 实验结果（自动带日期）
│   └── <model>_l<level>_<date>/
│       └── <task_id>/
│           ├── sample_0.cu             # LLM 生成的代码
│           └── ...
│
└── cache/                      # 编译缓存
```

## 三层架构

```
harness (harness_common.h)      — 通用：计时、warmup、validate 控制
  ↓
task_io (task_io.cu)            — 任务特定：解析输入、格式化输出
  ↓
solution (LLM 生成)             — 纯计算：LLM 只写算法逻辑
```

LLM 只需实现 `solution_init` + `solution_compute`（init_compute 模式）或 `solution_compute` + `solution_free`（compute_only 模式），无需处理文件 I/O、计时逻辑。

## 双接口模式

| 模式 | 适用场景 | 计时范围 | 防作弊 |
|------|----------|----------|--------|
| `init_compute` | 输入数据大（>1MB） | 只计时 compute | 允许 init 预处理 |
| `compute_only` | 输入数据小（<1MB） | 计时 setup+compute | 防止把计算藏进 init |

通过 `task.json` 的 `"interface_mode"` 字段控制，编译时自动传 `-DORBENCH_COMPUTE_ONLY` 宏。

## Prompt 分级

| Level | 包含内容 | 目标 |
|-------|----------|------|
| L1 | 任务描述 + 接口 + 算法背景 + CPU 代码 + 详细 GPU 优化提示 | 测试代码实现能力 |
| L2 | 任务描述 + 接口 + CPU 代码 + 简要提示 | 测试优化策略选择 |
| L3 | 任务描述 + 接口 + CPU 代码 | 测试独立分析能力 |

Prompt 由 `prompt_template.yaml` + `generate_prompt.py` 自动组装，CPU reference 代码自动注入。

## 评测指标

| 指标 | 说明 | 来源 |
|------|------|------|
| 编译通过率 | 生成代码能否 nvcc 编译 | nvcc |
| 正确率 | 所有输入规模上结果正确 | 对比 expected_output.txt |
| total_ms | init + solve 总耗时 | timing.json |
| 端到端加速比 | CPU 时间 / GPU 端到端时间 | CUDA Event |
| 纯 Kernel 加速比 | CPU 时间 / 纯 kernel 时间 | nsys trace |
| GPU 利用率 | kernel 时间 / 端到端时间 | nsys trace |

## 多轮 Agent 模式

```
Turn 0: 基础 prompt → LLM 生成代码 → 编译 → 评测 → nsys profile
Turn 1: 上轮代码 + 评测反馈 + nsys 分析 → LLM 改进 → 重新评测
Turn N: 持续迭代优化
```

每轮自动记录 `total_ms`、`kernel_time_ms`、`speedup_e2e` 等指标，生成 `agent_metrics.csv` 和 `agent_metrics.png` 趋势图。

## 新增任务

1. 在 `tasks/` 下创建目录
2. 编写 `task.json`（设置 `interface_mode`、难度、输入规模、容差）
3. 编写 `prompt_template.yaml`（任务描述、接口、分级 hints）
4. 编写 `cpu_reference.c`（纯计算，无文件 I/O）
5. 编写 `task_io.cu` 和 `task_io_cpu.c`（I/O 适配层）
6. 编写 `gen_data.py`（生成 input.bin + expected_output.txt）
7. 运行 `gen_data.py` 生成 small/medium/large 数据

## 环境要求

- Python 3.10+
- CUDA Toolkit 12.0+（nvcc）
- NVIDIA GPU
- nsys（推荐，用于 kernel 级分析）
- `pip install -r requirements.txt`
