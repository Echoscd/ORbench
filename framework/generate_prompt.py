"""
generate_prompt.py - Assemble LLM prompts from task template + CPU reference.

Each task provides:
  - prompt_template.yaml   (task description, interface, hints per level)
  - cpu_reference.c        (CPU baseline, auto-embedded)

This module combines them with generic constraints and input-size tables
(from task.json) to produce a complete prompt for any level (1, 2, 3).

Level semantics:
  L1 (easiest):  interface + constraints + sizes + CPU ref + algorithm background + detailed hints
  L2 (medium):   interface + constraints + sizes + CPU ref + brief hints
  L3 (hardest):  interface + constraints + sizes + CPU ref  (no hints)
"""

import os
import yaml
from typing import Optional

from .task import load_task, get_task_dir, TaskConfig


# ═══════════════════════════════════════════════════════════════
#  Generic constraint text (same for every task)
# ═══════════════════════════════════════════════════════════════

GENERIC_CONSTRAINTS_INIT_COMPUTE = """\
- All functions use `extern "C"` linkage
- All parameters to `solution_compute` are **host pointers**; you manage H2D/D2H yourself
- Do NOT call `cudaMalloc` inside `solution_compute` (it is called repeatedly); allocate in `solution_init`
- `solution_free` is optional (the framework provides a weak no-op default)
- Do NOT use any file I/O (`fopen`, `printf`, `fprintf`, etc.)
- Unreachable / invalid results use `1e30f`"""

GENERIC_CONSTRAINTS_COMPUTE_ONLY = """\
- `solution_compute` uses `extern "C"` linkage
- All parameters are **host pointers**; you manage H2D/D2H yourself
- Do ALL work inside `solution_compute` (including `cudaMalloc`, H2D, kernels, D2H, and any cleanup)
- `solution_compute` is called multiple times (warmup + timed trials); it must be idempotent
- Do NOT use any file I/O (`fopen`, `printf`, `fprintf`, etc.)
- Unreachable / invalid results use `1e30f`"""

RESPONSE_FORMAT_INIT_COMPUTE = """\
Return your complete CUDA implementation in a **single** fenced code block:

```c
// your code here
```

Do NOT include any explanation outside the code block. The code block must contain the full,
compilable `.cu` source with both `solution_init` and `solution_compute`."""

RESPONSE_FORMAT_COMPUTE_ONLY = """\
Return your complete CUDA implementation in a **single** fenced code block:

```c
// your code here
```

Do NOT include any explanation outside the code block. The code block must contain the full,
compilable `.cu` source with a single `solution_compute` function."""


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def _load_template(task_id: str) -> dict:
    """Load prompt_template.yaml for a task."""
    task_dir = get_task_dir(task_id)
    tmpl_path = os.path.join(task_dir, "prompt_template.yaml")
    if not os.path.exists(tmpl_path):
        raise FileNotFoundError(f"Prompt template not found: {tmpl_path}")
    with open(tmpl_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_cpu_reference(task_id: str) -> str:
    """Load cpu_reference.c, stripping the file-level comment header."""
    task_dir = get_task_dir(task_id)
    ref_path = os.path.join(task_dir, "cpu_reference.c")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"CPU reference not found: {ref_path}")
    with open(ref_path, "r", encoding="utf-8") as f:
        return f.read()


def _strip_header_comments(code: str) -> str:
    """
    Strip leading // comment block and blank lines from C source,
    keeping only the actual code (from #include onward).
    """
    lines = code.split("\n")
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("//"):
            start = i
            break
    return "\n".join(lines[start:]).strip()


def _build_input_sizes_table(task: TaskConfig) -> str:
    """Build a markdown table from task.json input_sizes."""
    if not task.input_sizes:
        return ""

    # Collect all column names from the first size entry
    first = next(iter(task.input_sizes.values()))
    columns = [k for k in first.keys() if k != "seed"]

    # Header
    header = "| Size | " + " | ".join(col.upper() if len(col) <= 2 else col.replace("_", " ").title() for col in columns) + " |"
    sep = "|" + "|".join("---" for _ in range(len(columns) + 1)) + "|"

    rows = []
    for size_name, params in task.input_sizes.items():
        vals = []
        for col in columns:
            v = params.get(col, "")
            if isinstance(v, int) and v >= 1000:
                vals.append(f"{v:,}")
            else:
                vals.append(str(v))
        rows.append(f"| {size_name} | " + " | ".join(vals) + " |")

    return "\n".join([header, sep] + rows)


# ═══════════════════════════════════════════════════════════════
#  Main assembly
# ═══════════════════════════════════════════════════════════════

SPLIT_KERNELS_GUIDANCE = """\
## Kernel Decomposition (Profiling-Friendly Mode)

You are encouraged to **split the implementation into multiple CUDA kernels** instead of one monolithic kernel.
This makes it easier to profile and optimize each stage. Guidelines:
- Use **separate kernels** for distinct pipeline phases (e.g., build grid / generate pairs / sort+dedup / narrow-phase SAT / write-back).
- Keep each kernel focused, with clear inputs/outputs.
- Add a short comment above each `__global__` kernel describing its responsibility, e.g. `// KERNEL: build grid entries`.
- Minimize atomics and global synchronization; prefer prefix-sums + deterministic write offsets when possible.
"""


def generate_prompt(task_id: str, level: int, split_kernels: bool = False) -> str:
    """
    Assemble a complete prompt for *task_id* at difficulty *level* (1, 2, or 3).

    Returns:
        Markdown string ready to send to an LLM.
    """
    if level not in (1, 2, 3):
        raise ValueError(f"level must be 1, 2, or 3, got {level}")

    tmpl = _load_template(task_id)
    task = load_task(task_id)

    # ── Sections ─────────────────────────────────────────────

    # 1. Title
    title = "# CUDA Acceleration Task"

    # 2. Task description
    task_desc = tmpl.get("task_description", "").strip()
    if task_desc:
        task_desc_section = f"\n\n{task_desc}\n"
    else:
        task_desc_section = "\n\nConvert the following CPU reference implementation to a high-performance CUDA version.\n"

    # 3. Interface (function signatures — task-specific)
    interface_code = tmpl.get("interface", "").strip()
    if task.interface_mode == "compute_only":
        interface_intro = (
            "Implement a single `solution_compute` `extern \"C\"` function in one `.cu` file. "
            "It receives all inputs as host pointers, does the full work on GPU "
            "(allocation, H2D, kernel, D2H, free), and synchronizes before returning. "
            "It is called multiple times and must be idempotent. "
            "**Do NOT** write `main()`, do NOT read/write any files."
        )
    else:
        interface_intro = (
            "Implement `solution_init` and `solution_compute` as `extern \"C\"` functions "
            "in a single `.cu` file. `solution_init` receives inputs once; "
            "`solution_compute` is called multiple times (must synchronize before returning). "
            "**Do NOT** write `main()`, do NOT read/write any files."
        )
    interface_section = f"\n\n## Interface\n\n{interface_intro}\n\n{interface_code}\n"

    # 4. Constraints (generic, mode-dependent)
    if task.interface_mode == "compute_only":
        constraints_text = GENERIC_CONSTRAINTS_COMPUTE_ONLY
    else:
        constraints_text = GENERIC_CONSTRAINTS_INIT_COMPUTE
    constraints_section = f"\n### Constraints\n\n{constraints_text}\n"

    # 4b. Optional split-kernel guidance (profiling-friendly)
    split_section = ""
    if split_kernels:
        split_section = f"\n\n{SPLIT_KERNELS_GUIDANCE}\n"

    # 5. Input sizes (auto-generated from task.json)
    sizes_table = _build_input_sizes_table(task)
    sizes_section = f"\n\n## Input Sizes\n\n{sizes_table}\n" if sizes_table else ""

    # 6. Additional size info (task-specific, e.g. "10 sources × 10 targets")
    size_notes = tmpl.get("input_size_notes", "").strip()
    if size_notes:
        sizes_section += f"\n{size_notes}\n"

    # 6b. Output contract (task-specific, optional)
    output_contract = tmpl.get("output_contract", "").strip()
    output_contract_section = ""
    if output_contract:
        output_contract_section = f"\n\n## Output Contract\n\n{output_contract}\n"

    # 7. Algorithm background (L1 only)
    algo_section = ""
    if level == 1:
        algo_bg = tmpl.get("algorithm_background", "").strip()
        if algo_bg:
            algo_section = f"\n\n## Algorithm Background\n\n{algo_bg}\n"

    # 8. CPU Reference (auto from cpu_reference.c)
    raw_code = _load_cpu_reference(task_id)
    clean_code = _strip_header_comments(raw_code)
    cpu_section = f"\n\n## CPU Reference Implementation\n\n```c\n{clean_code}\n```\n"

    # 9. Optimization hints (level-dependent)
    hints_section = ""
    if level == 1:
        hints = tmpl.get("hints_l1", "").strip()
        if hints:
            hints_section = f"\n\n## Detailed Optimization Guide\n\n{hints}\n"
    elif level == 2:
        hints = tmpl.get("hints_l2", "").strip()
        if hints:
            hints_section = f"\n\n## Optimization Hints\n\n{hints}\n"
    # L3: no hints

    # 10. Response format (mode-dependent)
    if task.interface_mode == "compute_only":
        response_fmt = RESPONSE_FORMAT_COMPUTE_ONLY
    else:
        response_fmt = RESPONSE_FORMAT_INIT_COMPUTE
    response_format_section = f"\n\n## Response Format\n\n{response_fmt}\n"

    # ── Assemble ─────────────────────────────────────────────

    prompt = (
        title
        + task_desc_section
        + interface_section
        + constraints_section
        + split_section
        + sizes_section
        + output_contract_section
        + algo_section
        + cpu_section
        + hints_section
        + response_format_section
    )

    return prompt.strip() + "\n"

