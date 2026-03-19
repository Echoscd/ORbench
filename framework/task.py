"""
task.py - Task definition loading and management
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional


ORBENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = os.path.join(ORBENCH_ROOT, "tasks")


@dataclass
class TaskConfig:
    """A single benchmark task definition"""
    task_id: str
    name: str
    category: str           # e.g. "graph_traversal", "reduction", "stencil"
    difficulty: int          # 1-4 stars
    tags: list[str] = field(default_factory=list)

    # Input sizes for multi-scale testing
    input_sizes: dict = field(default_factory=dict)

    # Correctness checking config
    correctness_mode: str = "numerical"   # "exact" or "numerical"
    correctness_field: str = "cs"         # used when correctness_mode == "checksum"
    atol: float = 0.01
    rtol: float = 0.01
    allow_different_iterations: bool = False

    # Build
    build_command: str = "nvcc -O2 -arch=sm_89 -o solution solution.cu"
    extra_build_flags: str = ""

    # Timing
    warmup: int = 3
    trials: int = 10
    timeout: int = 180

    # Interface mode: "init_compute" (default) or "compute_only"
    interface_mode: str = "init_compute"

    # GPU optimization points this task could benefit from
    gpu_optimization_points: list[str] = field(default_factory=list)


def load_task(task_id: str) -> TaskConfig:
    """Load task configuration from tasks/{task_id}/task.json"""
    task_dir = os.path.join(TASKS_DIR, task_id)
    task_json_path = os.path.join(task_dir, "task.json")

    if not os.path.exists(task_json_path):
        raise FileNotFoundError(f"Task config not found: {task_json_path}")

    with open(task_json_path, "r") as f:
        data = json.load(f)

    return TaskConfig(
        task_id=data["task_id"],
        name=data["name"],
        category=data.get("category", "unknown"),
        difficulty=data.get("difficulty", 1),
        tags=data.get("tags", []),
        input_sizes=data.get("input_sizes", {}),
        correctness_mode=data.get("correctness", {}).get("mode", "numerical"),
        correctness_field=data.get("correctness", {}).get("field", "cs"),
        atol=data.get("correctness", {}).get("atol", 0.01),
        rtol=data.get("correctness", {}).get("rtol", 0.01),
        allow_different_iterations=data.get("correctness", {}).get("allow_different_iterations", False),
        build_command=data.get("build_command", "nvcc -O2 -arch=sm_89 -o solution solution.cu"),
        extra_build_flags=data.get("extra_build_flags", ""),
        warmup=data.get("timing", {}).get("warmup", 3),
        trials=data.get("timing", {}).get("trials", 10),
        timeout=data.get("timing", {}).get("timeout", 180),
        interface_mode=data.get("interface_mode", "init_compute"),
        gpu_optimization_points=data.get("gpu_optimization_points", []),
    )


def load_all_tasks() -> list[TaskConfig]:
    """Load all tasks from tasks/ directory"""
    tasks = []
    for task_id in sorted(os.listdir(TASKS_DIR)):
        task_dir = os.path.join(TASKS_DIR, task_id)
        if os.path.isdir(task_dir) and os.path.exists(os.path.join(task_dir, "task.json")):
            tasks.append(load_task(task_id))
    return tasks


def get_task_dir(task_id: str) -> str:
    return os.path.join(TASKS_DIR, task_id)


def load_prompt(task_id: str, level: int, split_kernels: bool = False) -> str:
    """
    Load prompt for a task at a given difficulty level (1, 2, or 3).

    Resolution order:
      1. prompt_template.yaml  →  assembled by generate_prompt.py  (preferred)
      2. prompt_l{level}.md    →  static file fallback (legacy)
    """
    task_dir = get_task_dir(task_id)

    # Prefer template-based generation
    tmpl_path = os.path.join(task_dir, "prompt_template.yaml")
    if os.path.exists(tmpl_path):
        from .generate_prompt import generate_prompt
        return generate_prompt(task_id, level, split_kernels=split_kernels)

    # Fallback: static markdown files
    prompt_path = os.path.join(task_dir, f"prompt_l{level}.md")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(
            f"Neither prompt_template.yaml nor prompt_l{level}.md found for task '{task_id}'"
        )

    with open(prompt_path, "r") as f:
        return f.read()


def load_cpu_reference(task_id: str) -> str:
    """Load the CPU reference source code (ORBench v2 uses cpu_reference.c)"""
    task_dir = get_task_dir(task_id)
    ref_path = os.path.join(task_dir, "cpu_reference.c")

    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"CPU reference not found: {ref_path}")

    with open(ref_path, "r") as f:
        return f.read()
