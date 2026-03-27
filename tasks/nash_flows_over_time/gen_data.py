#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate Nash Flows Over Time instances.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import os
import sys
import re
import subprocess
import numpy as np
from pathlib import Path

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))
from framework.orbench_io_py import write_input_bin

SIZES = {
    "small": {"num_nodes": 100, "num_edges": 400, "num_steps": 1000, "seed": 42},
    "medium": {"num_nodes": 1000, "num_edges": 4000, "num_steps": 5000, "seed": 42},
    "large": {"num_nodes": 5000, "num_edges": 20000, "num_steps": 10000, "seed": 42}
}

def generate_graph(num_nodes, num_edges, seed):
    np.random.seed(seed)
    edges = set()

    # Ensure a path from 0 to N-1
    for i in range(num_nodes - 1):
        edges.add((i, i + 1))

    # Add random edges
    while len(edges) < num_edges:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u != v:
            edges.add((u, v))

    edges = list(edges)
    np.random.shuffle(edges)

    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)
    edge_capacity = np.random.uniform(1.0, 10.0, size=num_edges).astype(np.float32)
    edge_transit_time = np.random.randint(1, 31, size=num_edges).astype(np.int32)

    return edge_u, edge_v, edge_capacity, edge_transit_time


# ---------------------------------------------------------------------------
# CPU baseline compile/run
# ---------------------------------------------------------------------------

def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "nash_flows_over_time" / "solution_cpu"
    src = orbench_root / "tasks" / "nash_flows_over_time" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "nash_flows_over_time" / "task_io_cpu.c"
    harness = orbench_root / "framework" / "harness_cpu.c"

    sources = [src, task_io_cpu, harness]
    if exe.exists():
        try:
            exe_m = exe.stat().st_mtime
            if all(exe_m >= s.stat().st_mtime for s in sources):
                return exe
        except Exception:
            pass

    cmd = [
        "gcc", "-O2", "-DORBENCH_COMPUTE_ONLY",
        "-I", str(orbench_root / "framework"),
        str(harness), str(task_io_cpu), str(src),
        "-o", str(exe), "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}")
    return exe


def run_cpu_time(exe: Path, data_dir: Path, timeout: int = 600) -> float:
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline run failed:\n{r.stderr}\n{r.stdout}")
    m = re.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"TIME_MS not found in stdout:\n{r.stdout}")
    return float(m.group(1))


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)

    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = (len(sys.argv) == 4 and sys.argv[3] == "--with-expected")
    out_dir.mkdir(parents=True, exist_ok=True)

    if size_name not in SIZES:
        raise ValueError(f"Unknown size: {size_name}. Available: {list(SIZES.keys())}")

    config = SIZES[size_name]
    num_nodes = config["num_nodes"]
    num_edges = config["num_edges"]
    num_steps = config["num_steps"]

    edge_u, edge_v, edge_capacity, edge_transit_time = generate_graph(
        num_nodes, num_edges, config["seed"]
    )

    print(f"[gen_data] {size_name}: num_nodes={num_nodes}, num_edges={num_edges}, num_steps={num_steps}")

    # Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("edge_u", "int32", edge_u),
            ("edge_v", "int32", edge_v),
            ("edge_capacity", "float32", edge_capacity),
            ("edge_transit_time", "int32", edge_transit_time),
        ],
        params={
            "num_nodes": int(num_nodes),
            "num_edges": int(num_edges),
            "num_steps": int(num_steps),
        },
    )

    # requests.txt
    inflow_rates = [5.0 + i * 1.5 for i in range(10)]
    with open(out_dir / "requests.txt", "w") as f:
        for rate in inflow_rates:
            f.write(f"{rate:.2f}\n")

    if with_expected:
        try:
            exe = compile_cpu_baseline(_ORBENCH_ROOT)
            print(f"[gen_data] Running CPU baseline...")
            time_ms = run_cpu_time(exe, out_dir)
            with open(out_dir / "cpu_time_ms.txt", "w") as f:
                f.write(f"{time_ms:.3f}\n")
            print(f"[gen_data] {size_name}: CPU time={time_ms:.1f}ms, wrote all files in {out_dir}")
        except Exception as e:
            print(f"[gen_data] CPU baseline failed: {e}")
            print(f"[gen_data] {size_name}: wrote input.bin only in {out_dir}")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin in {out_dir}")

if __name__ == "__main__":
    main()
