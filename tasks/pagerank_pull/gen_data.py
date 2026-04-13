#!/usr/bin/env python3
"""
Generate data for pagerank_pull.
Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""
import os
import re
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))
from framework.orbench_io_py import write_input_bin

SIZES = {
    "small": {"V": 4096, "avg_out_degree": 8, "max_iters": 20, "seed": 42},
    "medium": {"V": 32768, "avg_out_degree": 12, "max_iters": 20, "seed": 42},
    "large": {"V": 131072, "avg_out_degree": 12, "max_iters": 20, "seed": 42},
}

DAMPING = np.float32(0.85)
EPSILON = np.float32(1e-4)


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "pagerank_pull" / "solution_cpu"
    src = orbench_root / "tasks" / "pagerank_pull" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "pagerank_pull" / "task_io_cpu.c"
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
        "gcc", "-O2",
        "-I", str(orbench_root / "framework"),
        str(harness), str(task_io_cpu), str(src),
        "-o", str(exe), "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}")
    return exe


def run_cpu_time(exe: Path, data_dir: Path, timeout: int = 1200) -> float:
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline run failed:\n{r.stderr}\n{r.stdout}")
    m = re.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"TIME_MS not found in stdout:\n{r.stdout}")
    return float(m.group(1))


def run_cpu_expected_output(exe: Path, data_dir: Path, timeout: int = 1200) -> None:
    out_txt = data_dir / "output.txt"
    if out_txt.exists():
        out_txt.unlink()
    r = subprocess.run([str(exe), str(data_dir), "--validate"], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate failed:\n{r.stderr}\n{r.stdout}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")
    shutil.copy2(out_txt, data_dir / "expected_output.txt")


def generate_graph(V: int, avg_out_degree: int, seed: int):
    rng = np.random.default_rng(seed)
    out_neighbors = [set() for _ in range(V)]

    # ring edges for strong connectivity and guaranteed positive out-degree
    for u in range(V):
        out_neighbors[u].add((u + 1) % V)

    extra_per_node = max(0, avg_out_degree - 1)
    for u in range(V):
        while len(out_neighbors[u]) < 1 + extra_per_node:
            v = int(rng.integers(0, V))
            if v != u:
                out_neighbors[u].add(v)

    out_degree = np.empty(V, dtype=np.int32)
    in_lists = [[] for _ in range(V)]
    E = 0
    for u in range(V):
        neigh = sorted(out_neighbors[u])
        out_degree[u] = len(neigh)
        E += len(neigh)
        for v in neigh:
            in_lists[v].append(u)

    in_row_offsets = np.zeros(V + 1, dtype=np.int32)
    for u in range(V):
        in_row_offsets[u + 1] = in_row_offsets[u] + len(in_lists[u])
    in_col_indices = np.empty(E, dtype=np.int32)
    cursor = 0
    for u in range(V):
        srcs = sorted(in_lists[u])
        n = len(srcs)
        in_col_indices[cursor:cursor + n] = srcs
        cursor += n

    return in_row_offsets, in_col_indices, out_degree


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)

    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = len(sys.argv) == 4 and sys.argv[3] == "--with-expected"
    out_dir.mkdir(parents=True, exist_ok=True)

    if size_name not in SIZES:
        raise ValueError(f"Unknown size: {size_name}")
    cfg = SIZES[size_name]

    V = cfg["V"]
    avg_out_degree = cfg["avg_out_degree"]
    max_iters = cfg["max_iters"]
    seed = cfg["seed"]

    in_row_offsets, in_col_indices, out_degree = generate_graph(V, avg_out_degree, seed)
    E = int(in_col_indices.size)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("in_row_offsets", "int32", in_row_offsets),
            ("in_col_indices", "int32", in_col_indices),
            ("out_degree", "int32", out_degree),
            ("fparams", "float32", np.array([DAMPING, EPSILON], dtype=np.float32)),
        ],
        params={
            "V": int(V),
            "E": int(E),
            "max_iters": int(max_iters),
        },
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write("pagerank\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")
        run_cpu_expected_output(exe, out_dir)
        print(f"[gen_data] {size_name}: V={V}, E={E}, CPU time={time_ms:.3f} ms")
    else:
        print(f"[gen_data] {size_name}: V={V}, E={E}")


if __name__ == "__main__":
    main()
