#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate input.bin + requests + expected_output.txt + cpu_time_ms.txt

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import os
import sys
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

from framework.orbench_io_py import write_input_bin


SIZES = {
    "small":  {"V": 1000,   "E": 5000,    "seed": 42, "num_requests": 100},
    "medium": {"V": 100000, "E": 500000,  "seed": 42, "num_requests": 100},
    "large":  {"V": 500000, "E": 2500000, "seed": 42, "num_requests": 100},
}


def generate_graph_csr(V: int, E: int, seed: int):
    """
    Generate random directed graph in CSR using numpy for speed/reproducibility.
    Ensures no self-loops by rejection (may slightly oversample then truncate).
    """
    rng = np.random.default_rng(seed)
    # Oversample to reduce rejection loops for large E
    need = E
    us = []
    vs = []
    ws = []
    while need > 0:
        batch = int(need * 1.1) + 1024
        u = rng.integers(0, V, size=batch, dtype=np.int32)
        v = rng.integers(0, V, size=batch, dtype=np.int32)
        mask = (u != v)
        u = u[mask]
        v = v[mask]
        w = rng.random(size=u.shape[0], dtype=np.float32) * np.float32(99.0) + np.float32(1.0)
        take = min(need, u.shape[0])
        us.append(u[:take])
        vs.append(v[:take])
        ws.append(w[:take])
        need -= take

    u = np.concatenate(us)
    v = np.concatenate(vs)
    w = np.concatenate(ws)

    # Sort by u for CSR
    order = np.argsort(u, kind="mergesort")
    u = u[order]
    v = v[order]
    w = w[order]

    counts = np.bincount(u, minlength=V).astype(np.int32)
    row_offsets = np.empty(V + 1, dtype=np.int32)
    row_offsets[0] = 0
    np.cumsum(counts, out=row_offsets[1:])
    col_indices = v.astype(np.int32, copy=False)
    weights = w.astype(np.float32, copy=False)
    return row_offsets, col_indices, weights


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "bellman_ford" / "solution_cpu"
    src = orbench_root / "tasks" / "bellman_ford" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "bellman_ford" / "task_io_cpu.c"
    harness = orbench_root / "framework" / "harness_cpu.c"
    # Simple cache: if exe exists and is newer than all sources, reuse
    sources = [src, task_io_cpu, harness]
    if exe.exists():
        try:
            exe_m = exe.stat().st_mtime
            if all(exe_m >= s.stat().st_mtime for s in sources):
                return exe
        except Exception:
            pass
    # v2.1: three-file compilation  harness + task_io + cpu_reference
    cmd = [
        "gcc", "-O2",
        "-I", str(orbench_root / "framework"),
        str(harness),
        str(task_io_cpu),
        str(src),
        "-o", str(exe),
        "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}")
    return exe


def run_cpu_time(exe: Path, data_dir: Path) -> float:
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline run failed:\n{r.stderr}")
    m = re.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"TIME_MS not found in stdout:\n{r.stdout}")
    return float(m.group(1))


def run_cpu_expected_output(exe: Path, data_dir: Path) -> None:
    # --validate writes output.txt (text format: space-separated floats)
    out_txt = data_dir / "output.txt"
    if out_txt.exists():
        out_txt.unlink()
    r = subprocess.run([str(exe), str(data_dir), "--validate"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate run failed:\n{r.stderr}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")
    expected = data_dir / "expected_output.txt"
    shutil.copy2(out_txt, expected)


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)

    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = (len(sys.argv) == 4 and sys.argv[3] == "--with-expected")
    out_dir.mkdir(parents=True, exist_ok=True)

    if size_name not in SIZES:
        raise ValueError(f"Unknown size: {size_name}")
    V = int(SIZES[size_name]["V"])
    E = int(SIZES[size_name]["E"])
    seed = int(SIZES[size_name]["seed"])
    num_requests = int(SIZES[size_name]["num_requests"])

    # 1) Generate graph tensors
    row_offsets, col_indices, weights = generate_graph_csr(V, E, seed)

    # 2) Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("row_offsets", "int32", row_offsets),
            ("col_indices", "int32", col_indices),
            ("weights", "float32", weights),
        ],
        params={"V": V, "E": E},
    )

    # 3) Generate requests: (s, t) pairs
    # s 分 10 组，每组 10 个不同的 s，每个 s 配 10 个不同的 t，共 100 条
    rng = np.random.default_rng(seed)
    with open(out_dir / "requests.txt", "w") as f:
        # 生成 10 个不同的 s（每组一个）
        s_group = rng.integers(0, V, size=10, dtype=np.int32)
        for s in s_group:
            # 每个 s 配 10 个不同的 t
            t_list = rng.integers(0, V, size=10, dtype=np.int32)
            for t in t_list:
                f.write(f"{int(s)} {int(t)}\n")

    if with_expected:
        # 4) Compile + run CPU baseline to produce expected_output and cpu_time
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")
        run_cpu_expected_output(exe, out_dir)
        print(f"[gen_data] {size_name}: wrote input.bin/requests.txt/expected_output.txt/cpu_time_ms.txt in {out_dir}")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin/requests.txt in {out_dir} (expected/cpu_time skipped; pass --with-expected)")


if __name__ == "__main__":
    main()
