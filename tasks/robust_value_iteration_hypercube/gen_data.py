#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate Robust Value Iteration Hypercube instances.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import os
import sys
import re
import shutil
import subprocess
import numpy as np
from pathlib import Path

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))
from framework.orbench_io_py import write_input_bin

SIZES = {
    "small": {"S": 200, "A": 10, "T": 100, "gamma": 0.95, "seed": 42},
    "medium": {"S": 1000, "A": 10, "T": 200, "gamma": 0.95, "seed": 42},
    "large": {"S": 2000, "A": 20, "T": 500, "gamma": 0.95, "seed": 42}
}

def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "robust_value_iteration_hypercube" / "solution_cpu"
    src = orbench_root / "tasks" / "robust_value_iteration_hypercube" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "robust_value_iteration_hypercube" / "task_io_cpu.c"
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


def run_cpu_time(exe: Path, data_dir: Path, timeout: int = 1200) -> float:
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True,
                       timeout=timeout)
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
    r = subprocess.run([str(exe), str(data_dir), "--validate"],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate failed:\n{r.stderr}\n{r.stdout}")
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
        raise ValueError(f"Unknown size: {size_name}. Available: {list(SIZES.keys())}")

    cfg = SIZES[size_name]
    S = cfg["S"]
    A = cfg["A"]
    T = cfg["T"]
    gamma = cfg["gamma"]
    seed = cfg["seed"]

    np.random.seed(seed)

    rew = np.random.rand(S, A).astype(np.float32)

    P_nom = np.random.rand(S, A, S).astype(np.float32)
    P_nom /= P_nom.sum(axis=-1, keepdims=True)

    P_down = (P_nom * 0.8).astype(np.float32)
    P_up = (P_nom * 1.2).astype(np.float32)
    P_up = np.clip(P_up, 0.0, 1.0)

    print(f"[gen_data] {size_name}: S={S}, A={A}, T={T}, gamma={gamma}, seed={seed}")

    # Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("rew", "float32", rew.flatten()),
            ("P_up", "float32", P_up.flatten()),
            ("P_down", "float32", P_down.flatten()),
        ],
        params={
            "S": int(S),
            "A": int(A),
            "T": int(T),
            "gamma_x10000": int(round(gamma * 10000)),
        },
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write("req\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")
        run_cpu_expected_output(exe, out_dir)
        print(f"[gen_data] {size_name}: CPU time={time_ms:.1f}ms, wrote all files in {out_dir}")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin in {out_dir}")

if __name__ == "__main__":
    main()
