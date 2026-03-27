#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) — Generate Gittins Index DP instances.

No random data needed (problem is fully determined by N and a).
Generates input.bin with parameters, then compiles/runs CPU baseline
to produce expected_output.txt and cpu_time_ms.txt.

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
    "small":  {"N": 50,  "a_x10000": 9000, "num_bisect": 40},
    "medium": {"N": 150, "a_x10000": 9500, "num_bisect": 40},
    "large":  {"N": 300, "a_x10000": 9900, "num_bisect": 40},
}


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "gittins_index" / "solution_cpu"
    src = orbench_root / "tasks" / "gittins_index" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "gittins_index" / "task_io_cpu.c"
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
    N = cfg["N"]
    a_x10000 = cfg["a_x10000"]
    num_bisect = cfg["num_bisect"]
    S = N * (N + 1) // 2

    print(f"[gen_data] Generating {size_name}: N={N}, a={a_x10000/10000:.4f}, S={S}")

    # Write input.bin — use a dummy tensor (framework requires at least one)
    dummy = np.zeros(1, dtype=np.int32)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("dummy", "int32", dummy),
        ],
        params={
            "N": N,
            "a_x10000": a_x10000,
            "S": S,
            "num_bisect": num_bisect,
        },
    )

    # Dummy requests.txt
    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        # Rough estimate: O(N^4 * num_bisect), ~1ns per inner op
        est_ops = N**4 * num_bisect // 6
        est_timeout = max(120, int(est_ops * 1e-9) + 60)
        print(f"[gen_data] Running CPU baseline (timeout={est_timeout}s, est_ops={est_ops:.2e})...")
        try:
            time_ms = run_cpu_time(exe, out_dir, timeout=est_timeout)
            with open(out_dir / "cpu_time_ms.txt", "w") as f:
                f.write(f"{time_ms:.3f}\n")
            run_cpu_expected_output(exe, out_dir, timeout=est_timeout)
            print(f"[gen_data] {size_name}: CPU time={time_ms:.1f}ms, wrote all files in {out_dir}")
        except subprocess.TimeoutExpired:
            print(f"[gen_data] CPU baseline timed out ({est_timeout}s). Skipping expected output.")
            print(f"[gen_data] {size_name}: wrote input.bin only in {out_dir}")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin in {out_dir}")


if __name__ == "__main__":
    main()
