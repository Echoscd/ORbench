#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate Thompson Sampling MC instances.

Generates Bernoulli bandit arm means and computes expected output via CPU simulation.

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
    "small":  {"N": 10,  "T": 10000,  "M": 1000,   "seed": 42},
    "medium": {"N": 50,  "T": 50000,  "M": 10000,  "seed": 42},
    "large":  {"N": 100, "T": 100000, "M": 5000,   "seed": 42},
}


def generate_arm_means(N, seed):
    """
    Generate N arm means for Bernoulli bandits.
    Arm 0 is the best arm with mu=0.5.
    Others are drawn from Uniform(0.1, 0.4).
    """
    rng = np.random.default_rng(seed)
    mu = np.zeros(N, dtype=np.float32)
    mu[0] = 0.5
    mu[1:] = rng.uniform(0.1, 0.4, size=N - 1).astype(np.float32)
    return mu


# ---------------------------------------------------------------------------
# CPU baseline compile/run
# ---------------------------------------------------------------------------

def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "thompson_sampling" / "solution_cpu"
    src = orbench_root / "tasks" / "thompson_sampling" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "thompson_sampling" / "task_io_cpu.c"
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


def run_cpu_expected_output(exe: Path, data_dir: Path, timeout: int = 600) -> None:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    T = cfg["T"]
    M = cfg["M"]
    seed = cfg["seed"]

    # Generate arm means
    arm_means = generate_arm_means(N, seed)
    mu_star = arm_means.max()
    print(f"[gen_data] {size_name}: N={N}, T={T}, M={M}, seed={seed}")
    print(f"[gen_data] Best arm: mu*={mu_star:.3f}, arm means: {arm_means[:5]}...")
    print(f"[gen_data] Estimated total Beta samples: {N * T * M:.2e}")

    # Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("arm_means", "float32", arm_means),
        ],
        params={
            "N": int(N),
            "T": int(T),
            "M": int(M),
            "seed": int(seed),
        },
    )

    # Dummy requests.txt
    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        # Estimate timeout: ~60ns per Beta sample (N * T * M total)
        est_timeout = max(120, int(N * T * M * 6e-8))
        print(f"[gen_data] Running CPU baseline (timeout={est_timeout}s)...")
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
