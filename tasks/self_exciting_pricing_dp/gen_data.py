#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate Self-Exciting Pricing DP instances.

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

def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "self_exciting_pricing_dp" / "solution_cpu"
    src = orbench_root / "tasks" / "self_exciting_pricing_dp" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "self_exciting_pricing_dp" / "task_io_cpu.c"
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


SIZES = {
    "small": {"N_x": 10000, "N_T": 100, "N_lambda": 50},
    "medium": {"N_x": 50000, "N_T": 500, "N_lambda": 100},
    "large": {"N_x": 100000, "N_T": 1000, "N_lambda": 200},
}

def solve_dp(N_x, N_T, N_lambda, dx, dt, alpha, beta, phi, c, a, gamma, k, lambda_min, lambda_max):
    V_prev = np.zeros(N_x, dtype=np.float32)
    V_new = np.zeros(N_x, dtype=np.float32)

    lambdas = np.linspace(lambda_min, lambda_max, N_lambda, dtype=np.float32)
    r_lambdas = lambdas * (np.log(a / lambdas) - c)

    alpha_idx_shift = int(np.round(alpha / dx))

    x_arr = np.arange(N_x, dtype=np.float32) * dx
    h_x = k * (x_arr ** gamma)
    h_x_phi = h_x + phi

    for t in range(N_T):
        V_deriv = np.zeros_like(V_prev)
        V_deriv[1:] = (V_prev[1:] - V_prev[:-1]) / dx

        idx_alpha = np.minimum(np.arange(N_x) + alpha_idx_shift, N_x - 1)
        V_jump = V_prev[idx_alpha] - V_prev

        vals = r_lambdas + np.outer(V_jump, lambdas)
        max_val = np.max(vals, axis=1)

        V_new = V_prev + dt * (-x_arr * beta * V_deriv + h_x_phi * max_val)

        V_prev, V_new = V_new, V_prev

    return V_prev

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

    params = SIZES[size_name]
    N_x = params["N_x"]
    N_T = params["N_T"]
    N_lambda = params["N_lambda"]

    dx = np.float32(0.01)
    dt = np.float32(0.001)
    alpha = np.float32(0.05)
    beta = np.float32(0.001)
    phi = np.float32(0.1)
    c = np.float32(1.0)
    a = np.float32(10.0)
    gamma = np.float32(0.5)
    k = np.float32(0.01)
    lambda_min = np.float32(0.1)
    lambda_max = np.float32(5.0)

    int_params = np.array([N_x, N_T, N_lambda], dtype=np.int32)
    float_params = np.array([dx, dt, alpha, beta, phi, c, a, gamma, k, lambda_min, lambda_max], dtype=np.float32)

    print(f"[gen_data] {size_name}: N_x={N_x}, N_T={N_T}, N_lambda={N_lambda}")

    # Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("int_params", "int32", int_params),
            ("float_params", "float32", float_params),
        ],
        params={
            "N_x": int(N_x),
            "N_T": int(N_T),
            "N_lambda": int(N_lambda),
        },
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write("req1\n")

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
