#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate Hawkes Dynamic Pricing HJB instances.

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

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "hawkes_dynamic_pricing_hjb" / "solution_cpu"
    src = orbench_root / "tasks" / "hawkes_dynamic_pricing_hjb" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "hawkes_dynamic_pricing_hjb" / "task_io_cpu.c"
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
    "small": {"batch_size": 2, "J": 1000, "N": 500, "num_actions": 50},
    "medium": {"batch_size": 8, "J": 2500, "N": 1000, "num_actions": 100},
    "large": {"batch_size": 32, "J": 5000, "N": 1000, "num_actions": 100}
}

@njit
def compute_expected(batch_size, J, N, num_actions, h, tau, lambda0, a, b_param, alpha0, beta):
    U = np.zeros((batch_size, J, N), dtype=np.float32)
    Lambda = np.zeros((batch_size, J, N), dtype=np.float32)

    for n in range(N - 2, -1, -1):
        for b in range(batch_size):
            a0 = alpha0[b]
            bet = beta[b]
            for j in range(J):
                real_j = float(j + 1)
                x = real_j * h

                bb1 = real_j * (1.0 - bet * tau)
                bb2 = int(np.floor(bb1))

                cc1 = bb1 + (a0 / h) * (1.0 - bet * tau)
                cc2 = int(np.floor(cc1))

                if bb2 == 0:
                    deltabb = U[b, 1, n+1] - U[b, 0, n+1]
                    Vbb = U[b, 0, n+1] - (1.0 - bb1) * deltabb
                else:
                    deltabb = U[b, bb2, n+1] - U[b, bb2-1, n+1]
                    Vbb = U[b, bb2-1, n+1] + (bb1 - float(bb2)) * deltabb

                if cc1 >= float(J):
                    deltacc = U[b, J-1, n+1] - U[b, J-2, n+1]
                    Vcc = U[b, J-1, n+1] + (cc1 - float(J)) * deltacc
                else:
                    safe_cc2 = max(1, cc2)
                    deltacc = U[b, safe_cc2, n+1] - U[b, safe_cc2-1, n+1]
                    Vcc = U[b, safe_cc2-1, n+1] + (cc1 - float(safe_cc2)) * deltacc

                max_A = -1e30
                best_lambda = lambda0

                for pp in range(num_actions):
                    lambda_val = lambda0 + pp * 0.001
                    A_val = lambda_val * ((a - lambda_val) / b_param + Vcc - Vbb)
                    if A_val > max_A:
                        max_A = A_val
                        best_lambda = lambda_val

                Lambda[b, j, n] = best_lambda
                U[b, j, n] = Vbb + (x + 0.001 * x * x + 1.0) * tau * max_A

    return U, Lambda

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
    batch_size = cfg["batch_size"]
    J = cfg["J"]
    N = cfg["N"]
    num_actions = cfg["num_actions"]

    np.random.seed(42)

    h = np.float32(0.01)
    tau = np.float32(0.1)
    lambda0 = np.float32(0.196)
    a = np.float32(0.392)
    b_param = np.float32(0.392)

    alpha0 = np.random.uniform(0.08, 0.12, size=batch_size).astype(np.float32)
    beta = np.random.uniform(0.09, 0.13, size=batch_size).astype(np.float32)

    print(f"[gen_data] {size_name}: batch_size={batch_size}, J={J}, N={N}, num_actions={num_actions}")

    # Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("alpha0", "float32", alpha0),
            ("beta", "float32", beta),
        ],
        params={
            "batch_size": int(batch_size),
            "J": int(J),
            "N": int(N),
            "num_actions": int(num_actions),
            "h_x1e6": int(round(float(h) * 1e6)),
            "tau_x1e6": int(round(float(tau) * 1e6)),
            "lambda0_x1e6": int(round(float(lambda0) * 1e6)),
            "a_x1e6": int(round(float(a) * 1e6)),
            "b_param_x1e6": int(round(float(b_param) * 1e6)),
        },
    )

    # requests.txt
    with open(out_dir / "requests.txt", "w") as f:
        f.write("run\n")

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
