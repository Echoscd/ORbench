#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) -- Generate Monte Carlo Asian option pricing instances.

Generates batched option contracts plus a shared matrix of standard-normal shocks,
writes ORBench input.bin, and optionally writes expected_output.txt using the compiled
CPU baseline. A small Python reference is used for spot-checking a few contracts.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

from framework.orbench_io_py import write_input_bin

SIZES = {
    "small":  {"N": 256,  "num_paths": 512,  "num_steps": 64,  "seed": 42},
    "medium": {"N": 512,  "num_paths": 2048, "num_steps": 64,  "seed": 42},
    "large":  {"N": 1024, "num_paths": 4096, "num_steps": 96,  "seed": 42},
}


def generate_contracts(N: int, seed: int):
    rng = np.random.default_rng(seed)
    s0 = rng.uniform(50.0, 150.0, size=N).astype(np.float32)
    strike = (s0 * rng.uniform(0.85, 1.15, size=N)).astype(np.float32)
    rate = rng.uniform(0.01, 0.08, size=N).astype(np.float32)
    sigma = rng.uniform(0.10, 0.60, size=N).astype(np.float32)
    maturity = rng.uniform(0.25, 2.00, size=N).astype(np.float32)
    option_type = rng.integers(0, 2, size=N, dtype=np.int32)
    return s0, strike, rate, sigma, maturity, option_type


def generate_shocks(num_paths: int, num_steps: int, seed: int):
    rng = np.random.default_rng(seed + 12345)
    return rng.standard_normal(size=(num_paths, num_steps), dtype=np.float32).reshape(-1)


def asian_price_python(s0, strike, rate, sigma, maturity, option_type, shocks, num_paths, num_steps):
    z = shocks.reshape(num_paths, num_steps)
    dt = maturity / float(num_steps)
    drift = (rate - 0.5 * sigma * sigma) * dt
    vol_term = sigma * np.sqrt(dt)
    S = np.full(num_paths, s0, dtype=np.float64)
    path_sum = np.zeros(num_paths, dtype=np.float64)
    for s in range(num_steps):
        S *= np.exp(drift + vol_term * z[:, s])
        path_sum += S
    avg_S = path_sum / float(num_steps)
    if option_type == 1:
        payoff = np.maximum(strike - avg_S, 0.0)
    else:
        payoff = np.maximum(avg_S - strike, 0.0)
    return float(np.exp(-rate * maturity) * payoff.mean())


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "asian_option_pricing_mc" / "solution_cpu"
    src = orbench_root / "tasks" / "asian_option_pricing_mc" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "asian_option_pricing_mc" / "task_io_cpu.c"
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


def run_cpu_time(exe: Path, data_dir: Path, timeout: int) -> float:
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline run failed:\n{r.stderr}\n{r.stdout}")
    m = re.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"TIME_MS not found in stdout:\n{r.stdout}")
    return float(m.group(1))


def run_cpu_expected_output(exe: Path, data_dir: Path, timeout: int) -> None:
    out_txt = data_dir / "output.txt"
    if out_txt.exists():
        out_txt.unlink()
    r = subprocess.run([str(exe), str(data_dir), "--validate"], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate failed:\n{r.stderr}\n{r.stdout}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")


def estimate_timeout(N: int, num_paths: int, num_steps: int) -> int:
    work = N * num_paths * num_steps
    return max(120, min(2400, int(work / 1.5e8) + 120))


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
    num_paths = cfg["num_paths"]
    num_steps = cfg["num_steps"]
    seed = cfg["seed"]

    print(f"[gen_data] Generating {size_name}: N={N}, paths={num_paths}, steps={num_steps}...")
    s0, strike, rate, sigma, maturity, option_type = generate_contracts(N, seed)
    shocks = generate_shocks(num_paths, num_steps, seed)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("s0", "float32", s0),
            ("strike", "float32", strike),
            ("rate", "float32", rate),
            ("sigma", "float32", sigma),
            ("maturity", "float32", maturity),
            ("option_type", "int32", option_type),
            ("shocks", "float32", shocks),
        ],
        params={
            "N": N,
            "num_paths": num_paths,
            "num_steps": num_steps,
        },
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        num_check = min(4, N)
        print(f"  Spot-checking first {num_check} contracts with Python Asian-option reference...")
        for i in range(num_check):
            asian_price_python(float(s0[i]), float(strike[i]), float(rate[i]), float(sigma[i]), float(maturity[i]), int(option_type[i]), shocks, num_paths, num_steps)

        timeout = estimate_timeout(N, num_paths, num_steps)
        print(f"  Compiling/running CPU baseline (timeout={timeout}s) for expected output + timing...")
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir, timeout=timeout)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")

        run_cpu_expected_output(exe, out_dir, timeout=timeout)
        os.replace(out_dir / "output.txt", out_dir / "expected_output.txt")
        print(f"[gen_data] {size_name}: CPU time={time_ms:.3f} ms, wrote all files in {out_dir}")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin in {out_dir}")


if __name__ == "__main__":
    main()
