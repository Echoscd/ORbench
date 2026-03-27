#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate Batched LHPCA Portfolio instances.

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
    "small":  {"S": 4,  "T": 256,  "N": 256,  "K": 4,  "seed": 42},
    "medium": {"S": 16, "T": 512,  "N": 1024, "K": 8,  "seed": 42},
    "large":  {"S": 32, "T": 1024, "N": 2048, "K": 16, "seed": 42},
}

def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "batched_lhpca_portfolio" / "solution_cpu"
    src = orbench_root / "tasks" / "batched_lhpca_portfolio" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "batched_lhpca_portfolio" / "task_io_cpu.c"
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


def compute_portfolio_weights(R_batch, K):
    S, T, N = R_batch.shape
    w_batch = np.zeros((S, N), dtype=np.float32)

    for s in range(S):
        R = R_batch[s]

        norms = np.linalg.norm(R, axis=0)
        norms[norms == 0] = 1.0
        R_norm = R / norms

        M = R_norm @ R_norm.T

        eigen_val, eigen_vec = np.linalg.eigh(M)
        Phi = eigen_vec[:, -K:]

        Z = Phi.T @ R
        F = Phi @ Z

        E = R - F

        C_F = (F.T @ F) / T
        V_E = np.sum(E**2, axis=0) / T
        Sigma_hat = C_F + np.diag(V_E)

        ones = np.ones(N, dtype=np.float32)
        x = np.linalg.solve(Sigma_hat, ones)
        w_batch[s] = x / np.sum(x)

    return w_batch

def generate_data(S, T, N, K, seed):
    np.random.seed(seed)
    R = np.zeros((S, T, N), dtype=np.float32)
    for s in range(S):
        factors = np.random.randn(T, K).astype(np.float32)
        for k in range(K):
            factors[:, k] *= (K - k + 5)
        loadings = np.random.randn(K, N).astype(np.float32)
        noise = np.random.randn(T, N).astype(np.float32) * 0.5
        R[s] = factors @ loadings + noise
    return R

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
    T = cfg["T"]
    N = cfg["N"]
    K = cfg["K"]
    seed = cfg["seed"]

    R = generate_data(S, T, N, K, seed)

    print(f"[gen_data] {size_name}: S={S}, T={T}, N={N}, K={K}, seed={seed}")

    # Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("R", "float32", R.flatten()),
        ],
        params={
            "S": int(S),
            "T": int(T),
            "N": int(N),
            "K": int(K),
            "seed": int(seed),
        },
    )

    # Dummy requests.txt
    with open(out_dir / "requests.txt", "w") as f:
        f.write(f"{S} {T} {N} {K}\n")

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
