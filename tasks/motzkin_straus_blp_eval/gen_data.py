#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate Motzkin-Straus BLP Eval instances.

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
    "small": {"N": 100, "M": 1000, "seed": 42},
    "medium": {"N": 1000, "M": 5000, "seed": 42},
    "large": {"N": 5000, "M": 10000, "seed": 42}
}

def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "motzkin_straus_blp_eval" / "solution_cpu"
    src = orbench_root / "tasks" / "motzkin_straus_blp_eval" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "motzkin_straus_blp_eval" / "task_io_cpu.c"
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


def generate_data(N, M, seed):
    np.random.seed(seed)
    mu = np.float32(np.random.uniform(1.0, 10.0))
    # A is an adjacency matrix (symmetric, 0 or 1)
    A = np.random.randint(0, 2, size=(N, N)).astype(np.float32)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)

    x = np.random.uniform(-0.1, 1.1, size=(M, N)).astype(np.float32)
    q = np.random.uniform(-0.1, 1.1, size=(M, N)).astype(np.float32)
    s = np.random.uniform(-0.5, 5.0, size=M).astype(np.float32)

    return N, M, mu, A, x, q, s

def compute_reference(N, M, mu, A, x, q, s):
    obj = np.zeros(M, dtype=np.float32)
    max_viol = np.zeros(M, dtype=np.float32)

    for m in range(M):
        current_x = x[m]
        current_q = q[m]
        current_s = s[m]

        # obj = s - mu * sum(min(x, q))
        obj[m] = current_s - mu * np.sum(np.minimum(current_x, current_q))

        # viol1: s - Ax - q = 0
        ax = A @ current_x
        viol1 = np.abs(current_s - ax - current_q)

        # viol2: sum(x) - 1 = 0
        viol2 = np.abs(np.sum(current_x) - 1.0)

        # viol3: 0 <= q <= 1
        viol3_1 = np.maximum(0.0, -current_q)
        viol3_2 = np.maximum(0.0, current_q - 1.0)

        # viol4: x >= 0
        viol4 = np.maximum(0.0, -current_x)

        # viol5: s >= 0
        viol5 = np.maximum(0.0, -current_s)

        max_viol[m] = max(
            np.max(viol1),
            viol2,
            np.max(viol3_1),
            np.max(viol3_2),
            np.max(viol4),
            viol5
        )

    return obj, max_viol

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
    N, M, mu, A, x, q, s = generate_data(cfg["N"], cfg["M"], cfg["seed"])

    print(f"[gen_data] {size_name}: N={N}, M={M}, seed={cfg['seed']}")

    # Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("A", "float32", A.flatten()),
            ("x", "float32", x.flatten()),
            ("q", "float32", q.flatten()),
            ("s", "float32", s),
            ("fparams", "float32", np.array([mu], dtype=np.float32)),
        ],
        params={
            "N": int(N),
            "M": int(M),
        },
    )

    # Write requests.txt
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
