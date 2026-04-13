#!/usr/bin/env python3
"""
Generate ORBench input.bin for auction_lap.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))
from framework.orbench_io_py import write_input_bin

SIZES = {
    "small": {"n": 256, "seed": 101},
    "medium": {"n": 768, "seed": 202},
    "large": {"n": 1536, "seed": 303},
}


def make_profit_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 100000, size=(n, n), dtype=np.int64)
    row_bonus = rng.integers(0, 4096, size=(n, 1), dtype=np.int64)
    col_bonus = rng.integers(0, 4096, size=(1, n), dtype=np.int64)
    uniq = ((np.arange(n, dtype=np.int64)[:, None] * 1315423911)
            ^ (np.arange(n, dtype=np.int64)[None, :] * 2654435761)) & 15
    profit = base * 16 + row_bonus + col_bonus + uniq
    return profit.astype(np.int32, copy=False).reshape(-1)


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "auction_lap" / "solution_cpu"
    src = orbench_root / "tasks" / "auction_lap" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "auction_lap" / "task_io_cpu.c"
    harness = orbench_root / "framework" / "harness_cpu.c"

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
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}\n{r.stdout}")
    return exe


def run_cpu_time(exe: Path, data_dir: Path) -> float:
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline run failed:\n{r.stderr}\n{r.stdout}")
    m = re.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"TIME_MS not found in stdout:\n{r.stdout}")
    return float(m.group(1))


def run_cpu_expected_output(exe: Path, data_dir: Path) -> None:
    out_txt = data_dir / "output.txt"
    if out_txt.exists():
        out_txt.unlink()
    r = subprocess.run([str(exe), str(data_dir), "--validate"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate run failed:\n{r.stderr}\n{r.stdout}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")
    shutil.copy2(out_txt, data_dir / "expected_output.txt")


def main() -> None:
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)

    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = len(sys.argv) == 4 and sys.argv[3] == "--with-expected"
    out_dir.mkdir(parents=True, exist_ok=True)

    if size_name not in SIZES:
        raise ValueError(f"Unknown size {size_name}")

    n = SIZES[size_name]["n"]
    seed = SIZES[size_name]["seed"]
    profit = make_profit_matrix(n, seed)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[("profit", "int32", profit)],
        params={"n": n, "seed": seed},
    )

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        t = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{t:.3f}\n")
        run_cpu_expected_output(exe, out_dir)


if __name__ == "__main__":
    main()
