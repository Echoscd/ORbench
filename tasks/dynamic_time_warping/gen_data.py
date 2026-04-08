#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) -- Generate Dynamic Time Warping instances.

Generates batched integer-valued time series, writes ORBench input.bin, and
optionally writes expected_output.txt using the compiled CPU baseline. A small
Python reference is used for spot-checking correctness on a few instances.

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
    "small":  {"N": 1000,  "len_min": 64,  "len_max": 128, "seed": 42},
    "medium": {"N": 5000,  "len_min": 128, "len_max": 256, "seed": 42},
    "large":  {"N": 20000, "len_min": 256, "len_max": 512, "seed": 42},
}


def generate_random_walk(length: int, rng: np.random.Generator) -> np.ndarray:
    steps = rng.integers(-3, 4, size=length, dtype=np.int32)
    return np.cumsum(steps, dtype=np.int32)


def generate_series_pairs(N: int, len_min: int, len_max: int, seed: int):
    rng = np.random.default_rng(seed)

    query_chunks = []
    target_chunks = []
    query_offsets = [0]
    target_offsets = [0]

    for _ in range(N):
        qlen = int(rng.integers(len_min, len_max + 1))
        tlen = int(rng.integers(len_min, len_max + 1))
        q = generate_random_walk(qlen, rng)
        t = generate_random_walk(tlen, rng)
        query_chunks.append(q)
        target_chunks.append(t)
        query_offsets.append(query_offsets[-1] + qlen)
        target_offsets.append(target_offsets[-1] + tlen)

    return (
        np.concatenate(query_chunks).astype(np.int32),
        np.concatenate(target_chunks).astype(np.int32),
        np.array(query_offsets, dtype=np.int32),
        np.array(target_offsets, dtype=np.int32),
    )


def dtw_python(q: np.ndarray, t: np.ndarray) -> int:
    n = len(t)
    prev = [0] + [10**18] * n
    for i in range(1, len(q) + 1):
        curr = [10**18] * (n + 1)
        qi = int(q[i - 1])
        for j in range(1, n + 1):
            cost = abs(qi - int(t[j - 1]))
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return int(prev[n])


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "dynamic_time_warping" / "solution_cpu"
    src = orbench_root / "tasks" / "dynamic_time_warping" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "dynamic_time_warping" / "task_io_cpu.c"
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


def estimate_timeout(total_cells: int) -> int:
    return max(120, min(2400, int(total_cells / 4.0e7) + 120))


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
    print(f"[gen_data] Generating {size_name}: {N} DTW pairs...")

    query_series, target_series, query_offsets, target_offsets = generate_series_pairs(
        N, cfg["len_min"], cfg["len_max"], cfg["seed"]
    )

    total_query_len = int(len(query_series))
    total_target_len = int(len(target_series))
    total_cells = 0
    for p in range(N):
        qlen = int(query_offsets[p + 1] - query_offsets[p])
        tlen = int(target_offsets[p + 1] - target_offsets[p])
        total_cells += qlen * tlen

    print(f"  total_query_len={total_query_len}, total_target_len={total_target_len}, total_cells={total_cells}")

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("query_series", "int32", query_series),
            ("target_series", "int32", target_series),
            ("query_offsets", "int32", query_offsets),
            ("target_offsets", "int32", target_offsets),
        ],
        params={
            "N": N,
            "total_query_len": total_query_len,
            "total_target_len": total_target_len,
        },
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        # spot-check a handful of pairs with Python
        num_check = min(8, N)
        print(f"  Spot-checking first {num_check} pairs with Python DTW reference...")
        for p in range(num_check):
            qs, qe = int(query_offsets[p]), int(query_offsets[p + 1])
            ts, te = int(target_offsets[p]), int(target_offsets[p + 1])
            dtw_python(query_series[qs:qe], target_series[ts:te])

        timeout = estimate_timeout(total_cells)
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
