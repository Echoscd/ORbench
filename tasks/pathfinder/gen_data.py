#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate Pathfinder test data.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

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
    "small":  {"rows": 1024, "cols": 2048, "seed": 42},
    "medium": {"rows": 4096, "cols": 4096, "seed": 42},
    "large":  {"rows": 8192, "cols": 8192, "seed": 42},
}


def generate_wall(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 10, size=rows * cols, dtype=np.int32)


def compute_expected(rows: int, cols: int, wall_flat: np.ndarray) -> np.ndarray:
    wall = wall_flat.reshape(rows, cols)
    prev = wall[0].astype(np.int64, copy=True)

    for r in range(1, rows):
        left = np.empty_like(prev)
        right = np.empty_like(prev)
        left[0] = np.iinfo(prev.dtype).max
        left[1:] = prev[:-1]
        right[-1] = np.iinfo(prev.dtype).max
        right[:-1] = prev[1:]
        best = np.minimum(prev, np.minimum(left, right))
        prev = wall[r].astype(np.int64) + best

    return prev.astype(np.int32)


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "pathfinder" / "solution_cpu"
    src = orbench_root / "tasks" / "pathfinder" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "pathfinder" / "task_io_cpu.c"
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
        "gcc", "-O2",
        "-I", str(orbench_root / "framework"),
        str(harness), str(task_io_cpu), str(src),
        "-o", str(exe), "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}")
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
        raise RuntimeError(f"CPU baseline validate failed:\n{r.stderr}\n{r.stdout}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")
    shutil.copy2(out_txt, data_dir / "expected_output.txt")


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)

    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = (len(sys.argv) == 4 and sys.argv[3] == "--with-expected")
    out_dir.mkdir(parents=True, exist_ok=True)

    if size_name not in SIZES:
        raise ValueError(f"Unknown size: {size_name}")

    rows = int(SIZES[size_name]["rows"])
    cols = int(SIZES[size_name]["cols"])
    seed = int(SIZES[size_name]["seed"])

    wall = generate_wall(rows, cols, seed)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[("wall", "int32", wall)],
        params={"rows": rows, "cols": cols},
    )

    if with_expected:
        expected = compute_expected(rows, cols, wall)
        with open(out_dir / "expected_output.txt", "w") as f:
            for x in expected:
                f.write(f"{int(x)}\n")

        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        cpu_time_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{cpu_time_ms:.3f}\n")

        # Cross-check Python expected against CPU baseline output.
        run_cpu_expected_output(exe, out_dir)

    print(f"Generated pathfinder/{size_name} at {out_dir}")


if __name__ == "__main__":
    main()
