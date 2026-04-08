#!/usr/bin/env python3
"""
Generate ORBench input/expected data for nbody_simulation.

Usage:
  python tasks/nbody_simulation/gen_data.py <size_name> <output_dir> [--with-expected]
"""

import os
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
    "small":  {"N": 1024, "seed": 42},
    "medium": {"N": 4096, "seed": 42},
    "large":  {"N": 8192, "seed": 42},
}
SOFTENING = 0.001


def generate_particles(N: int, seed: int):
    rng = np.random.default_rng(seed)
    pos_x = rng.normal(0.0, 1.0, size=N).astype(np.float32)
    pos_y = rng.normal(0.0, 1.0, size=N).astype(np.float32)
    pos_z = rng.normal(0.0, 1.0, size=N).astype(np.float32)
    mass = rng.uniform(0.1, 10.0, size=N).astype(np.float32)
    return pos_x, pos_y, pos_z, mass


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "nbody_simulation" / "solution_cpu"
    src = orbench_root / "tasks" / "nbody_simulation" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "nbody_simulation" / "task_io_cpu.c"
    harness = orbench_root / "framework" / "harness_cpu.c"

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


def run_and_get_time(exe: Path, data_dir: Path, validate: bool = False):
    cmd = [str(exe), str(data_dir)]
    if validate:
        cmd.append("--validate")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline run failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    m = re.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"TIME_MS not found in stdout:\n{r.stdout}")
    return float(m.group(1))


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python tasks/nbody_simulation/gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)

    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = len(sys.argv) == 4 and sys.argv[3] == "--with-expected"
    out_dir.mkdir(parents=True, exist_ok=True)

    if size_name not in SIZES:
        raise ValueError(f"Unknown size {size_name}")

    cfg = SIZES[size_name]
    N = cfg["N"]
    seed = cfg["seed"]

    pos_x, pos_y, pos_z, mass = generate_particles(N, seed)
    softening_x1e6 = int(round(SOFTENING * 1e6))

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("pos_x", "float32", pos_x),
            ("pos_y", "float32", pos_y),
            ("pos_z", "float32", pos_z),
            ("mass",  "float32", mass),
        ],
        params={
            "N": N,
            "softening_x1e6": softening_x1e6,
        },
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_and_get_time(exe, out_dir, validate=False)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")

        run_and_get_time(exe, out_dir, validate=True)
        out_txt = out_dir / "output.txt"
        if not out_txt.exists():
            raise RuntimeError("output.txt not produced by CPU baseline")
        shutil.copy2(out_txt, out_dir / "expected_output.txt")

    print(f"[gen_data] {size_name}: wrote files in {out_dir}")


if __name__ == "__main__":
    main()
