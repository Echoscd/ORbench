#!/usr/bin/env python3
"""
Generate ORBench MRI-Q task data.

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
    "small":  {"num_k": 1024, "num_x": 4096,  "sample_stride": 64, "seed": 42},
    "medium": {"num_k": 2048, "num_x": 16384, "sample_stride": 64, "seed": 42},
    "large":  {"num_k": 4096, "num_x": 32768, "sample_stride": 64, "seed": 42},
}


def generate_instance(num_k: int, num_x: int, seed: int):
    rng = np.random.default_rng(seed)
    kx = rng.uniform(-0.5, 0.5, size=num_k).astype(np.float32)
    ky = rng.uniform(-0.5, 0.5, size=num_k).astype(np.float32)
    kz = rng.uniform(-0.5, 0.5, size=num_k).astype(np.float32)
    phi_r = rng.uniform(-1.0, 1.0, size=num_k).astype(np.float32)
    phi_i = rng.uniform(-1.0, 1.0, size=num_k).astype(np.float32)
    x = rng.uniform(-0.5, 0.5, size=num_x).astype(np.float32)
    y = rng.uniform(-0.5, 0.5, size=num_x).astype(np.float32)
    z = rng.uniform(-0.5, 0.5, size=num_x).astype(np.float32)
    return kx, ky, kz, phi_r, phi_i, x, y, z


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "mri_q" / "solution_cpu"
    src = orbench_root / "tasks" / "mri_q" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "mri_q" / "task_io_cpu.c"
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
        raise ValueError(f"Unknown size: {size_name}. Available: {list(SIZES.keys())}")

    cfg = SIZES[size_name]
    kx, ky, kz, phi_r, phi_i, x, y, z = generate_instance(cfg["num_k"], cfg["num_x"], cfg["seed"])

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("kx", "float32", kx),
            ("ky", "float32", ky),
            ("kz", "float32", kz),
            ("phi_r", "float32", phi_r),
            ("phi_i", "float32", phi_i),
            ("x", "float32", x),
            ("y", "float32", y),
            ("z", "float32", z),
        ],
        params={
            "num_k": cfg["num_k"],
            "num_x": cfg["num_x"],
            "sample_stride": cfg["sample_stride"],
        },
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        run_cpu_expected_output(exe, out_dir)
        ms = run_cpu_time(exe, out_dir)
        print(f"[gen_data] {size_name}: CPU TIME_MS ~ {ms:.3f}")
        print(f"[gen_data] wrote input.bin, requests.txt, expected_output.txt in {out_dir}")
    else:
        print(f"[gen_data] wrote input.bin and requests.txt in {out_dir}")


if __name__ == "__main__":
    main()
