#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate N-body gravitational simulation test data.

Generates random particle distributions and computes expected gravitational forces.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import os
import sys
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

from framework.orbench_io_py import write_input_bin

SIZES = {
    "small":  {"N": 4096,  "seed": 42},
    "medium": {"N": 16384, "seed": 42},
    "large":  {"N": 65536, "seed": 42},
}

SOFTENING = 0.001  # softening parameter to prevent singularities
G = 1.0             # gravitational constant (natural units)


def generate_particles(N, seed):
    """Generate N particles with random positions and masses.

    Positions are drawn from a Plummer sphere model (realistic astrophysical
    distribution), masses are uniform in [0.1, 10.0].
    """
    rng = np.random.default_rng(seed)

    # Plummer sphere: r = 1/sqrt(U^{-2/3} - 1), where U ~ Uniform(0,1)
    u = rng.uniform(0.01, 1.0, size=N).astype(np.float64)
    r = 1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    # Cap radius to prevent extreme outliers
    r = np.clip(r, 0, 50.0)

    # Random direction on sphere
    cos_theta = rng.uniform(-1, 1, size=N)
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    phi = rng.uniform(0, 2 * np.pi, size=N)

    pos_x = (r * sin_theta * np.cos(phi)).astype(np.float32)
    pos_y = (r * sin_theta * np.sin(phi)).astype(np.float32)
    pos_z = (r * cos_theta).astype(np.float32)

    mass = rng.uniform(0.1, 10.0, size=N).astype(np.float32)

    return pos_x, pos_y, pos_z, mass


def compute_forces_python(N, px, py, pz, mass, softening):
    """Direct O(N^2) force computation in Python/numpy for verification."""
    eps2 = softening * softening
    # Use float64 for reference accuracy
    px64 = px.astype(np.float64)
    py64 = py.astype(np.float64)
    pz64 = pz.astype(np.float64)
    m64 = mass.astype(np.float64)

    fx = np.zeros(N, dtype=np.float64)
    fy = np.zeros(N, dtype=np.float64)
    fz = np.zeros(N, dtype=np.float64)

    for i in range(N):
        dx = px64 - px64[i]
        dy = py64 - py64[i]
        dz = pz64 - pz64[i]

        dist2 = dx * dx + dy * dy + dz * dz + eps2
        inv_dist = 1.0 / np.sqrt(dist2)
        inv_dist3 = inv_dist ** 3

        # Zero out self-interaction
        inv_dist3[i] = 0.0

        f = m64 * inv_dist3
        ax = np.sum(f * dx)
        ay = np.sum(f * dy)
        az = np.sum(f * dz)

        fx[i] = m64[i] * ax
        fy[i] = m64[i] * ay
        fz[i] = m64[i] * az

        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{N} particles done")

    return fx.astype(np.float32), fy.astype(np.float32), fz.astype(np.float32)


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "nbody_simulation" / "solution_cpu"
    src = orbench_root / "tasks" / "nbody_simulation" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "nbody_simulation" / "task_io_cpu.c"
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
    r = subprocess.run([str(exe), str(data_dir), "--validate"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate failed:\n{r.stderr}\n{r.stdout}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")
    expected = data_dir / "expected_output.txt"
    shutil.copy2(out_txt, expected)


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
    seed = cfg["seed"]

    print(f"[gen_data] Generating {size_name}: {N} particles...")

    pos_x, pos_y, pos_z, mass = generate_particles(N, seed)

    softening_x1e6 = int(round(SOFTENING * 1e6))

    print(f"  {N} particles, softening={SOFTENING}")

    # Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("pos_x", "float32", pos_x),
            ("pos_y", "float32", pos_y),
            ("pos_z", "float32", pos_z),
            ("mass", "float32", mass),
        ],
        params={
            "N": N,
            "softening_x1e6": softening_x1e6,
        },
    )

    # Dummy requests
    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        print(f"  Computing expected forces via Python reference...")
        fx, fy, fz = compute_forces_python(N, pos_x, pos_y, pos_z, mass, SOFTENING)

        with open(out_dir / "expected_output.txt", "w") as f:
            for i in range(N):
                f.write(f"{fx[i]:.6e} {fy[i]:.6e} {fz[i]:.6e}\n")

        print(f"[gen_data] {size_name}: wrote all files in {out_dir}")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin in {out_dir}")


if __name__ == "__main__":
    main()
