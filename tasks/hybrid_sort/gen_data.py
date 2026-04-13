#!/usr/bin/env python3
"""
gen_data.py — Generate input.bin + expected_output.txt for hybrid_sort

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import re
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))
from framework.orbench_io_py import write_input_bin

SIZES = {
    "small":  {"N": 16384,   "seed": 42, "key_range": 1 << 20},
    "medium": {"N": 262144,  "seed": 43, "key_range": 1 << 24},
    "large":  {"N": 1048576, "seed": 44, "key_range": 1 << 28},
}


def make_keys(cfg):
    rng = np.random.default_rng(cfg["seed"])
    N = cfg["N"]
    key_range = cfg["key_range"]

    centers = np.array([key_range // 8, key_range // 3, key_range // 2, 7 * key_range // 8], dtype=np.int64)
    which = rng.integers(0, len(centers), size=N, dtype=np.int32)
    noise = rng.normal(loc=0.0, scale=max(1, key_range / 32), size=N)
    keys = centers[which].astype(np.float64) + noise
    keys = np.clip(keys, 0, key_range - 1).astype(np.int32)

    mask = rng.random(N) < 0.25
    keys[mask] = rng.integers(0, key_range, size=int(mask.sum()), dtype=np.int32)
    return keys


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "hybrid_sort" / "solution_cpu"
    src = orbench_root / "tasks" / "hybrid_sort" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "hybrid_sort" / "task_io_cpu.c"
    harness = orbench_root / "framework" / "harness_cpu.c"

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
        raise RuntimeError(f"CPU baseline validate run failed:\n{r.stderr}\n{r.stdout}")
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

    cfg = SIZES[size_name]
    keys = make_keys(cfg)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[("keys", "int32", keys)],
        params={"N": int(cfg["N"]), "seed": int(cfg["seed"]), "key_range": int(cfg["key_range"])}
    )

    with open(out_dir / "meta.txt", "w") as f:
        f.write(f"size={size_name}\n")
        f.write(f"N={cfg['N']}\n")
        f.write(f"seed={cfg['seed']}\n")
        f.write(f"key_range={cfg['key_range']}\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")
        run_cpu_expected_output(exe, out_dir)

    print(f"[gen_data] Wrote {size_name} to {out_dir}")


if __name__ == "__main__":
    main()
