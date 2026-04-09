#!/usr/bin/env python3
"""
gen_data.py -- Generate batched Nussinov RNA folding data.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
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
    "small":  {"N": 300,  "len_min": 90,  "len_max": 120, "seed": 42},
    "medium": {"N": 500,  "len_min": 130, "len_max": 170, "seed": 42},
    "large":  {"N": 700,  "len_min": 180, "len_max": 230, "seed": 42},
}
MIN_LOOP_LEN = 3


def generate_sequences(N, len_min, len_max, seed):
    rng = np.random.default_rng(seed)
    seqs = []
    offsets = [0]
    for _ in range(N):
        n = int(rng.integers(len_min, len_max + 1))
        seq = rng.integers(0, 4, size=n, dtype=np.int32)
        seqs.append(seq)
        offsets.append(offsets[-1] + n)
    return np.concatenate(seqs).astype(np.int32), np.array(offsets, dtype=np.int32)


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "nussinov_rna_folding" / "solution_cpu"
    src = orbench_root / "tasks" / "nussinov_rna_folding" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "nussinov_rna_folding" / "task_io_cpu.c"
    harness = orbench_root / "framework" / "harness_cpu.c"
    cmd = [
        "gcc", "-O2", "-DORBENCH_COMPUTE_ONLY",
        "-I", str(orbench_root / "framework"),
        str(harness), str(task_io_cpu), str(src),
        "-o", str(exe), "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}\n{r.stdout}")
    return exe


def run_cpu_time(exe: Path, data_dir: Path, warmup=None, trials=None) -> float:
    cmd = [str(exe), str(data_dir)]
    if warmup is not None:
        cmd.extend(["--warmup", str(warmup)])
    if trials is not None:
        cmd.extend(["--trials", str(trials)])
    r = subprocess.run(cmd, capture_output=True, text=True)
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
    r = subprocess.run([str(exe), str(data_dir), "--validate", "--warmup", "1", "--trials", "1"],
                       capture_output=True, text=True)
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

    cfg = SIZES[size_name]
    seqs, offsets = generate_sequences(cfg["N"], cfg["len_min"], cfg["len_max"], cfg["seed"])
    total_seq_len = int(seqs.size)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("seqs", "int32", seqs),
            ("offsets", "int32", offsets),
        ],
        params={
            "N": int(cfg["N"]),
            "total_seq_len": total_seq_len,
            "min_loop_len": int(MIN_LOOP_LEN),
        },
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        run_cpu_expected_output(exe, out_dir)
        # lighter timing for large, default for others
        if size_name == "large":
            ms = run_cpu_time(exe, out_dir, warmup=1, trials=3)
        else:
            ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{ms:.3f}\n")

    print(f"[gen_data] {size_name}: wrote data to {out_dir}")


if __name__ == "__main__":
    main()
