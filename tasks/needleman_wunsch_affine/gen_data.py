#!/usr/bin/env python3
"""
gen_data.py -- Generate batched affine-gap global alignment data.

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
    "small":  {"N": 800,  "len_min": 80,  "len_max": 144, "seed": 42},
    "medium": {"N": 2500, "len_min": 128, "len_max": 224, "seed": 42},
    "large":  {"N": 5000, "len_min": 224, "len_max": 384, "seed": 42},
}

MATCH_SCORE = 2
MISMATCH_PENALTY = -1
GAP_OPEN_PENALTY = -3
GAP_EXTEND_PENALTY = -1


def generate_sequences(N, len_min, len_max, seed):
    rng = np.random.default_rng(seed)
    qseqs, tseqs = [], []
    qoff = [0]
    toff = [0]
    for _ in range(N):
        qlen = int(rng.integers(len_min, len_max + 1))
        tlen = int(rng.integers(len_min, len_max + 1))
        q = rng.integers(0, 4, size=qlen, dtype=np.int32)
        t = rng.integers(0, 4, size=tlen, dtype=np.int32)
        qseqs.append(q)
        tseqs.append(t)
        qoff.append(qoff[-1] + qlen)
        toff.append(toff[-1] + tlen)
    return (
        np.concatenate(qseqs).astype(np.int32),
        np.concatenate(tseqs).astype(np.int32),
        np.array(qoff, dtype=np.int32),
        np.array(toff, dtype=np.int32),
    )


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "needleman_wunsch_affine" / "solution_cpu"
    src = orbench_root / "tasks" / "needleman_wunsch_affine" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "needleman_wunsch_affine" / "task_io_cpu.c"
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

    cfg = SIZES[size_name]
    qseqs, tseqs, qoff, toff = generate_sequences(
        cfg["N"], cfg["len_min"], cfg["len_max"], cfg["seed"]
    )
    total_query_len = int(len(qseqs))
    total_target_len = int(len(tseqs))

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("query_seqs", "int32", qseqs),
            ("target_seqs", "int32", tseqs),
            ("query_offsets", "int32", qoff),
            ("target_offsets", "int32", toff),
        ],
        params={
            "N": int(cfg["N"]),
            "total_query_len": total_query_len,
            "total_target_len": total_target_len,
            "match_score": MATCH_SCORE,
            "mismatch_penalty": MISMATCH_PENALTY,
            "gap_open_penalty": GAP_OPEN_PENALTY,
            "gap_extend_penalty": GAP_EXTEND_PENALTY,
        },
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        run_cpu_expected_output(exe, out_dir)
        ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{ms:.3f}\n")

    print(f"[gen_data] {size_name}: wrote data to {out_dir}")


if __name__ == "__main__":
    main()
