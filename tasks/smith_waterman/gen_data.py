#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate Smith-Waterman sequence alignment test data.

Generates random DNA sequence pairs and computes expected local alignment scores.

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
    "small":  {"N": 1000,  "len_min": 64,  "len_max": 128, "seed": 42},
    "medium": {"N": 5000,  "len_min": 128, "len_max": 256, "seed": 42},
    "large":  {"N": 20000, "len_min": 256, "len_max": 512, "seed": 42},
}

MATCH_SCORE = 2
MISMATCH_PENALTY = -1
GAP_PENALTY = -2


def generate_sequences(N, len_min, len_max, seed):
    """Generate N pairs of random DNA sequences (encoded as 0-3)."""
    rng = np.random.default_rng(seed)

    query_seqs = []
    target_seqs = []
    query_offsets = [0]
    target_offsets = [0]

    for _ in range(N):
        qlen = rng.integers(len_min, len_max + 1)
        tlen = rng.integers(len_min, len_max + 1)
        q = rng.integers(0, 4, size=qlen, dtype=np.int32)
        t = rng.integers(0, 4, size=tlen, dtype=np.int32)
        query_seqs.append(q)
        target_seqs.append(t)
        query_offsets.append(query_offsets[-1] + qlen)
        target_offsets.append(target_offsets[-1] + tlen)

    return (
        np.concatenate(query_seqs).astype(np.int32),
        np.concatenate(target_seqs).astype(np.int32),
        np.array(query_offsets, dtype=np.int32),
        np.array(target_offsets, dtype=np.int32),
    )


def smith_waterman_python(query, target, match, mismatch, gap):
    """Pure Python Smith-Waterman for verification."""
    m, n = len(query), len(target)
    prev = [0] * (n + 1)
    best = 0
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        qi = query[i - 1]
        for j in range(1, n + 1):
            s = match if qi == target[j - 1] else mismatch
            val = max(0, prev[j - 1] + s, prev[j] + gap, curr[j - 1] + gap)
            curr[j] = val
            if val > best:
                best = val
        prev = curr
    return best


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "smith_waterman" / "solution_cpu"
    src = orbench_root / "tasks" / "smith_waterman" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "smith_waterman" / "task_io_cpu.c"
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

    print(f"[gen_data] Generating {size_name}: {N} sequence pairs...")

    query_seqs, target_seqs, query_offsets, target_offsets = generate_sequences(
        N, cfg["len_min"], cfg["len_max"], seed
    )

    total_query_len = len(query_seqs)
    total_target_len = len(target_seqs)
    print(f"  {N} pairs, total_query_len={total_query_len}, total_target_len={total_target_len}")

    # Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("query_seqs", "int32", query_seqs),
            ("target_seqs", "int32", target_seqs),
            ("query_offsets", "int32", query_offsets),
            ("target_offsets", "int32", target_offsets),
        ],
        params={
            "N": N,
            "total_query_len": total_query_len,
            "total_target_len": total_target_len,
            "match_score": MATCH_SCORE,
            "mismatch_penalty": MISMATCH_PENALTY,
            "gap_penalty": GAP_PENALTY,
        },
    )

    # Dummy requests (all pairs computed at once)
    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        # Use Python reference to generate expected output (no task_io_cpu.c needed)
        print(f"  Computing expected output via Python reference...")
        scores = []
        for p in range(N):
            qs = query_offsets[p]
            qe = query_offsets[p + 1]
            ts = target_offsets[p]
            te = target_offsets[p + 1]
            s = smith_waterman_python(
                query_seqs[qs:qe], target_seqs[ts:te],
                MATCH_SCORE, MISMATCH_PENALTY, GAP_PENALTY,
            )
            scores.append(s)
            if (p + 1) % 500 == 0:
                print(f"    {p + 1}/{N} pairs done")

        with open(out_dir / "expected_output.txt", "w") as f:
            for s in scores:
                f.write(f"{s}\n")

        print(f"[gen_data] {size_name}: wrote all files in {out_dir}")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin in {out_dir}")


if __name__ == "__main__":
    main()
