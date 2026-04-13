#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
FRAMEWORK_DIR = THIS_DIR.parent.parent / "framework"
sys.path.append(str(FRAMEWORK_DIR))
from orbench_io_py import write_input_bin

SIZES: Dict[str, Dict[str, int]] = {
    "small":  {"num_rows": 4096,   "num_cols": 4096,   "target_nnz": 65536,   "max_row_nnz": 64,  "seed": 17},
    "medium": {"num_rows": 32768,  "num_cols": 32768,  "target_nnz": 786432,  "max_row_nnz": 96,  "seed": 17},
    "large":  {"num_rows": 131072, "num_cols": 131072, "target_nnz": 3932160, "max_row_nnz": 128, "seed": 17},
}


def make_row_lengths(num_rows: int, target_nnz: int, max_row_nnz: int, rng: np.random.Generator) -> np.ndarray:
    avg = target_nnz / float(num_rows)
    base = rng.poisson(lam=max(1.0, avg * 0.75), size=num_rows) + 1
    spikes_mask = rng.random(num_rows) < 0.07
    spike_mult = rng.integers(2, 8, size=num_rows)
    base[spikes_mask] *= spike_mult[spikes_mask]
    lengths = np.clip(base, 1, max_row_nnz).astype(np.int32)

    current = int(lengths.sum())
    target = int(target_nnz)
    # Deterministic correction to match target nnz exactly.
    idx = 0
    while current < target:
        if lengths[idx] < max_row_nnz:
            lengths[idx] += 1
            current += 1
        idx += 1
        if idx == num_rows:
            idx = 0
    idx = 0
    while current > target:
        if lengths[idx] > 1:
            lengths[idx] -= 1
            current -= 1
        idx += 1
        if idx == num_rows:
            idx = 0
    return lengths


def build_sparse_rows(num_rows: int, num_cols: int, lengths: np.ndarray, rng: np.random.Generator) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    row_cols: List[np.ndarray] = []
    row_vals: List[np.ndarray] = []
    for i in range(num_rows):
        k = int(lengths[i])
        cols = rng.integers(0, num_cols, size=k, dtype=np.int32)
        vals = rng.uniform(-1.0, 1.0, size=k).astype(np.float32)
        row_cols.append(cols)
        row_vals.append(vals)
    return row_cols, row_vals


def convert_to_jds(lengths: np.ndarray, row_cols: List[np.ndarray], row_vals: List[np.ndarray]):
    num_rows = len(lengths)
    perm = np.argsort(-lengths, kind="stable").astype(np.int32)
    row_nnz_sorted = lengths[perm].astype(np.int32)
    num_diags = int(row_nnz_sorted[0])

    jad_ptr = [0]
    packed_cols: List[np.ndarray] = []
    packed_vals: List[np.ndarray] = []

    active = num_rows
    for d in range(num_diags):
        while active > 0 and int(row_nnz_sorted[active - 1]) <= d:
            active -= 1
        diag_cols = np.empty(active, dtype=np.int32)
        diag_vals = np.empty(active, dtype=np.float32)
        for rp in range(active):
            orig_r = int(perm[rp])
            diag_cols[rp] = row_cols[orig_r][d]
            diag_vals[rp] = row_vals[orig_r][d]
        packed_cols.append(diag_cols)
        packed_vals.append(diag_vals)
        jad_ptr.append(jad_ptr[-1] + active)

    col_idx = np.concatenate(packed_cols).astype(np.int32, copy=False)
    values = np.concatenate(packed_vals).astype(np.float32, copy=False)
    return perm, row_nnz_sorted, np.asarray(jad_ptr, dtype=np.int32), col_idx, values, num_diags


def cpu_reference(num_rows: int, row_cols: List[np.ndarray], row_vals: List[np.ndarray], x: np.ndarray) -> np.ndarray:
    y = np.zeros(num_rows, dtype=np.float32)
    for r in range(num_rows):
        cols = row_cols[r]
        vals = row_vals[r]
        s = np.float32(0.0)
        for j in range(len(cols)):
            s = np.float32(s + vals[j] * x[cols[j]])
        y[r] = s
    return y


def write_expected(out_path: Path, y: np.ndarray) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for v in y:
            f.write(f"{float(v):.6e}\n")


def generate(size: str, out_dir: Path, with_expected: bool = True) -> None:
    cfg = SIZES[size]
    num_rows = int(cfg["num_rows"])
    num_cols = int(cfg["num_cols"])
    target_nnz = int(cfg["target_nnz"])
    max_row_nnz = int(cfg["max_row_nnz"])
    seed = int(cfg["seed"])

    rng = np.random.default_rng(seed)
    lengths = make_row_lengths(num_rows, target_nnz, max_row_nnz, rng)
    row_cols, row_vals = build_sparse_rows(num_rows, num_cols, lengths, rng)
    x = rng.uniform(-1.0, 1.0, size=num_cols).astype(np.float32)

    perm, row_nnz_sorted, jad_ptr, col_idx, values, num_diags = convert_to_jds(lengths, row_cols, row_vals)
    assert int(values.size) == target_nnz

    out_dir.mkdir(parents=True, exist_ok=True)
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("jad_ptr", "int32", jad_ptr),
            ("col_idx", "int32", col_idx),
            ("values", "float32", values),
            ("perm", "int32", perm),
            ("row_nnz", "int32", row_nnz_sorted),
            ("x", "float32", x),
        ],
        params={
            "num_rows": num_rows,
            "num_cols": num_cols,
            "nnz": int(target_nnz),
            "num_diags": int(num_diags),
        },
    )

    if with_expected:
        y = cpu_reference(num_rows, row_cols, row_vals, x)
        write_expected(out_dir / "expected_output.txt", y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ORBench data for Parboil-style JDS SpMV")
    parser.add_argument("size", choices=sorted(SIZES.keys()))
    parser.add_argument("out_dir")
    parser.add_argument("--with-expected", action="store_true")
    args = parser.parse_args()
    generate(args.size, Path(args.out_dir), with_expected=args.with_expected)


if __name__ == "__main__":
    main()
