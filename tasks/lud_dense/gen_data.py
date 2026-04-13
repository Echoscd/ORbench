#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.orbench_io_py import write_input_bin

SIZES = {
    "small": {"n": 128, "seed": 42},
    "medium": {"n": 384, "seed": 42},
    "large": {"n": 768, "seed": 42},
}


def make_lu_constructed_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    L = np.eye(n, dtype=np.int64)
    U = np.zeros((n, n), dtype=np.int64)

    # Lower entries: small integers keep values moderate.
    for i in range(1, n):
        vals = rng.integers(-2, 3, size=i, dtype=np.int64)
        L[i, :i] = vals

    # Upper matrix: diagonal are powers of two, superdiagonal entries small ints.
    for i in range(n):
        U[i, i] = 8 + 2 * (i % 4)
        if i + 1 < n:
            vals = rng.integers(-3, 4, size=(n - i - 1), dtype=np.int64)
            U[i, i + 1:] = vals

    A = (L @ U).astype(np.float64)
    return A.reshape(-1)


def lud_inplace(matrix: np.ndarray, n: int) -> np.ndarray:
    a = matrix.astype(np.float64, copy=True).reshape(n, n)
    for k in range(n):
        for j in range(k, n):
            s = 0.0
            for p in range(k):
                s += a[k, p] * a[p, j]
            a[k, j] = a[k, j] - s
        pivot = a[k, k]
        for i in range(k + 1, n):
            s = 0.0
            for p in range(k):
                s += a[i, p] * a[p, k]
            a[i, k] = (a[i, k] - s) / pivot
    return a.reshape(-1)


def compute_w_from_compact_lu(compact_lu: np.ndarray, n: int) -> np.ndarray:
    a = compact_lu.reshape(n, n)
    u1 = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(i, n):
            s += a[i, j]
        u1[i] = s

    w = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = u1[i]
        for j in range(i):
            s += a[i, j] * u1[j]
        w[i] = s
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("size", choices=SIZES.keys())
    ap.add_argument("out_dir")
    ap.add_argument("--with-expected", action="store_true")
    args = ap.parse_args()

    cfg = SIZES[args.size]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = make_lu_constructed_matrix(cfg["n"], cfg["seed"])
    write_input_bin(
        str(out_dir / "input.bin"),
        [("matrix", "float64", matrix)],
        {"n": int(cfg["n"])}
    )

    if args.with_expected:
        lu = lud_inplace(matrix, cfg["n"])
        w = compute_w_from_compact_lu(lu, cfg["n"])
        with open(out_dir / "expected_output.txt", "w") as f:
            for v in w:
                f.write(f"{float(v):.17g}\n")


if __name__ == "__main__":
    main()
