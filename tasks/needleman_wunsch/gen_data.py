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
    "small": {"len_a": 512, "len_b": 512, "alphabet_size": 16, "gap_penalty": 5, "seed": 42},
    "medium": {"len_a": 2048, "len_b": 2048, "alphabet_size": 16, "gap_penalty": 5, "seed": 42},
    "large": {"len_a": 3072, "len_b": 3072, "alphabet_size": 16, "gap_penalty": 5, "seed": 42},
}


def make_score_matrix(alphabet_size: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.integers(-3, 3, size=(alphabet_size, alphabet_size), dtype=np.int32)
    mat = ((mat + mat.T) // 2).astype(np.int32)
    diag = rng.integers(2, 6, size=alphabet_size, dtype=np.int32)
    for i in range(alphabet_size):
        mat[i, i] = int(diag[i])
    return mat.reshape(-1).astype(np.int32)


def nw_last_row(seq_a: np.ndarray, seq_b: np.ndarray, score_matrix: np.ndarray, alphabet_size: int, gap_penalty: int) -> np.ndarray:
    len_a = int(seq_a.shape[0])
    len_b = int(seq_b.shape[0])
    score = score_matrix.reshape(alphabet_size, alphabet_size)

    prev = np.empty(len_b + 1, dtype=np.int32)
    curr = np.empty(len_b + 1, dtype=np.int32)
    prev[0] = 0
    for j in range(1, len_b + 1):
        prev[j] = prev[j - 1] - gap_penalty

    for i in range(1, len_a + 1):
        curr[0] = -i * gap_penalty
        a = int(seq_a[i - 1])
        for j in range(1, len_b + 1):
            b = int(seq_b[j - 1])
            diag = int(prev[j - 1]) + int(score[a, b])
            up = int(prev[j]) - gap_penalty
            left = int(curr[j - 1]) - gap_penalty
            best = diag
            if up > best:
                best = up
            if left > best:
                best = left
            curr[j] = best
        prev, curr = curr, prev
    return prev.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("size", choices=SIZES.keys())
    ap.add_argument("out_dir")
    ap.add_argument("--with-expected", action="store_true")
    args = ap.parse_args()

    cfg = SIZES[args.size]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg["seed"])
    seq_a = rng.integers(0, cfg["alphabet_size"], size=cfg["len_a"], dtype=np.int32)
    seq_b = rng.integers(0, cfg["alphabet_size"], size=cfg["len_b"], dtype=np.int32)
    score_matrix = make_score_matrix(cfg["alphabet_size"], rng)

    tensors = [
        ("seq_a", "int32", seq_a),
        ("seq_b", "int32", seq_b),
        ("score_matrix", "int32", score_matrix),
    ]
    params = {
        "len_a": int(cfg["len_a"]),
        "len_b": int(cfg["len_b"]),
        "alphabet_size": int(cfg["alphabet_size"]),
        "gap_penalty": int(cfg["gap_penalty"]),
    }
    write_input_bin(str(out_dir / "input.bin"), tensors, params)

    if args.with_expected:
        last_row = nw_last_row(seq_a, seq_b, score_matrix, cfg["alphabet_size"], cfg["gap_penalty"])
        with open(out_dir / "expected_output.txt", "w") as f:
            for v in last_row:
                f.write(f"{int(v)}\n")


if __name__ == "__main__":
    main()
