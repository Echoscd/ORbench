#!/usr/bin/env python3
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.orbench_io_py import write_input_bin


def build_bplustree(num_records: int, order: int):
    max_keys = order - 1
    keys_all = np.arange(2, 2 * num_records + 2, 2, dtype=np.int32)
    vals_all = (np.arange(num_records, dtype=np.int32) * 3 + 7).astype(np.int32)

    nodes = []
    level = []

    # Leaves
    for start in range(0, num_records, max_keys):
        end = min(start + max_keys, num_records)
        count = end - start
        nk = np.full(max_keys, -1, dtype=np.int32)
        nv = np.full(max_keys, -1, dtype=np.int32)
        nk[:count] = keys_all[start:end]
        nv[:count] = vals_all[start:end]
        node = {
            "is_leaf": 1,
            "count": count,
            "keys": nk,
            "children": np.full(order, -1, dtype=np.int32),
            "values": nv,
            "min_key": int(nk[0]),
        }
        idx = len(nodes)
        nodes.append(node)
        level.append(idx)

    # Internal levels
    while len(level) > 1:
        next_level = []
        for gstart in range(0, len(level), order):
            group = level[gstart:gstart + order]
            child_count = len(group)
            key_count = child_count - 1
            nk = np.full(max_keys, -1, dtype=np.int32)
            ch = np.full(order, -1, dtype=np.int32)
            for i, child_idx in enumerate(group):
                ch[i] = child_idx
                if i > 0:
                    nk[i - 1] = nodes[child_idx]["min_key"]
            node = {
                "is_leaf": 0,
                "count": key_count,
                "keys": nk,
                "children": ch,
                "values": np.full(max_keys, -1, dtype=np.int32),
                "min_key": nodes[group[0]]["min_key"],
            }
            idx = len(nodes)
            nodes.append(node)
            next_level.append(idx)
        level = next_level

    root_idx = level[0]
    num_nodes = len(nodes)

    is_leaf = np.array([n["is_leaf"] for n in nodes], dtype=np.int32)
    key_counts = np.array([n["count"] for n in nodes], dtype=np.int32)
    keys = np.concatenate([n["keys"] for n in nodes]).astype(np.int32)
    children = np.concatenate([n["children"] for n in nodes]).astype(np.int32)
    values = np.concatenate([n["values"] for n in nodes]).astype(np.int32)
    return {
        "keys_all": keys_all,
        "vals_all": vals_all,
        "is_leaf": is_leaf,
        "key_counts": key_counts,
        "keys": keys,
        "children": children,
        "values": values,
        "root_idx": root_idx,
        "num_nodes": num_nodes,
        "max_keys": max_keys,
    }


def make_queries(keys_all: np.ndarray, vals_all: np.ndarray, num_queries: int, seed: int):
    rng = np.random.default_rng(seed)
    hit_mask = rng.random(num_queries) < 0.8
    queries = np.empty(num_queries, dtype=np.int32)
    expected = np.empty(num_queries, dtype=np.int32)

    hit_count = int(hit_mask.sum())
    miss_count = num_queries - hit_count

    if hit_count > 0:
        idx = rng.integers(0, len(keys_all), size=hit_count, dtype=np.int64)
        queries[hit_mask] = keys_all[idx]
        expected[hit_mask] = vals_all[idx]
    if miss_count > 0:
        miss_idx = rng.integers(0, len(keys_all), size=miss_count, dtype=np.int64)
        queries[~hit_mask] = keys_all[miss_idx] + 1  # odd => guaranteed absent
        expected[~hit_mask] = -1
    return queries, expected


SIZES = {
    "small":  {"num_records": 32768,   "num_queries": 20000,  "order": 16, "seed": 42},
    "medium": {"num_records": 262144,  "num_queries": 200000, "order": 16, "seed": 42},
    "large":  {"num_records": 1048576, "num_queries": 800000, "order": 16, "seed": 42},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("size", choices=SIZES.keys())
    ap.add_argument("out_dir")
    ap.add_argument("--with-expected", action="store_true")
    args = ap.parse_args()

    cfg = SIZES[args.size]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tree = build_bplustree(cfg["num_records"], cfg["order"])
    queries, expected = make_queries(tree["keys_all"], tree["vals_all"], cfg["num_queries"], cfg["seed"])

    tensors = [
        ("is_leaf", "int32", tree["is_leaf"]),
        ("key_counts", "int32", tree["key_counts"]),
        ("keys", "int32", tree["keys"]),
        ("children", "int32", tree["children"]),
        ("values", "int32", tree["values"]),
        ("query_keys", "int32", queries),
    ]
    params = {
        "num_nodes": int(tree["num_nodes"]),
        "order": int(cfg["order"]),
        "max_keys": int(tree["max_keys"]),
        "root_idx": int(tree["root_idx"]),
        "num_queries": int(cfg["num_queries"]),
        "num_records": int(cfg["num_records"]),
    }

    write_input_bin(str(out_dir / "input.bin"), tensors, params)

    if args.with_expected:
        with open(out_dir / "expected_output.txt", "w") as f:
            for v in expected:
                f.write(f"{int(v)}\n")


if __name__ == "__main__":
    main()
