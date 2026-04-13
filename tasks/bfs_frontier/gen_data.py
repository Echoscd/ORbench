#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from framework.orbench_io_py import write_input_bin

SIZES = {
    'small':  {'V': 5000,   'avg_degree': 8, 'seed': 42},
    'medium': {'V': 100000, 'avg_degree': 8, 'seed': 42},
    'large':  {'V': 400000, 'avg_degree': 8, 'seed': 42},
}


def generate_connected_undirected_graph(V: int, avg_degree: int, seed: int):
    rng = np.random.default_rng(seed)
    target_undirected = max(V - 1, V * avg_degree // 2)

    edges = set()

    # random spanning tree to guarantee connectivity
    parents = rng.integers(0, np.arange(1, V), size=V - 1, endpoint=False)
    for child, parent in enumerate(parents, start=1):
        u = int(parent)
        v = int(child)
        if u > v:
            u, v = v, u
        edges.add((u, v))

    while len(edges) < target_undirected:
        u = int(rng.integers(0, V))
        v = int(rng.integers(0, V))
        if u == v:
            continue
        if u > v:
            u, v = v, u
        edges.add((u, v))

    deg = np.zeros(V, dtype=np.int64)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    row_offsets = np.empty(V + 1, dtype=np.int32)
    row_offsets[0] = 0
    np.cumsum(deg, out=row_offsets[1:])

    E = int(row_offsets[-1])
    col_indices = np.empty(E, dtype=np.int32)
    cursor = row_offsets[:-1].astype(np.int64).copy()

    for u, v in edges:
        col_indices[cursor[u]] = v
        cursor[u] += 1
        col_indices[cursor[v]] = u
        cursor[v] += 1

    for u in range(V):
        s = row_offsets[u]
        e = row_offsets[u + 1]
        if e - s > 1:
            col_indices[s:e].sort()

    return row_offsets, col_indices


def bfs_reference(V: int, row_offsets: np.ndarray, col_indices: np.ndarray, source: int):
    dist = np.full(V, -1, dtype=np.int32)
    q = deque([source])
    dist[source] = 0
    while q:
        u = q.popleft()
        du = int(dist[u])
        for idx in range(int(row_offsets[u]), int(row_offsets[u + 1])):
            v = int(col_indices[idx])
            if dist[v] == -1:
                dist[v] = du + 1
                q.append(v)
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('size', choices=SIZES.keys())
    ap.add_argument('out_dir')
    ap.add_argument('--with-expected', action='store_true')
    args = ap.parse_args()

    cfg = SIZES[args.size]
    V = int(cfg['V'])
    avg_degree = int(cfg['avg_degree'])
    seed = int(cfg['seed'])
    source = 0

    os.makedirs(args.out_dir, exist_ok=True)

    row_offsets, col_indices = generate_connected_undirected_graph(V, avg_degree, seed)
    E = int(col_indices.size)

    write_input_bin(
        os.path.join(args.out_dir, 'input.bin'),
        tensors=[
            ('row_offsets', 'int32', row_offsets),
            ('col_indices', 'int32', col_indices),
        ],
        params={
            'V': V,
            'E': E,
            'source': source,
        },
    )

    if args.with_expected:
        dist = bfs_reference(V, row_offsets, col_indices, source)
        with open(os.path.join(args.out_dir, 'expected_output.txt'), 'w') as f:
            for x in dist:
                f.write(f'{int(x)}\n')


if __name__ == '__main__':
    main()
