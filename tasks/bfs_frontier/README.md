# BFS Frontier Traversal

## Problem background
Breadth-First Search (BFS) is the canonical graph traversal algorithm for unweighted graphs. Starting from a source vertex, BFS discovers vertices level by level, where the level of a vertex is the minimum number of hops from the source. BFS is a basic building block in graph analytics, routing, social-network exploration, and many sparse graph workloads.

This task is adapted from the Rodinia BFS / graph-traversal benchmark family. The ORBench version focuses on the core computation: given a graph in CSR form and one source node, compute the BFS distance (hop count) from the source to every vertex.

## Algorithm source
- Rodinia Graph Traversal / BFS benchmark
- Public GitHub mirror: `yuhc/gpu-rodinia`

## Why it fits GPU acceleration
BFS on large sparse graphs has abundant parallelism inside each frontier expansion step:
- all vertices in the current frontier can be processed in parallel;
- outgoing edges of frontier vertices can be scanned in parallel;
- each newly discovered vertex performs a simple distance write / frontier insertion.

The main bottlenecks are irregular memory access, load imbalance from varying vertex degrees, and synchronization around the visited / distance arrays.

## Input format
Stored in `input.bin`:
- `row_offsets` (`int32`, length `V+1`): CSR row pointers
- `col_indices` (`int32`, length `E`): CSR adjacency list

Parameters:
- `V`: number of vertices
- `E`: number of directed edges in CSR storage
- `source`: BFS source vertex id

## Output format
`output.txt` / `expected_output.txt` contain one `int32` distance per line for all vertices `0..V-1`:
- `dist[v] =` minimum hop count from `source` to `v`
- unreachable vertices use `-1`
