# PageRank Pull (Jacobi) for ORBench

## Problem background
PageRank computes an importance score for each vertex in a directed graph by repeatedly propagating score mass along outgoing edges. It is a classic graph analytics kernel used in web search, citation analysis, and network ranking.

This ORBench task is adapted from the GAP Benchmark Suite PageRank kernel, specifically the pull-direction iterative implementation that updates all vertices from their incoming neighbors.

## Algorithm source
- GAP Benchmark Suite (GAPBS)
- Public repo: `sbeamer/gapbs`
- Kernel family: `PageRank (PR)`
- Style used here: pull-direction, Jacobi-style iterative update

## Why it is suitable for GPU acceleration
Each PageRank iteration updates every vertex independently once the previous iteration's scores are available. This exposes abundant data parallelism:
- one thread (or warp) can process one destination vertex;
- incoming-neighbor reductions dominate runtime and can be parallelized;
- graph structure is read-only and can stay on device across repeated runs.

The main bottlenecks are irregular memory access over sparse graph edges and load imbalance from skewed in-degrees.

## Input format
Stored in `input.bin`.

Parameters:
- `V` (`int`): number of vertices
- `E` (`int`): number of directed edges
- `max_iters` (`int`): maximum PageRank iterations

Tensors:
- `in_row_offsets` (`int32`, shape `[V+1]`): inbound CSR row offsets
- `in_col_indices` (`int32`, shape `[E]`): source vertices for each inbound edge
- `out_degree` (`int32`, shape `[V]`): outgoing degree of each vertex
- `fparams` (`float32`, shape `[2]`): `[damping, epsilon]`

All arrays are stored in row-major flat form.

## Output format
`output.txt` contains `V` lines, one PageRank score per vertex (float), after convergence or after `max_iters` iterations.

## Notes on this adaptation
The original GAPBS benchmark is a full graph benchmark suite. This ORBench task isolates the PageRank pull kernel into a reusable `solution_init / solution_compute / solution_free` interface.
