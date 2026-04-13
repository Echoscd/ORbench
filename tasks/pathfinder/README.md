# Pathfinder Grid Dynamic Programming

## Problem Background

Pathfinder computes the minimum accumulated path cost through a 2D cost grid.
Starting from any cell in the first row, the path advances one row at a time.
At each step, it may move straight down, diagonally down-left, or diagonally down-right.
The accumulated cost of a cell is its own weight plus the minimum accumulated cost
among the reachable predecessor cells in the previous row.

This is a classic dynamic programming stencil-like computation. It appears in the
Rodinia benchmark suite as a representative irregular/non-ML GPU workload.

## Algorithm Source

This ORBench task is adapted from the real GitHub implementation in:
- Rodinia benchmark suite (GitHub mirror): `yuhc/gpu-rodinia`
- Source path: `openmp/pathfinder/pathfinder.cpp`
- Benchmark page: Rodinia Pathfinder

The original Rodinia code initializes a random integer grid and computes the final
DP row by iterating row by row.

## Why It Fits GPU Acceleration

The dependency is sequential across rows, but **all columns within the same row are
independent once the previous row is known**. This gives substantial per-row parallelism:

1. **Column-parallel update:** one thread per column.
2. **Regular memory access:** each update reads at most three neighboring values from
   the previous row and one value from the current row.
3. **Low arithmetic intensity but high throughput opportunity:** performance depends on
   memory coalescing, buffering of the previous/current rows, and avoiding extra transfers.
4. **Wavefront-style DP:** a common GPU pattern where inter-row dependence remains,
   but intra-row work is embarrassingly parallel.

## Input Format

Binary file `input.bin` (ORBench v2 format):

| Tensor | Type | Size | Description |
|--------|------|------|-------------|
| `wall` | int32 | `rows * cols` | Row-major grid of nonnegative cell costs |

| Parameter | Type | Description |
|-----------|------|-------------|
| `rows` | int64 | Number of grid rows |
| `cols` | int64 | Number of grid columns |

Interpretation:
- `wall[r * cols + c]` is the cost at row `r`, column `c`
- The initial DP row is exactly the first row of `wall`
- For row `r > 0`,
  `dp[r][c] = wall[r][c] + min(dp[r-1][c-1], dp[r-1][c], dp[r-1][c+1])`
  with boundary checks at the left/right edges

## Output Format

File `expected_output.txt` contains `cols` lines.
Each line is one integer: the final accumulated cost for that column in the last row.

## Data Sizes

| Size | Rows | Cols | Cells | Input bytes |
|------|------|------|-------|-------------|
| small | 1024 | 2048 | 2,097,152 | ~8 MB |
| medium | 4096 | 4096 | 16,777,216 | ~64 MB |
| large | 8192 | 8192 | 67,108,864 | ~256 MB |

These sizes keep the problem simple while providing enough parallel work per row
for a GPU implementation to matter.
