# Auction Algorithm for Dense Linear Assignment (LAP)

## Background

Given an `n x n` profit matrix `P`, the dense linear assignment problem asks for a one-to-one matching between rows and columns that maximizes total profit:

\[
\max_{\pi} \sum_{i=0}^{n-1} P[i, \pi(i)]
\]

where `π` is a permutation of the columns.

This ORBench task uses the **auction algorithm**: each unassigned row bids for its most profitable column under the current column prices, and columns are repeatedly re-assigned to the highest bidders until every row owns exactly one column.

## Source

This task is adapted from public GitHub implementations of Bertsekas' auction algorithm:

- `bkj/auction-lap` — a PyTorch implementation of LAP via the auction algorithm (`auction_lap.py`)
- `bkj/cuda_auction` — a lower-level CUDA implementation of the same method

## Why it fits GPU acceleration

The core bidding phase has strong data parallelism:

1. **Row-parallel bidding**: each currently unassigned row independently scans all columns to find its best and second-best adjusted values.
2. **Column-parallel winner selection**: for each column, all incoming bids are reduced to the highest bidder.
3. **Repeated frontier-like iterations**: assignment state evolves over rounds until all rows are matched.

The main bottlenecks are dense matrix access and reductions over competing bids, both of which map naturally to GPU kernels.

## Input format

`input.bin` contains:

- Tensor `profit` (`int32`, length `n*n`): row-major dense profit matrix
- Param `n` (`int64`): matrix dimension

Interpretation:

- Row `i` is agent `i`
- Column `j` is task/object `j`
- `profit[i*n + j]` is the profit if row `i` is assigned to column `j`

## Output format

`output.txt` contains exactly **one line**:

- the maximum total assignment profit as a signed 64-bit integer

## Interface

```c
extern "C" void solution_init(int n, const int* h_profit);
extern "C" void solution_compute(long long* out_total_profit);
extern "C" void solution_free(void);
```

## Size presets

- **small**: `n = 256`
- **medium**: `n = 768`
- **large**: `n = 1536`

All three are dense integer LAP instances with deterministic seeds.
