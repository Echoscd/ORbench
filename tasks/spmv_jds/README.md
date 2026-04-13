# Parboil SpMV (JDS) — Sparse Matrix × Dense Vector

## Background

Sparse matrix-vector multiplication (SpMV) computes `y = A x` where `A` is sparse and `x` is a dense vector. It is one of the core kernels in iterative solvers, graph analytics, PDE discretizations, and many scientific computing pipelines.

This task is adapted from the **Parboil SpMV benchmark**. In the original benchmark, the sparse matrix is read in coordinate format and converted to **JDS (Jagged Diagonal Storage)**, a layout chosen because it is better suited to parallel/vector processors than plain CSR for this workload.

## Why it fits GPU acceleration

SpMV has abundant row-level parallelism but also challenging irregular memory access:

- different rows have different nonzero counts;
- accesses to the dense vector are indirect and data-dependent;
- the kernel is typically memory-bandwidth bound.

JDS helps regularize accesses to sparse values/indices and improve load balance across adjacent rows after permutation by row length.

## Inputs

All tensors are stored in `input.bin`.

- `jad_ptr` (`int32`, length = `num_diags + 1`): offset of each jagged diagonal in the packed JDS arrays.
- `col_idx` (`int32`, length = `nnz`): column index for each stored nonzero.
- `values` (`float32`, length = `nnz`): nonzero values.
- `perm` (`int32`, length = `num_rows`): maps **permuted JDS row index → original row index**.
- `row_nnz` (`int32`, length = `num_rows`): nonzero count of each **permuted** row, sorted in non-increasing order.
- `x` (`float32`, length = `num_cols`): dense input vector.

Scalar parameters:

- `num_rows` (`int64`)
- `num_cols` (`int64`)
- `nnz` (`int64`)
- `num_diags` (`int64`)

## Outputs

- `y` (`float32`, length = `num_rows`): the dense result vector in **original row order**.
- Output is written as one float per line to `output.txt`.

## Source

- Parboil benchmark suite
- SPEC ACCEL 112.spmv benchmark description
- Public GitHub mirror/fork: `yuhc/gpu-parboil`

## Files

```text
spmv_jds/
├── README.md
├── task.json
├── prompt_template.yaml
├── gen_data.py
├── cpu_reference.c
├── task_io_cpu.c
├── task_io.cu
└── data/
    ├── small/
    ├── medium/
    └── large/
```
