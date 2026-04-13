# LUD Dense Matrix Factorization

## Background
LU decomposition factorizes a square matrix `A` into a lower-triangular matrix `L` and an upper-triangular matrix `U`, typically written as `A = L * U`. It is a classic dense linear-algebra primitive used in solving linear systems, computing determinants, and as a building block for many numerical methods.

This task is adapted from the Rodinia **LUD** benchmark. Rodinia / SPEC ACCEL describe it as a dense linear-algebra kernel whose core computation is LU decomposition on a generated dense matrix.

## Why it fits GPU acceleration
The algorithm proceeds panel by panel, but each pivot step contains substantial parallel work:
- updating the current row of `U`
- updating the current column of `L`
- updating the trailing submatrix with many independent rank-1 update operations

The main bottleneck is the repeated dense trailing-matrix update, which exposes `O(n^2)` fine-grained parallelism per pivot and strong spatial locality when tiled.

## Input
- `matrix` (`float64`, length `n*n`): input dense matrix in row-major order
- parameter `n`: matrix dimension

The generated matrices are guaranteed to admit LU decomposition without pivoting. They are constructed as an exact product of an integer unit-lower matrix and an integer upper-triangular matrix.

## Output
Return the vector `w = (L * U) * 1`, where `1` is the all-ones vector and `L,U` are the factors recovered by the decomposition. Equivalently, this is the row-sum vector of the original matrix, but the benchmark requires computing it from the decomposed form.

The output buffer has length `n` and stores `float64` values.
