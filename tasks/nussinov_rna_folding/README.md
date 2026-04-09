# Batched Nussinov RNA Secondary-Structure Folding

This task benchmarks the classical Nussinov dynamic programming algorithm for RNA secondary-structure prediction.
For each RNA sequence, the algorithm computes the maximum number of canonical base pairs subject to a minimum
hairpin loop-length constraint. The output is a single integer score per sequence.

## Why this is a good ORBench task

- **Classic non-AI algorithm** from bioinformatics and dynamic programming
- **Clear GPU parallelism** via batched sequences and anti-diagonal / wavefront scheduling
- **Nontrivial dependency pattern** due to interval DP and split reductions
- **Simple output contract**: one integer per sequence, easy to validate

## Input format

Input is stored in `input.bin`.

Tensors:
- `seqs` (`int32`): concatenated RNA sequences encoded as 0=A, 1=C, 2=G, 3=U
- `offsets` (`int32`): length `N+1`, giving per-sequence boundaries in `seqs`

Params:
- `N`: number of RNA sequences
- `total_seq_len`: total concatenated sequence length
- `min_loop_len`: minimum hairpin loop length

## Output format

- `scores` (`int32`): one integer per input sequence
- `output.txt` contains one line per sequence

## Problem source

Nussinov, R. and Jacobson, A. (1980). Fast algorithm for predicting the secondary structure of single-stranded RNA.
Classical interval dynamic programming in bioinformatics.
