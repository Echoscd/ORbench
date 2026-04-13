# Needleman-Wunsch Wavefront Alignment Score

## Background

Needleman-Wunsch is a classic dynamic-programming algorithm for global sequence alignment in bioinformatics. Given two sequences and a substitution score table, it fills a 2D DP matrix where each cell depends on its northwest, north, and west neighbors. The final row (and in particular the bottom-right cell) summarizes optimal global alignment scores under the chosen substitution and gap penalties.

This task is adapted from the Rodinia Needleman-Wunsch benchmark. The original benchmark emphasizes the wavefront / anti-diagonal dependency pattern that makes the DP fill suitable for parallel accelerators.

## Why it fits GPU acceleration

The DP matrix cannot be filled fully in parallel because of data dependencies, but **each anti-diagonal can be processed in parallel** once the previous anti-diagonal is complete. This creates a well-structured wavefront parallel pattern:

- Each cell update is independent within one anti-diagonal
- Memory access is regular over the DP matrix / rolling frontier
- Large sequence lengths expose substantial fine-grained parallelism
- Shared memory / tiling can improve locality for block-wise implementations

## ORBench adaptation

The Rodinia benchmark includes traceback-oriented full-matrix computation. For ORBench, this task focuses on the **score-propagation phase** and validates the **entire final DP row**. This keeps the task deterministic and easy to validate while preserving the core wavefront computation.

## Input format

All tensors are stored in `input.bin`.

Parameters:
- `len_a` : length of sequence A
- `len_b` : length of sequence B
- `alphabet_size` : symbol vocabulary size
- `gap_penalty` : constant linear gap penalty

Tensors:
- `seq_a` : int32 array of length `len_a`, symbols in `[0, alphabet_size)`
- `seq_b` : int32 array of length `len_b`, symbols in `[0, alphabet_size)`
- `score_matrix` : int32 array of length `alphabet_size * alphabet_size`, flattened row-major substitution matrix where `score_matrix[x * alphabet_size + y]` is the substitution score for aligning symbol `x` with symbol `y`

## Output format

- `output.txt` contains `len_b + 1` lines
- Line `j` is the DP score of the final row at column `j`
- Exact integer match is required

## Real source

This task is based on the Rodinia Needleman-Wunsch benchmark and its public GitHub mirror/fork `yuhc/gpu-rodinia`.
