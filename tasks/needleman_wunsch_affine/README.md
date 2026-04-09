# Needleman-Wunsch Global Alignment with Affine Gap Penalties

This task benchmarks batched **global sequence alignment** with an **affine gap penalty** model.
Compared with basic edit distance or constant-gap alignment, affine gaps use three DP states
(match/mismatch, gap-in-query, gap-in-target), making the dependency structure richer and the
GPU parallelization challenge more interesting.

## Problem
For each sequence pair `(query, target)`, compute the optimal global alignment score under:
- match reward
- mismatch penalty
- gap-open penalty
- gap-extend penalty

The recurrence uses three DP states:
- `M[i,j]`: alignment ends with matching/mismatching `query[i-1]` and `target[j-1]`
- `X[i,j]`: alignment ends with a gap in the target (vertical move)
- `Y[i,j]`: alignment ends with a gap in the query (horizontal move)

Output one integer score per sequence pair.

## Why it is good for GPU benchmarking
- The algorithm has clear **anti-diagonal / wavefront dependencies**
- It is harder than constant-gap Smith-Waterman because it maintains **three coupled DP states**
- Batched alignment provides coarse-grained parallelism, while each matrix provides fine-grained DP structure
- Input sizes are naturally controllable by batch size and sequence lengths

## Input format
Tensors in `input.bin`:
- `query_seqs` (`int32`): concatenated query sequences
- `target_seqs` (`int32`): concatenated target sequences
- `query_offsets` (`int32`, length `N+1`): offsets into `query_seqs`
- `target_offsets` (`int32`, length `N+1`): offsets into `target_seqs`

Integer params:
- `N`: number of sequence pairs
- `total_query_len`
- `total_target_len`
- `match_score`
- `mismatch_penalty`
- `gap_open_penalty`
- `gap_extend_penalty`

Sequences are encoded as integers in `[0, 3]` (DNA alphabet).

## Output format
`output.txt` / `expected_output.txt` contains one line per pair:
- global alignment score (`int`)

## Source
Needleman, S. B., & Wunsch, C. D. (1970). A general method applicable to the search for similarities
in the amino acid sequence of two proteins. *Journal of Molecular Biology*.
Affine gap modeling is the standard Gotoh-style extension used throughout bioinformatics.
