# Smith-Waterman Local Sequence Alignment

## Problem Background

The Smith-Waterman algorithm is the gold-standard method for local sequence alignment in bioinformatics. Given two biological sequences (DNA, RNA, or protein), it finds the highest-scoring local alignment — the pair of subsequences with the best match. Unlike global alignment (Needleman-Wunsch), Smith-Waterman can identify conserved regions within otherwise dissimilar sequences.

The algorithm fills an (m+1) x (n+1) scoring matrix using dynamic programming, where each cell H[i][j] represents the best local alignment score ending at positions i and j. The recurrence is:

```
H[i][j] = max(
    0,
    H[i-1][j-1] + score(query[i], target[j]),  // match/mismatch
    H[i-1][j]   + gap_penalty,                  // gap in target
    H[i][j-1]   + gap_penalty                   // gap in query
)
```

The optimal local alignment score is max(H[i][j]) over all i, j.

## Algorithm Source

- Original paper: Smith & Waterman, "Identification of Common Molecular Subsequences", J. Mol. Biol. (1981)
- Textbook: Durbin et al., "Biological Sequence Analysis" (1998)
- Industry: BLAST-like tools (NCBI), sequence database search, variant calling

## Why GPU Acceleration

1. **Inter-pair parallelism**: Each sequence pair alignment is independent — thousands of pairs can execute simultaneously on different thread blocks.
2. **Intra-pair parallelism (anti-diagonal wavefront)**: Within each DP matrix, cells on the same anti-diagonal are independent and can be computed in parallel by threads within a block.
3. **Compute-bound**: O(m*n) arithmetic per pair with simple memory access patterns — ideal for GPU ALUs.
4. **Shared memory tiling**: The DP matrix can be tiled into blocks that fit in shared memory, with only border values communicated between tiles.

## Input Format

Binary file `input.bin` (ORBench v2 format):

| Tensor | Type | Size | Description |
|--------|------|------|-------------|
| `query_seqs` | int32 | total_query_len | Concatenated query sequences (0=A, 1=C, 2=G, 3=T) |
| `target_seqs` | int32 | total_target_len | Concatenated target sequences |
| `query_offsets` | int32 | N+1 | CSR offsets into query_seqs |
| `target_offsets` | int32 | N+1 | CSR offsets into target_seqs |

| Parameter | Type | Description |
|-----------|------|-------------|
| `N` | int64 | Number of sequence pairs |
| `total_query_len` | int64 | Total length of all query sequences |
| `total_target_len` | int64 | Total length of all target sequences |
| `match_score` | int64 | Score for matching bases (default: 2) |
| `mismatch_penalty` | int64 | Penalty for mismatch (default: -1, stored as -1) |
| `gap_penalty` | int64 | Penalty for gap (default: -2, stored as -2) |

## Output Format

File `expected_output.txt`: N lines, each containing one integer — the optimal local alignment score for that pair.

```
Format: "%d\n" per score
Correctness: exact match (integer scores)
```

## Data Sizes

| Size | N (pairs) | Query length | Target length |
|------|-----------|-------------|---------------|
| small | 1000 | 64-128 | 64-128 |
| medium | 5000 | 128-256 | 128-256 |
| large | 20000 | 256-512 | 256-512 |
