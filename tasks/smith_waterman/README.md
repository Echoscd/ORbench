# Smith-Waterman Local Sequence Alignment

## Problem Background

The Smith-Waterman algorithm is the classic dynamic-programming method for **local** sequence alignment. Given two biological sequences (DNA, RNA, or protein), it finds the highest-scoring aligned subsequences rather than forcing a full end-to-end alignment. This makes it especially useful for detecting conserved local regions inside otherwise dissimilar sequences.

For a query sequence `Q` of length `m` and a target sequence `T` of length `n`, define a DP table `H` with shape `(m+1) x (n+1)` and initialize row 0 and column 0 to zero. For `i >= 1, j >= 1`:

```text
s = (Q[i-1] == T[j-1]) ? match_score : mismatch_penalty
H[i][j] = max(
    0,
    H[i-1][j-1] + s,
    H[i-1][j]   + gap_penalty,
    H[i][j-1]   + gap_penalty
)
```

The answer for one pair is `max(H[i][j])` over the whole table.

## Why This Is a Good ORBench Task

1. **Inter-pair parallelism**: each sequence pair is independent.
2. **Wavefront parallelism**: cells on the same anti-diagonal are independent.
3. **Arithmetic intensity**: the DP update is simple and regular, with substantial total work.
4. **Batched setting**: thousands of sequence pairs can be processed together.
5. **Exact verification**: each pair outputs one integer score, which fits ORBench's validator well.

## Algorithm Source

- Smith & Waterman, *Identification of Common Molecular Subsequences*, J. Mol. Biol. (1981)
- Durbin et al., *Biological Sequence Analysis* (1998)
- Commonly used in bioinformatics pipelines, sequence database search, and variant-analysis workflows

## Current ORBench Packaging

This task is packaged as a **`compute_only`** ORBench task. The full task directory contains:

```text
tasks/smith_waterman/
├── README.md
├── task.json
├── prompt_template.yaml
├── cpu_reference.c
├── gen_data.py
├── task_io.cu
├── task_io_cpu.c
└── data/
    └── small/
```

`compute_only` is a good fit here because the whole alignment workload belongs to the timed region; there is no natural long-lived static structure that should be separated into `init` and `compute`.

## Input Format

The task uses ORBench `input.bin` with the following tensors:

| Tensor | Type | Shape | Description |
|---|---|---:|---|
| `query_seqs` | int32 | `total_query_len` | concatenated query sequences |
| `target_seqs` | int32 | `total_target_len` | concatenated target sequences |
| `query_offsets` | int32 | `N+1` | CSR-style offsets into `query_seqs` |
| `target_offsets` | int32 | `N+1` | CSR-style offsets into `target_seqs` |

Symbol encoding:

- `0 = A`
- `1 = C`
- `2 = G`
- `3 = T`

Parameters stored in `input.bin`:

| Parameter | Type | Description |
|---|---|---|
| `N` | int64 | number of sequence pairs |
| `total_query_len` | int64 | total concatenated query length |
| `total_target_len` | int64 | total concatenated target length |
| `match_score` | int64 | match score |
| `mismatch_penalty` | int64 | mismatch penalty |
| `gap_penalty` | int64 | gap penalty |

## Output Format

`expected_output.txt` contains exactly `N` lines, one integer per pair:

```text
scores[p] = optimal Smith-Waterman local-alignment score for pair p
```

The current validator expects exact numerical equality.

## Data Sizes

| Size | N | Query length | Target length |
|---|---:|---:|---:|
| `small` | 1,000 | 64–128 | 64–128 |
| `medium` | 5,000 | 128–256 | 128–256 |
| `large` | 20,000 | 256–512 | 256–512 |

## Data Generation

Generate one scale with:

```bash
python3 tasks/smith_waterman/gen_data.py small tasks/smith_waterman/data/small --with-expected
```

This will:

1. generate random DNA pairs,
2. write `input.bin` and `requests.txt`,
3. compute `expected_output.txt` with a Python reference,
4. compile/run the CPU baseline to produce `cpu_time_ms.txt`, and
5. cross-check the CPU baseline output against the Python reference.
