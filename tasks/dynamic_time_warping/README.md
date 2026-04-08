# Dynamic Time Warping Distance

## Problem Background

Dynamic Time Warping (DTW) is a classic dynamic-programming algorithm for aligning two time series that may be locally stretched or compressed along the time axis. It is widely used in speech recognition, gesture matching, sensor analytics, and general time-series similarity search.

For two sequences `Q` and `T`, DTW computes the minimum-cost alignment path under local moves `(i-1,j)`, `(i,j-1)`, and `(i-1,j-1)`. In this task, the local matching cost is the absolute difference `|Q[i-1] - T[j-1]|`, and the output is the exact total alignment cost.

## Why This Is a Good ORBench Task

1. **Independent batch structure**: every sequence pair is independent.
2. **Wavefront parallelism**: cells on the same anti-diagonal are independent once previous diagonals finish.
3. **Memory-regular DP**: the recurrence is simple and uses predictable accesses.
4. **Exact verification**: each instance outputs one integer distance, which fits the current validator well.

## Algorithm Source

- Sakoe & Chiba, *Dynamic Programming Algorithm Optimization for Spoken Word Recognition* (1978)
- Rabiner & Juang, *Fundamentals of Speech Recognition* (1993)
- Standard textbook / industrial time-series matching primitive

## Input Format

Tensors stored in `input.bin`:

| Tensor | Type | Shape | Description |
|---|---|---:|---|
| `query_series` | int32 | `total_query_len` | concatenated query time series |
| `target_series` | int32 | `total_target_len` | concatenated target time series |
| `query_offsets` | int32 | `N+1` | CSR-style offsets into `query_series` |
| `target_offsets` | int32 | `N+1` | CSR-style offsets into `target_series` |

Parameters:

| Parameter | Type | Description |
|---|---|---|
| `N` | int64 | number of sequence pairs |
| `total_query_len` | int64 | total concatenated query length |
| `total_target_len` | int64 | total concatenated target length |

## Output Format

`expected_output.txt` contains exactly `N` lines, one integer per pair:

```text
DTW distance for pair p
```

## Data Sizes

| Size | N | Query length | Target length |
|---|---:|---:|---:|
| `small` | 1,000 | 64–128 | 64–128 |
| `medium` | 5,000 | 128–256 | 128–256 |
| `large` | 20,000 | 256–512 | 256–512 |

## Data Generation

```bash
python3 tasks/dynamic_time_warping/gen_data.py small tasks/dynamic_time_warping/data/small --with-expected
```

This generates `input.bin`, `requests.txt`, `expected_output.txt`, `cpu_time_ms.txt`, and `timing.json`.
