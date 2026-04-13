# Hybrid Sort (Rodinia-style key sorting)

## Problem background
Sorting is one of the most fundamental problems in computer science and is also a classic GPU benchmark. Rodinia includes a **Hybrid Sort** benchmark that combines a parallel bucket-sort stage with parallel merge-sorts on the resulting sublists. In ORBench, we adapt the benchmark to a clean key-sorting task: given an array of 32-bit integer keys, sort them in nondecreasing order.

This preserves the core computational kernel that makes the benchmark interesting for GPUs: a large amount of data-parallel key movement, partitioning, and reordering, followed by structured local sorting/merging.

## Algorithm source
- Rodinia Benchmark Suite: **Hybrid Sort**
- Public benchmark mirror: `yuhc/gpu-rodinia`

The Rodinia benchmark page states that Hybrid Sort first uses a **parallel bucketsort** to split the list into enough sublists and then sorts those sublists in parallel using **merge-sort**.

## Why it is suitable for GPU acceleration
Sorting exposes abundant parallelism:
- independent classification of keys into buckets
- parallel prefix/counting for bucket placement
- parallel local sorts / block-level merge phases
- high throughput memory movement where GPU bandwidth matters a lot

Compared with the CPU baseline here, a GPU implementation can keep the unsorted input on device, reuse auxiliary buffers across trials, and parallelize both partition and merge stages.

## Input format
`input.bin` contains:
- tensor `keys` (`int32`, length `N`): unsorted keys
- param `N`: number of keys

## Output format
- sorted keys in **nondecreasing order**, written one key per line to `output.txt`

## Interface
This task uses **init_compute**:
```c
extern "C" void solution_init(int N, const int* h_keys);
extern "C" void solution_compute(int* h_sorted_keys);
```

`solution_init` is called once before timing. `solution_compute` is called repeatedly and must be idempotent.
