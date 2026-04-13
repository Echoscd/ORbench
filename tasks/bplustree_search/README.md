# B+ Tree Batch Exact Search

## Background

B+ trees are a classic index structure used in databases and storage systems for ordered key lookup.  
The Rodinia **B+ Tree** benchmark studies **batched parallel tree traversal** on GPUs: many independent queries search the same tree concurrently, which exposes throughput-oriented parallelism even though each individual traversal is logarithmic and branchy.

This ORBench task adapts the **single-key search** mode of the benchmark into a compact benchmarking task: given a flattened B+ tree stored in contiguous arrays and a batch of query keys, return the matching values (or `-1` if the key is absent).

## Source

This task is adapted from the public Rodinia B+Tree benchmark and its public GitHub mirror (`yuhc/gpu-rodinia`).
The Rodinia benchmark page states that only the J/K query commands were ported to parallel languages, and the associated paper studies both single-parameter searches and range searches. This ORBench task focuses on the **single-key exact-search path**.

## Why it fits GPU acceleration

A single B+ tree traversal is only moderately parallel, but **a large batch of independent queries** can be processed concurrently:

- each query traverses the same read-only tree;
- the tree can be copied to GPU memory once and reused;
- one thread / warp / block can handle one query, or a braided strategy can parallelize node-level comparison work;
- the main bottlenecks are irregular memory access, branch divergence, and repeated global reads of tree nodes.

This makes the task a good benchmark for **throughput-oriented parallel search on irregular data structures**.

## Input format

All tensors are 1D and stored in `input.bin`.

Parameters:
- `num_nodes`: total number of nodes in the flattened tree
- `order`: B+ tree order (maximum children per internal node)
- `max_keys`: maximum keys per node (`order - 1`)
- `root_idx`: index of root node in flattened arrays
- `num_queries`: number of batched lookup keys
- `num_records`: number of leaf records indexed in the tree

Tensors:
- `is_leaf[num_nodes]` (int32): `1` for leaf nodes, `0` for internal nodes
- `key_counts[num_nodes]` (int32): valid key count for each node
- `keys[num_nodes * max_keys]` (int32): sorted separator / leaf keys for each node
- `children[num_nodes * order]` (int32): child indices for internal nodes, `-1` otherwise
- `values[num_nodes * max_keys]` (int32): payload values for leaf keys, `-1` for internal nodes
- `query_keys[num_queries]` (int32): exact-match lookup keys

### Traversal rule

For an internal node with sorted separators `k[0..m-1]`, descend to child `c`, where `c` is the first position such that `query < k[c]`; if no such position exists, use the last child.

At a leaf node, linearly scan the valid keys and return the corresponding value if found; otherwise return `-1`.

## Output format

- `out_values[num_queries]` (int32)
- `out_values[i]` is the value associated with `query_keys[i]`, or `-1` if absent
