# N-Body Gravitational Simulation

## Problem Background

The N-body problem computes the gravitational force on each particle in a system of N bodies due to all other particles. For each particle i, the gravitational force from particle j is:

```
F_ij = G * m_i * m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
```

where `eps` (softening parameter) prevents singularities when particles are very close. The total force on particle i is the sum over all j != i.

This direct-sum approach has O(N^2) complexity and is the fundamental building block for more advanced methods (Barnes-Hut, FMM). It appears in astrophysical simulations (galaxy formation, stellar dynamics), molecular dynamics, and particle-based fluid simulations.

## Algorithm Source

- Textbook: Hockney & Eastwood, "Computer Simulation Using Particles" (1988)
- NVIDIA CUDA Samples: "nbody" — the canonical GPU computing benchmark
- Industry: GADGET (cosmological N-body), LAMMPS (molecular dynamics)

## Why GPU Acceleration

1. **Embarrassingly parallel outer loop**: Each particle's force computation is independent — one thread per particle.
2. **Shared memory tiling**: The inner loop (sum over all other particles) can be tiled so that tiles of particle data are loaded into shared memory, reducing global memory bandwidth by a factor of the tile size.
3. **Compute-bound**: 20+ FLOPs per pair interaction with regular memory access — ideal for GPU throughput.
4. **SIMD-friendly**: All particles execute the same instruction sequence with no branching.

## Input Format

Binary file `input.bin` (ORBench v2 format):

| Tensor | Type | Size | Description |
|--------|------|------|-------------|
| `pos_x` | float32 | N | X coordinates of particles |
| `pos_y` | float32 | N | Y coordinates of particles |
| `pos_z` | float32 | N | Z coordinates of particles |
| `mass` | float32 | N | Mass of each particle |

| Parameter | Type | Description |
|-----------|------|-------------|
| `N` | int64 | Number of particles |
| `softening_x1e6` | int64 | Softening parameter * 1e6 (divide by 1e6 to get float) |

## Output Format

File `expected_output.txt`: N lines, each containing three space-separated floats — the force components (fx, fy, fz) on that particle.

```
Format: "%.6e %.6e %.6e\n" per particle
Tolerance: rtol=1e-4, atol=1e-6
```

Gravitational constant G = 1.0 (natural units).

## Data Sizes

| Size | N (particles) | Pairs computed |
|------|---------------|----------------|
| small | 4096 | ~16.7M |
| medium | 16384 | ~268M |
| large | 65536 | ~4.3B |
