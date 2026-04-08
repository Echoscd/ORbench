# N-Body Gravitational Simulation

## Problem Background

The N-body problem computes the gravitational force on each particle in a system of bodies due to all other bodies. For particle `i`, the force contributed by particle `j` is:

```text
F_ij = G * m_i * m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
```

where `eps` is a softening parameter that avoids singularities when two particles are very close. The total force on particle `i` is the sum over all `j != i`.

The direct-summation method has `O(N^2)` complexity and is a classic benchmark in astrophysics, molecular dynamics, and particle simulation.

## Algorithm Source

- Hockney & Eastwood, *Computer Simulation Using Particles* (1988)
- Classical direct-sum N-body simulation used in computational physics
- NVIDIA CUDA samples and many HPC tutorials use N-body as a canonical GPU benchmark

## Why GPU Acceleration

1. **Embarrassingly parallel outer loop**: each particle's output force can be computed independently.
2. **Regular dense interaction pattern**: every particle interacts with every other particle.
3. **Shared-memory tiling opportunity**: blocks can reuse tiles of particle positions and masses.
4. **High arithmetic intensity**: each pairwise interaction performs multiple floating-point operations.

## Input Format

`input.bin` contains four float32 tensors and two integer parameters:

- `pos_x`, `pos_y`, `pos_z`: particle coordinates, length `N`
- `mass`: particle masses, length `N`
- `N`: number of particles
- `softening_x1e6`: softening parameter multiplied by `1e6`

## Output Format

`expected_output.txt` contains **3N lines**. For each particle `i`, write:

- line `3*i + 0`: `fx[i]`
- line `3*i + 1`: `fy[i]`
- line `3*i + 2`: `fz[i]`

Each line contains a single float. This flattened format matches the current ORBench validator.

## Data Sizes

| Size | N |
|------|---:|
| small | 1024 |
| medium | 4096 |
| large | 8192 |
