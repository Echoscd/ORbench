# HotSpot 2D Thermal Simulation (Transient Solver)

## Background

HotSpot estimates chip temperature from a spatial power map by solving a transient thermal diffusion model on a 2D grid. Each grid cell represents the average temperature of a small chip region. The solver iteratively updates every cell using its current temperature, the local power dissipation, its neighboring temperatures, and ambient cooling.

This task is adapted from the **Rodinia HotSpot benchmark**, a classic heterogeneous-systems benchmark derived from the HotSpot thermal modeling methodology for VLSI design.

## Why it fits GPU acceleration

The computation is a regular 5-point stencil with per-cell updates repeated for many time steps. Within each iteration, almost every grid cell can be updated independently once the previous iteration's temperature grid is known. This exposes high data parallelism, regular memory access, and substantial arithmetic intensity.

The main performance bottlenecks are:
- repeated full-grid stencil sweeps over many iterations;
- boundary handling at grid edges and corners;
- efficient reuse of neighboring temperature values.

## Inputs

All arrays are row-major.

- `temp_init` (`float32`, length = `rows * cols`): initial temperature field.
- `power` (`float32`, length = `rows * cols`): per-cell power dissipation.

Scalar parameters stored in `input.bin`:
- `rows` (`int64`): number of grid rows.
- `cols` (`int64`): number of grid columns.
- `iterations` (`int64`): number of transient update steps.
- `output_stride` (`int64`): flatten the final temperature grid row-major and keep every `output_stride`-th entry for validation/output.

## Outputs

- sampled final temperatures (`float32`), written as one float per line.
- Let `final_temp` be the converged grid after `iterations` updates.
- Output entry `k` is `final_temp_flat[k * output_stride]` for all valid sampled indices.

## Source

- Rodinia benchmark suite (University of Virginia)
- Rodinia HotSpot benchmark page
- Public GitHub mirror/fork: `yuhc/gpu-rodinia`

## Files

```text
hotspot_2d/
├── README.md
├── task.json
├── prompt_template.yaml
├── gen_data.py
├── cpu_reference.c
├── task_io_cpu.c
├── task_io.cu
└── data/
    ├── small/
    ├── medium/
    └── large/
```
