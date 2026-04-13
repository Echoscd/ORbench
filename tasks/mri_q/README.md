# MRI-Q Reconstruction (Parboil-style Q Matrix)

## Background

MRI-Q computes a precomputable complex-valued **Q matrix** used in 3D magnetic resonance image reconstruction. Given a set of **K-space trajectory samples** and a set of **X-space output locations**, each output element accumulates contributions from all K-space samples via a complex exponential.

This task is adapted from the **Parboil MRI-Q benchmark** and the corresponding **SPEC ACCEL 114.mriq** benchmark description.

## Why it fits GPU acceleration

Each X-space output point is independent once the input K-space arrays are fixed. That makes the problem highly parallel:

- Parallelism: assign one thread (or one warp/block tile) per output X-space point.
- Bottleneck: repeated accumulation over all K-space samples with expensive trigonometric operations.
- Optimization opportunities: persistent device input arrays, tiling K-space data, constant/shared-memory staging, and loop unrolling.

The kernel is fundamentally compute-heavy rather than irregular-memory dominated.

## Mathematical form

For each output point `x_i = (x[i], y[i], z[i])`, compute

\[
Q_i = \sum_{k=0}^{K-1} \Phi_k \cdot e^{j 2\pi (k_x[k] x[i] + k_y[k] y[i] + k_z[k] z[i])}
\]

where `Phi_k = phi_r[k] + j * phi_i[k]` is the complex K-space sample value.

Writing the output as `Q_i = Qr[i] + j * Qi[i]`, each term contributes

- `Qr += phi_r * cos(arg) - phi_i * sin(arg)`
- `Qi += phi_r * sin(arg) + phi_i * cos(arg)`

with `arg = 2*pi*(kx*x + ky*y + kz*z)`.

## Input format

All tensors are stored in `input.bin`:

- `kx`, `ky`, `kz` — float32 arrays of length `num_k`
- `phi_r`, `phi_i` — float32 arrays of length `num_k`
- `x`, `y`, `z` — float32 arrays of length `num_x`

Scalar parameters:

- `num_k` — number of K-space samples
- `num_x` — number of output X-space points
- `sample_stride` — stride used only when writing validation output

## Output format

The computational kernel produces full-length arrays:

- `Qr[num_x]` — real part of Q
- `Qi[num_x]` — imaginary part of Q

For validation/output file size control, `task_write_output` writes only every `sample_stride`-th output element:

```text
Qr[0] Qi[0]
Qr[stride] Qi[stride]
Qr[2*stride] Qi[2*stride]
...
```

## Real source / benchmark lineage

- Parboil benchmark suite (public GitHub mirror: `yuhc/gpu-parboil`)
- SPEC ACCEL benchmark `114.mriq`

## Directory contents

```
tasks/mri_q/
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
