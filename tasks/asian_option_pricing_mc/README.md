# Monte Carlo Arithmetic Asian Option Pricing

## Problem Background

Asian options are path-dependent financial derivatives whose payoff depends on an average underlying price over time rather than only the terminal price. This task prices **arithmetic-average Asian call/put options** using Monte Carlo simulation under a geometric Brownian motion (GBM) model.

Unlike vanilla options, the payoff depends on the full simulated path, so each Monte Carlo path must be stepped through time and aggregated. That makes this a good benchmark for batched simulation and reduction on GPUs.

## Why This Is a Good ORBench Task

1. **Simulation-heavy**: each option requires many independent path simulations.
2. **Path parallelism**: Monte Carlo paths are independent before reduction.
3. **Contract parallelism**: different contracts are completely independent.
4. **Deterministic verification**: all standard-normal shocks are provided in the input, so expected output is reproducible.

## Algorithm Source

- Boyle, *Options: A Monte Carlo Approach* (1977)
- Glasserman, *Monte Carlo Methods in Financial Engineering* (2004)
- Standard computational-finance benchmark pattern

## Input Format

Tensors stored in `input.bin`:

| Tensor | Type | Shape | Description |
|---|---|---:|---|
| `s0` | float32 | `N` | initial prices |
| `strike` | float32 | `N` | strikes |
| `rate` | float32 | `N` | risk-free rates |
| `sigma` | float32 | `N` | volatilities |
| `maturity` | float32 | `N` | maturities |
| `option_type` | int32 | `N` | 0 = call, 1 = put |
| `shocks` | float32 | `num_paths * num_steps` | pre-generated standard-normal shocks, shared by all contracts |

Parameters:

| Parameter | Type | Description |
|---|---|---|
| `N` | int64 | number of contracts |
| `num_paths` | int64 | number of Monte Carlo paths |
| `num_steps` | int64 | number of time steps per path |

## Output Format

`expected_output.txt` contains exactly `N` lines, one float per contract:

```text
Monte Carlo arithmetic-Asian option price for contract i
```

## Data Sizes

| Size | N | Paths | Steps |
|---|---:|---:|---:|
| `small` | 256 | 512 | 64 |
| `medium` | 512 | 2,048 | 64 |
| `large` | 1,024 | 4,096 | 96 |

## Data Generation

```bash
python3 tasks/asian_option_pricing_mc/gen_data.py small tasks/asian_option_pricing_mc/data/small --with-expected
```

This generates `input.bin`, `requests.txt`, `expected_output.txt`, `cpu_time_ms.txt`, and `timing.json`.
