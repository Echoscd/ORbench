#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate Network RM DP instances.

Generates a network revenue management dynamic programming instance:
  - Hub-and-spoke airline network
  - Exponential demand model
  - Price menus (discretized price vectors)
  - CPU backward induction DP to compute expected V[1][·]

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import os
import sys
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

from framework.orbench_io_py import write_input_bin


SIZES = {
    "small":  {"m": 2, "n": 4, "T": 100, "L": 10, "cap_max": 20, "seed": 42},
    "medium": {"m": 3, "n": 8, "T": 300, "L": 20, "cap_max": 30, "seed": 42},
    "large":  {"m": 4, "n": 12, "T": 500, "L": 30, "cap_max": 40, "seed": 42},
}


# ---------------------------------------------------------------------------
# Network and product generation
# ---------------------------------------------------------------------------

def generate_hub_network(m, n, rng):
    """
    Generate a hub-and-spoke network.

    m = number of flight legs (resources)
    n = number of products (itineraries)

    Each product consumes a subset of legs. A[i][j] in {0, 1}.
    """
    # Build consumption matrix A[m][n]
    # Ensure each product uses at least one leg, each leg is used by at least one product
    A = np.zeros((m, n), dtype=np.int32)

    # First m products each use exactly one leg (direct flights)
    for j in range(min(m, n)):
        A[j, j] = 1

    # Remaining products use 1-2 legs (connecting itineraries)
    for j in range(m, n):
        num_legs = rng.integers(1, min(3, m + 1))
        legs = rng.choice(m, size=num_legs, replace=False)
        for leg in legs:
            A[leg, j] = 1

    return A


def generate_demand_model(n, rng):
    """
    Generate exponential demand parameters for each product.

    Returns:
        base_price: [n] reference prices
        base_lambda: [n] demand rates at reference price
        elasticity: [n] demand elasticity (negative)
    """
    base_price = rng.uniform(100.0, 500.0, size=n).astype(np.float32)
    base_lambda = rng.uniform(0.02, 0.15, size=n).astype(np.float32)
    elasticity = rng.uniform(-3.0, -1.0, size=n).astype(np.float32)

    return base_price, base_lambda, elasticity


def generate_menus(n, L, base_price, base_lambda, elasticity, rng):
    """
    Generate L price menus. Each menu specifies prices for all n products.

    Returns:
        menu_prices: [L, n] prices
        menu_probs: [L, n+1] demand probabilities per menu
            prob[l, 0] = no purchase, prob[l, j+1] = product j purchased
        demand_cons: [L, n+1, m] - not computed here (needs A)
        demand_revenue: [L, n+1] immediate revenue for each outcome
    """
    menu_prices = np.zeros((L, n), dtype=np.float32)

    for l in range(L):
        # Sample prices around base price (±30%)
        noise = rng.uniform(0.7, 1.3, size=n)
        menu_prices[l] = base_price * noise

    # Compute demand probabilities using exponential model
    # lambda_j(p_j) = base_lambda_j * exp(elasticity_j * (p_j / base_price_j - 1))
    menu_probs = np.zeros((L, n + 1), dtype=np.float32)
    demand_revenue = np.zeros((L, n + 1), dtype=np.float32)

    for l in range(L):
        lambdas = base_lambda * np.exp(
            elasticity * (menu_prices[l] / base_price - 1.0)
        )
        # Clamp total probability to < 1
        total_lambda = np.sum(lambdas)
        if total_lambda > 0.95:
            lambdas *= 0.95 / total_lambda

        # prob[0] = no purchase
        menu_probs[l, 0] = 1.0 - np.sum(lambdas)
        menu_probs[l, 1:] = lambdas

        # Revenue: 0 for no purchase, price_j for product j
        demand_revenue[l, 0] = 0.0
        demand_revenue[l, 1:] = menu_prices[l]

    return menu_prices, menu_probs, demand_revenue


def build_demand_cons(A, L, n, m):
    """
    Build demand consumption matrix.

    demand_cons[l][k][i]:
      k=0: no purchase → all zeros
      k=j+1: product j purchased → A[:, j] (resource consumption for product j)

    Returns: [L * (n+1) * m] int32 array (flattened)
    """
    demand_cons = np.zeros((L, n + 1, m), dtype=np.int32)
    for l in range(L):
        for j in range(n):
            demand_cons[l, j + 1, :] = A[:, j]
    return demand_cons.ravel()


# ---------------------------------------------------------------------------
# Python DP solver (CPU reference for generating expected output)
# ---------------------------------------------------------------------------

def solve_dp_python(m, n, T, L, S, capacity, A,
                    demand_prob, demand_cons_flat, demand_revenue):
    """
    Backward induction DP in Python.

    Returns V1[S]: value function at t=1 for all states.
    """
    cap = capacity.copy()

    # Precompute strides for state encoding
    strides = np.ones(m, dtype=np.int64)
    for i in range(1, m):
        strides[i] = strides[i - 1] * (cap[i - 1] + 1)

    demand_cons = demand_cons_flat.reshape(L, n + 1, m)

    # Decode state s → capacity vector c[m]
    def decode_state(s):
        c = np.zeros(m, dtype=np.int32)
        rem = s
        for i in range(m):
            c[i] = rem % (cap[i] + 1)
            rem //= (cap[i] + 1)
        return c

    # Encode capacity vector c[m] → state s
    def encode_state(c):
        s = 0
        for i in range(m):
            s += int(c[i]) * int(strides[i])
        return s

    V_next = np.zeros(S, dtype=np.float64)
    V_curr = np.zeros(S, dtype=np.float64)

    for t in range(T, 0, -1):
        if t % 50 == 0:
            print(f"  DP step t={t}/{T}")

        for s in range(S):
            c = decode_state(s)
            best_value = -1e30

            for l in range(L):
                expected_value = 0.0

                for k in range(n + 1):
                    prob = float(demand_prob[l * (n + 1) + k])
                    if prob < 1e-12:
                        continue

                    cons = demand_cons[l, k]
                    rev_k = float(demand_revenue[l * (n + 1) + k])

                    # Check feasibility
                    new_c = c - cons
                    if np.any(new_c < 0):
                        # Infeasible: customer can't buy → no revenue, state unchanged
                        future = V_next[s]
                        actual_rev = 0.0
                    else:
                        new_s = encode_state(new_c)
                        future = V_next[new_s]
                        actual_rev = rev_k

                    expected_value += prob * (actual_rev + future)

                if expected_value > best_value:
                    best_value = expected_value

            V_curr[s] = best_value

        # Swap
        V_next, V_curr = V_curr, V_next

    # After the loop, V_next holds V[1][·]
    return V_next.astype(np.float32)


# ---------------------------------------------------------------------------
# CPU baseline compile/run
# ---------------------------------------------------------------------------

def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "network_rm_dp" / "solution_cpu"
    src = orbench_root / "tasks" / "network_rm_dp" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "network_rm_dp" / "task_io_cpu.c"
    harness = orbench_root / "framework" / "harness_cpu.c"

    sources = [src, task_io_cpu, harness]
    if exe.exists():
        try:
            exe_m = exe.stat().st_mtime
            if all(exe_m >= s.stat().st_mtime for s in sources):
                return exe
        except Exception:
            pass

    cmd = [
        "gcc", "-O2", "-DORBENCH_COMPUTE_ONLY",
        "-I", str(orbench_root / "framework"),
        str(harness), str(task_io_cpu), str(src),
        "-o", str(exe), "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}")
    return exe


def run_cpu_time(exe: Path, data_dir: Path) -> float:
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True,
                       timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline run failed:\n{r.stderr}\n{r.stdout}")
    m = re.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"TIME_MS not found in stdout:\n{r.stdout}")
    return float(m.group(1))


def run_cpu_expected_output(exe: Path, data_dir: Path) -> None:
    out_txt = data_dir / "output.txt"
    if out_txt.exists():
        out_txt.unlink()
    r = subprocess.run([str(exe), str(data_dir), "--validate"],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate failed:\n{r.stderr}\n{r.stdout}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")
    expected = data_dir / "expected_output.txt"
    shutil.copy2(out_txt, expected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)

    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = (len(sys.argv) == 4 and sys.argv[3] == "--with-expected")
    out_dir.mkdir(parents=True, exist_ok=True)

    if size_name not in SIZES:
        raise ValueError(f"Unknown size: {size_name}. Available: {list(SIZES.keys())}")

    cfg = SIZES[size_name]
    m = cfg["m"]
    n = cfg["n"]
    T = cfg["T"]
    L = cfg["L"]
    cap_max = cfg["cap_max"]
    seed = cfg["seed"]

    rng = np.random.default_rng(seed)

    # 1) Generate network
    A = generate_hub_network(m, n, rng)
    print(f"[gen_data] Consumption matrix A ({m}x{n}):")
    print(A)

    # 2) Generate capacities (all equal to cap_max for simplicity)
    capacity = np.full(m, cap_max, dtype=np.int32)

    # 3) Compute state space size
    S = 1
    for i in range(m):
        S *= (capacity[i] + 1)
    print(f"[gen_data] State space S = {S}")

    # 4) Generate demand model
    base_price, base_lambda, elasticity = generate_demand_model(n, rng)

    # 5) Generate price menus
    menu_prices, menu_probs, demand_revenue = generate_menus(
        n, L, base_price, base_lambda, elasticity, rng
    )

    # Verify probabilities sum to 1
    for l in range(L):
        total = np.sum(menu_probs[l])
        assert abs(total - 1.0) < 1e-5, f"Menu {l}: probs sum to {total}"

    # 6) Build demand consumption matrix
    demand_cons = build_demand_cons(A, L, n, m)

    # Flatten arrays for binary output
    demand_prob_flat = menu_probs.ravel().astype(np.float32)
    demand_revenue_flat = demand_revenue.ravel().astype(np.float32)

    print(f"[gen_data] {size_name}: m={m}, n={n}, T={T}, L={L}, S={S}")
    total_ops = int(T) * int(S) * int(L) * (int(n) + 1) * int(m)
    print(f"[gen_data] Estimated total ops: {total_ops:.2e}")

    # 7) Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("capacity", "int32", capacity),
            ("consumption", "int32", A.ravel()),
            ("demand_prob", "float32", demand_prob_flat),
            ("demand_cons", "int32", demand_cons),
            ("demand_revenue", "float32", demand_revenue_flat),
        ],
        params={
            "m": int(m),
            "n": int(n),
            "T": int(T),
            "L": int(L),
            "S": int(S),
        },
    )

    # 8) Dummy requests.txt (single-shot task)
    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        # Try C baseline first; fall back to Python if C not compiled yet
        try:
            exe = compile_cpu_baseline(_ORBENCH_ROOT)
            time_ms = run_cpu_time(exe, out_dir)
            with open(out_dir / "cpu_time_ms.txt", "w") as f:
                f.write(f"{time_ms:.3f}\n")
            run_cpu_expected_output(exe, out_dir)
            print(f"[gen_data] {size_name}: wrote all files (C baseline) in {out_dir}")
        except Exception as e:
            print(f"[gen_data] C baseline failed ({e}), using Python DP solver...")
            V1 = solve_dp_python(
                m, n, T, L, S, capacity, A,
                demand_prob_flat, demand_cons, demand_revenue_flat
            )
            # Write expected_output.txt
            with open(out_dir / "expected_output.txt", "w") as f:
                for s in range(S):
                    f.write(f"{V1[s]:.6e}\n")
            print(f"[gen_data] {size_name}: wrote all files (Python DP) in {out_dir}")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin in {out_dir}")


if __name__ == "__main__":
    main()
