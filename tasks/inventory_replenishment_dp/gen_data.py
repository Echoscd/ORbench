#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate Inventory Replenishment DP instances.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import sys
import re
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))
from framework.orbench_io_py import write_input_bin

def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "inventory_replenishment_dp" / "solution_cpu"
    src = orbench_root / "tasks" / "inventory_replenishment_dp" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "inventory_replenishment_dp" / "task_io_cpu.c"
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


def run_cpu_time(exe: Path, data_dir: Path, timeout: int = 1200) -> float:
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline run failed:\n{r.stderr}\n{r.stdout}")
    m = re.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"TIME_MS not found in stdout:\n{r.stdout}")
    return float(m.group(1))


def run_cpu_expected_output(exe: Path, data_dir: Path, timeout: int = 1200) -> None:
    out_txt = data_dir / "output.txt"
    if out_txt.exists():
        out_txt.unlink()
    r = subprocess.run([str(exe), str(data_dir), "--validate"],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate failed:\n{r.stderr}\n{r.stdout}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")
    expected = data_dir / "expected_output.txt"
    shutil.copy2(out_txt, expected)


try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

SIZES = {
    "small": {"N_I": 10, "N_B": 10, "N_Phi": 10, "N_Psi": 10, "N_x": 10, "N_phi": 10, "T": 5, "seed": 42},
    "medium": {"N_I": 20, "N_B": 20, "N_Phi": 20, "N_Psi": 20, "N_x": 16, "N_phi": 16, "T": 10, "seed": 42},
    "large": {"N_I": 32, "N_B": 32, "N_Phi": 32, "N_Psi": 32, "N_x": 32, "N_phi": 32, "T": 15, "seed": 42}
}

if HAS_NUMBA:
    @njit
    def get_interp_weights(val, grid):
        N = len(grid)
        if val <= grid[0]:
            return 0, 0.0
        elif val >= grid[N-1]:
            return N - 2, 1.0
        else:
            low = 0
            high = N - 1
            while low <= high:
                mid = low + (high - low) // 2
                if grid[mid] <= val:
                    low = mid + 1
                else:
                    high = mid - 1
            idx = high
            if idx < 0: idx = 0
            if idx >= N - 1: idx = N - 2
            w = (val - grid[idx]) / (grid[idx+1] - grid[idx])
            return idx, w

    @njit(parallel=True)
    def compute_dp(N_I, N_B, N_Phi, N_Psi, N_x, N_phi, T,
                   c_t, h_t, b_t, mu, nu, alpha, y,
                   grid_I, grid_B, grid_Phi, grid_Psi,
                   actions, shocks, shock_probs):

        V_next = np.zeros((N_I, N_B, N_Phi, N_Psi), dtype=np.float32)
        V_curr = np.zeros((N_I, N_B, N_Phi, N_Psi), dtype=np.float32)

        for t in range(T - 1, -1, -1):
            for iI in prange(N_I):
                for iB in range(N_B):
                    for iPhi in range(N_Phi):
                        for iPsi in range(N_Psi):
                            min_cost = np.inf

                            for ix in range(N_x):
                                x = actions[ix]
                                expected_cost = 0.0

                                for iphi in range(N_phi):
                                    phi = shocks[iphi]
                                    prob = shock_probs[iphi]

                                    xi = mu + nu * (phi + alpha * grid_Phi[iPhi])
                                    I_next = grid_I[iI] + y + x - xi
                                    B_next = grid_B[iB] + max(0.0, -I_next)
                                    Phi_next = grid_Phi[iPhi] + phi
                                    Psi_next = grid_Psi[iPsi] + phi * phi

                                    cost = c_t[t] * abs(x) + h_t[t] * max(0.0, I_next) + b_t[t] * max(0.0, -I_next)

                                    idxI, wI = get_interp_weights(I_next, grid_I)
                                    idxB, wB = get_interp_weights(B_next, grid_B)
                                    idxPhi, wPhi = get_interp_weights(Phi_next, grid_Phi)
                                    idxPsi, wPsi = get_interp_weights(Psi_next, grid_Psi)

                                    future_cost = 0.0
                                    for dI in range(2):
                                        cI = wI if dI else (1.0 - wI)
                                        for dB in range(2):
                                            cB = wB if dB else (1.0 - wB)
                                            for dPhi in range(2):
                                                cPhi = wPhi if dPhi else (1.0 - wPhi)
                                                for dPsi in range(2):
                                                    cPsi = wPsi if dPsi else (1.0 - wPsi)

                                                    future_cost += cI * cB * cPhi * cPsi * V_next[idxI+dI, idxB+dB, idxPhi+dPhi, idxPsi+dPsi]

                                    expected_cost += prob * (cost + future_cost)

                                if expected_cost < min_cost:
                                    min_cost = expected_cost

                            V_curr[iI, iB, iPhi, iPsi] = min_cost

            V_next[:] = V_curr[:]

        return V_curr

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
    np.random.seed(cfg["seed"])

    N_I = cfg["N_I"]
    N_B = cfg["N_B"]
    N_Phi = cfg["N_Phi"]
    N_Psi = cfg["N_Psi"]
    N_x = cfg["N_x"]
    N_phi = cfg["N_phi"]
    T = cfg["T"]

    c_t = np.random.uniform(0.05, 0.15, T).astype(np.float32)
    h_t = np.random.uniform(0.02, 0.06, T).astype(np.float32)
    b_t = np.random.uniform(0.1, 0.3, T).astype(np.float32)
    b_t[-1] = 2.0

    mu = np.float32(200.0)
    nu = np.float32(200.0 / np.sqrt(T))
    alpha = np.float32(0.25)
    y = np.float32(200.0)

    grid_I = np.linspace(-100, 500, N_I).astype(np.float32)
    grid_B = np.linspace(0, 1000, N_B).astype(np.float32)
    grid_Phi = np.linspace(-5, 5, N_Phi).astype(np.float32)
    grid_Psi = np.linspace(0, 25, N_Psi).astype(np.float32)

    actions = np.linspace(-200, 200, N_x).astype(np.float32)
    shocks = np.linspace(-1, 1, N_phi).astype(np.float32)

    shock_probs = np.random.uniform(0.5, 1.5, N_phi).astype(np.float32)
    shock_probs /= shock_probs.sum()

    print(f"[gen_data] {size_name}: N_I={N_I}, N_B={N_B}, N_Phi={N_Phi}, N_Psi={N_Psi}, N_x={N_x}, N_phi={N_phi}, T={T}")

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("c_t", "float32", c_t),
            ("h_t", "float32", h_t),
            ("b_t", "float32", b_t),
            ("grid_I", "float32", grid_I),
            ("grid_B", "float32", grid_B),
            ("grid_Phi", "float32", grid_Phi),
            ("grid_Psi", "float32", grid_Psi),
            ("actions", "float32", actions),
            ("shocks", "float32", shocks),
            ("shock_probs", "float32", shock_probs),
            ("fparams", "float32", np.array([mu, nu, alpha, y], dtype=np.float32)),
        ],
        params={
            "N_I": int(N_I),
            "N_B": int(N_B),
            "N_Phi": int(N_Phi),
            "N_Psi": int(N_Psi),
            "N_x": int(N_x),
            "N_phi": int(N_phi),
            "T": int(T),
        },
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write("compute\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")
        run_cpu_expected_output(exe, out_dir)
        print(f"[gen_data] {size_name}: CPU time={time_ms:.1f}ms, wrote all files in {out_dir}")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin in {out_dir}")

if __name__ == "__main__":
    main()
