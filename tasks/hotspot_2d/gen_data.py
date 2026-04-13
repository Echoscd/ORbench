#!/usr/bin/env python3
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "framework"))
from orbench_io_py import write_input_bin  # noqa: E402

MAX_PD = 3.0e6
PRECISION = 0.001
SPEC_HEAT_SI = 1.75e6
K_SI = 100.0
FACTOR_CHIP = 0.5
T_CHIP = 0.0005
CHIP_HEIGHT = 0.016
CHIP_WIDTH = 0.016
AMB_TEMP = 80.0

SIZE_CONFIGS = {
    "small":  {"rows": 256,  "cols": 256,  "iterations": 60,  "seed": 7, "output_stride": 97},
    "medium": {"rows": 512,  "cols": 512,  "iterations": 120, "seed": 7, "output_stride": 97},
    "large":  {"rows": 1024, "cols": 1024, "iterations": 240, "seed": 7, "output_stride": 97},
}


def generate_inputs(rows: int, cols: int, seed: int):
    rng = np.random.default_rng(seed)
    temp = np.empty((rows, cols), dtype=np.float32)
    power = np.empty((rows, cols), dtype=np.float32)
    for i in range(rows):
        x = int(rng.integers(0, 512))
        y = float(int(rng.integers(0, 128))) * 1e-6
        temp[i, :] = x + rng.integers(0, 128, size=cols).astype(np.float32) * 1e-3
        power[i, :] = y
    return temp, power


def hotspot_reference(temp_init: np.ndarray, power: np.ndarray, iterations: int) -> np.ndarray:
    rows, cols = temp_init.shape
    grid_height = CHIP_HEIGHT / rows
    grid_width = CHIP_WIDTH / cols
    cap = FACTOR_CHIP * SPEC_HEAT_SI * T_CHIP * grid_width * grid_height
    rx = grid_width / (2.0 * K_SI * T_CHIP * grid_height)
    ry = grid_height / (2.0 * K_SI * T_CHIP * grid_width)
    rz = T_CHIP / (K_SI * grid_height * grid_width)
    max_slope = MAX_PD / (FACTOR_CHIP * T_CHIP * SPEC_HEAT_SI)
    step = PRECISION / max_slope

    cur = temp_init.astype(np.float32, copy=True)
    nxt = np.zeros_like(cur)

    for _ in range(iterations):
        center = cur[1:-1, 1:-1]
        nxt[1:-1, 1:-1] = center + (step / cap) * (
            power[1:-1, 1:-1]
            + (cur[2:, 1:-1] + cur[:-2, 1:-1] - 2.0 * center) / ry
            + (cur[1:-1, 2:] + cur[1:-1, :-2] - 2.0 * center) / rx
            + (AMB_TEMP - center) / rz
        )

        nxt[0, 0] = cur[0, 0] + (step / cap) * (
            power[0, 0] + (cur[0, 1] - cur[0, 0]) / rx + (cur[1, 0] - cur[0, 0]) / ry + (AMB_TEMP - cur[0, 0]) / rz
        )
        nxt[0, cols - 1] = cur[0, cols - 1] + (step / cap) * (
            power[0, cols - 1] + (cur[0, cols - 2] - cur[0, cols - 1]) / rx + (cur[1, cols - 1] - cur[0, cols - 1]) / ry + (AMB_TEMP - cur[0, cols - 1]) / rz
        )
        nxt[rows - 1, cols - 1] = cur[rows - 1, cols - 1] + (step / cap) * (
            power[rows - 1, cols - 1]
            + (cur[rows - 1, cols - 2] - cur[rows - 1, cols - 1]) / rx
            + (cur[rows - 2, cols - 1] - cur[rows - 1, cols - 1]) / ry
            + (AMB_TEMP - cur[rows - 1, cols - 1]) / rz
        )
        nxt[rows - 1, 0] = cur[rows - 1, 0] + (step / cap) * (
            power[rows - 1, 0] + (cur[rows - 1, 1] - cur[rows - 1, 0]) / rx + (cur[rows - 2, 0] - cur[rows - 1, 0]) / ry + (AMB_TEMP - cur[rows - 1, 0]) / rz
        )

        nxt[0, 1:-1] = cur[0, 1:-1] + (step / cap) * (
            power[0, 1:-1]
            + (cur[0, 2:] + cur[0, :-2] - 2.0 * cur[0, 1:-1]) / rx
            + (cur[1, 1:-1] - cur[0, 1:-1]) / ry
            + (AMB_TEMP - cur[0, 1:-1]) / rz
        )
        nxt[rows - 1, 1:-1] = cur[rows - 1, 1:-1] + (step / cap) * (
            power[rows - 1, 1:-1]
            + (cur[rows - 1, 2:] + cur[rows - 1, :-2] - 2.0 * cur[rows - 1, 1:-1]) / rx
            + (cur[rows - 2, 1:-1] - cur[rows - 1, 1:-1]) / ry
            + (AMB_TEMP - cur[rows - 1, 1:-1]) / rz
        )
        nxt[1:-1, cols - 1] = cur[1:-1, cols - 1] + (step / cap) * (
            power[1:-1, cols - 1]
            + (cur[2:, cols - 1] + cur[:-2, cols - 1] - 2.0 * cur[1:-1, cols - 1]) / ry
            + (cur[1:-1, cols - 2] - cur[1:-1, cols - 1]) / rx
            + (AMB_TEMP - cur[1:-1, cols - 1]) / rz
        )
        nxt[1:-1, 0] = cur[1:-1, 0] + (step / cap) * (
            power[1:-1, 0]
            + (cur[2:, 0] + cur[:-2, 0] - 2.0 * cur[1:-1, 0]) / ry
            + (cur[1:-1, 1] - cur[1:-1, 0]) / rx
            + (AMB_TEMP - cur[1:-1, 0]) / rz
        )

        cur, nxt = nxt, cur

    return cur


def write_expected(out_dir: Path, final_grid: np.ndarray, output_stride: int):
    flat = final_grid.reshape(-1)
    sampled = flat[::output_stride]
    with open(out_dir / "expected_output.txt", "w", encoding="utf-8") as f:
        for x in sampled:
            f.write(f"{float(x):.6f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("size", choices=SIZE_CONFIGS.keys())
    parser.add_argument("output_dir")
    parser.add_argument("--with-expected", action="store_true")
    args = parser.parse_args()

    cfg = SIZE_CONFIGS[args.size]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp, power = generate_inputs(cfg["rows"], cfg["cols"], cfg["seed"])
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("temp_init", "float32", temp.reshape(-1)),
            ("power", "float32", power.reshape(-1)),
        ],
        params={
            "rows": cfg["rows"],
            "cols": cfg["cols"],
            "iterations": cfg["iterations"],
            "output_stride": cfg["output_stride"],
        },
    )

    if args.with_expected:
        final_grid = hotspot_reference(temp, power, cfg["iterations"])
        write_expected(out_dir, final_grid, cfg["output_stride"])

    total = cfg["rows"] * cfg["cols"]
    num_samples = math.ceil(total / cfg["output_stride"])
    print(f"Generated {args.size}: rows={cfg['rows']} cols={cfg['cols']} iterations={cfg['iterations']} samples={num_samples}")


if __name__ == "__main__":
    main()
