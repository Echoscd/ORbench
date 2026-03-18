#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate input.bin + expected_output.txt for crew_pairing

Reads airline crew-pairing CSV files (from heurigym dataset) and converts them
to the ORBench binary format.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import os
import sys
import re
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

from framework.orbench_io_py import write_input_bin

# ---------------------------------------------------------------------------
# Size definitions: each maps to a CSV file from the heurigym dataset
# ---------------------------------------------------------------------------
SIZES = {
    "small":  {"instance": "DataA_chunk3_id1", "subdir": "demo"},      # ~37 legs
    "medium": {"instance": "DataA_chunk15_id1", "subdir": "eval"},     # ~206 legs
    "large":  {"instance": "DataB_chunk5_id1",  "subdir": "eval"},     # ~2239 legs
}

# Where the source CSV files live
_DATASET_BASE = _ORBENCH_ROOT / "tasks" / "heurigym" / "_datasets" / "crew_pairing"

# Crew-pairing constants (stored as params so C code can use them)
MAX_DUTY_MINUTES = 14 * 60       # 840
MAX_BLOCK_MINUTES = 10 * 60      # 600
MAX_LEGS_PER_DUTY = 6
MIN_REST_MINUTES = 9 * 60        # 540
POSITIONING_FEE_X100 = 1000000   # $10,000 * 100

TIME_FMT = "%m/%d/%Y %H:%M"


def parse_csv(csv_path: Path):
    """Parse a crew-pairing CSV file into structured numpy arrays.

    Returns:
        dep_minutes, arr_minutes, dep_stations, arr_stations: int32 arrays
        duty_cost_per_hour, pairing_cost_per_hour: float values
        station_names: list of station name strings (index = station ID)
        leg_tokens: list of token strings for each leg
    """
    import csv

    legs = []
    duty_rate = None
    pairing_rate = None

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dep_dt = datetime.strptime(
                f"{row['DptrDate'].strip()} {row['DptrTime'].strip()}", TIME_FMT
            )
            arr_dt = datetime.strptime(
                f"{row['ArrvDate'].strip()} {row['ArrvTime'].strip()}", TIME_FMT
            )
            flt = row["FltNum"].strip()
            dep_stn = row["DptrStn"].strip()
            arr_stn = row["ArrvStn"].strip()
            token = f"{flt}_{dep_dt.strftime('%Y-%m-%d')}"

            # Forward-fill cost rates
            dr = row.get("DutyCostPerHour", "").strip()
            pr = row.get("ParingCostPerHour", "").strip()
            if dr:
                duty_rate = float(dr)
            if pr:
                pairing_rate = float(pr)

            legs.append({
                "token": token,
                "dep_dt": dep_dt,
                "arr_dt": arr_dt,
                "dep_stn": dep_stn,
                "arr_stn": arr_stn,
            })

    if duty_rate is None or pairing_rate is None:
        raise ValueError("Missing cost rates in CSV")

    # Sort by departure time
    legs.sort(key=lambda x: x["dep_dt"])

    # Build station mapping
    all_stations = sorted(set(l["dep_stn"] for l in legs) | set(l["arr_stn"] for l in legs))
    stn_to_id = {s: i for i, s in enumerate(all_stations)}

    # Compute epoch (earliest departure)
    epoch = legs[0]["dep_dt"]

    N = len(legs)
    dep_minutes = np.zeros(N, dtype=np.int32)
    arr_minutes = np.zeros(N, dtype=np.int32)
    dep_stations = np.zeros(N, dtype=np.int32)
    arr_stations = np.zeros(N, dtype=np.int32)
    tokens = []

    for i, leg in enumerate(legs):
        dep_minutes[i] = int((leg["dep_dt"] - epoch).total_seconds() / 60)
        arr_minutes[i] = int((leg["arr_dt"] - epoch).total_seconds() / 60)
        dep_stations[i] = stn_to_id[leg["dep_stn"]]
        arr_stations[i] = stn_to_id[leg["arr_stn"]]
        tokens.append(leg["token"])

    base_id = stn_to_id.get("NKX", 0)

    return (dep_minutes, arr_minutes, dep_stations, arr_stations,
            duty_rate, pairing_rate, all_stations, base_id, tokens)


def compile_cpu_baseline(orbench_root: Path) -> Path:
    """Compile the crew_pairing CPU reference."""
    exe = orbench_root / "tasks" / "crew_pairing" / "solution_cpu"
    src = orbench_root / "tasks" / "crew_pairing" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "crew_pairing" / "task_io_cpu.c"
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
        "gcc", "-O2",
        "-I", str(orbench_root / "framework"),
        str(harness),
        str(task_io_cpu),
        str(src),
        "-o", str(exe),
        "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}")
    return exe


def run_cpu_time(exe: Path, data_dir: Path) -> float:
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True)
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
    r = subprocess.run([str(exe), str(data_dir), "--validate"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate run failed:\n{r.stderr}\n{r.stdout}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")
    expected = data_dir / "expected_output.txt"
    shutil.copy2(out_txt, expected)


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

    instance_name = SIZES[size_name]["instance"]
    subdir = SIZES[size_name]["subdir"]
    csv_path = _DATASET_BASE / subdir / f"{instance_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"[gen_data] Parsing {csv_path} ...")
    (dep_minutes, arr_minutes, dep_stations, arr_stations,
     duty_rate, pairing_rate, station_names, base_id, tokens) = parse_csv(csv_path)

    N = len(dep_minutes)
    num_stations = len(station_names)

    # Store cost rates as x100 integers (680.0 -> 68000, 20.0 -> 2000)
    duty_rate_x100 = int(round(duty_rate * 100))
    pairing_rate_x100 = int(round(pairing_rate * 100))

    print(f"[gen_data] N={N} legs, {num_stations} stations, base={station_names[base_id]} (id={base_id})")
    print(f"[gen_data] duty_rate={duty_rate}, pairing_rate={pairing_rate}")

    # 1) Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("dep_minutes", "int32", dep_minutes),
            ("arr_minutes", "int32", arr_minutes),
            ("dep_stations", "int32", dep_stations),
            ("arr_stations", "int32", arr_stations),
        ],
        params={
            "N": N,
            "num_stations": num_stations,
            "base_station": base_id,
            "duty_rate_x100": duty_rate_x100,
            "pairing_rate_x100": pairing_rate_x100,
            "max_duty_min": MAX_DUTY_MINUTES,
            "max_block_min": MAX_BLOCK_MINUTES,
            "max_legs_duty": MAX_LEGS_PER_DUTY,
            "min_rest_min": MIN_REST_MINUTES,
            "pos_fee_x100": POSITIONING_FEE_X100,
        },
    )

    # 2) Write tokens.txt (for task_io to reconstruct leg tokens in output)
    with open(out_dir / "tokens.txt", "w") as f:
        for tok in tokens:
            f.write(tok + "\n")

    # 3) Write stations.txt (for reference)
    with open(out_dir / "stations.txt", "w") as f:
        for i, name in enumerate(station_names):
            f.write(f"{i} {name}\n")

    # 4) Copy source CSV for reference / evaluation
    shutil.copy2(csv_path, out_dir / "instance.csv")

    # 5) Write a dummy requests.txt (framework expects it but crew_pairing has no requests)
    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")
        run_cpu_expected_output(exe, out_dir)
        print(f"[gen_data] {size_name}: wrote all files in {out_dir}")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin/tokens.txt/stations.txt in {out_dir} "
              "(expected/cpu_time skipped; pass --with-expected)")


if __name__ == "__main__":
    main()
