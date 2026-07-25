"""
benchmark.py - Performance measurement using CUDA Events
Two-level timing: end-to-end (from program stdout) + nsys trace (kernel-only)
"""

import os
import re
import signal
import subprocess
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .task import load_task, get_task_dir
from .profile import run_nsys_profile, analyze_nsys_trace, analyze_all_nsys_csvs, write_nsys_full_summary
from .config import get_config


@dataclass
class TimingStats:
    mean: float = -1.0
    std: float = -1.0
    min: float = -1.0
    max: float = -1.0
    num_trials: int = 0


@dataclass
class BenchmarkResult:
    # End-to-end timing (CUDA Event, from program stdout)
    e2e_time_ms: TimingStats = field(default_factory=TimingStats)

    # Kernel-only timing (from nsys trace)
    kernel_time_ms: Optional[float] = None
    gpu_utilization: Optional[float] = None     # kernel_time / e2e_time
    num_kernel_launches: Optional[int] = None
    memcpy_overhead_ms: Optional[float] = None
    nsys_csv_path: Optional[str] = None         # path to saved CSV

    # CPU baseline
    cpu_baseline_ms: float = -1.0

    # Speedups
    speedup_e2e: float = -1.0
    speedup_kernel: Optional[float] = None

    # Which data was used (so caller can find output.txt for validation)
    data_dir: str = ""
    size_name: str = ""

    # Metadata
    hardware: str = ""
    device_id: int = 0
    error: str = ""


def parse_timing_output(stdout: str) -> Optional[float]:
    """
    Parse timing from program stdout.
    
    Expected format (programs should print this):
        TIME_MS: 0.634
    
    Also supports:
        Time: 0.634 ms
        Elapsed: 0.634ms
    """
    patterns = [
        r'TIME_MS:\s*([\d.]+)',
        r'Time:\s*([\d.]+)\s*ms',
        r'Elapsed:\s*([\d.]+)\s*ms',
        r'gpu_time_ms=([\d.]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, stdout, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None


def get_gpu_name(device_id: int = 0) -> str:
    """Get GPU device name"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
             f"--id={device_id}"],
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def run_cpu_baseline(task_id: str, data_dir: str = None) -> float:
    """
    AccelEval v2: Read CPU baseline time from cpu_time_ms.txt in data_dir.
    """
    task_dir = get_task_dir(task_id)
    # Find data_dir if not provided
    if data_dir is None:
        for size_name in ["large", "medium", "small"]:
            candidate = os.path.join(task_dir, "data", size_name)
            if os.path.exists(os.path.join(candidate, "cpu_time_ms.txt")):
                data_dir = candidate
                break

    if data_dir is None:
        return -1.0

    cpu_time_path = os.path.join(data_dir, "cpu_time_ms.txt")
    if not os.path.exists(cpu_time_path):
        return -1.0
    try:
        with open(cpu_time_path, "r") as f:
            return float(f.read().strip())
    except Exception:
        return -1.0


def _run_exe(exe_path: str, args: list[str] = None, timeout: int = None,
             device_id: int = 0) -> tuple[bool, str, str]:
    """Run executable with CUDA device selection.

    On timeout we kill the entire process group with SIGKILL so that any
    CUDA child holding GPU memory is reaped immediately. Without this the
    child gets reparented to init and continues to occupy a GPU slot for
    hours, polluting subsequent timing measurements.
    """
    # Use config default if timeout not provided
    if timeout is None:
        config = get_config()
        timeout = config.eval.timeout

    cmd = [exe_path] + (args or [])
    env = os.environ.copy()
    # Respect parent CUDA_VISIBLE_DEVICES if the caller pinned a GPU at the
    # process level (used by dual-worker pipelines that run two evals in
    # parallel, one per physical GPU). Otherwise fall back to device_id.
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)

    proc = None
    try:
        # start_new_session=True puts the child in its own process group
        # so killpg can reach all descendants on timeout.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=env, start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return proc.returncode == 0, stdout, stderr
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            return False, stdout, "Timed out (process group SIGKILL'd)"
    except Exception as e:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        return False, "", str(e)


def benchmark_solution(
    task_id: str,
    exe_path: str,
    device_id: int = 0,
    run_nsys: bool = True,
    save_nsys_csv: bool = False,
    save_nsys_csv_dir: str = None,
) -> BenchmarkResult:
    """
    Benchmark a compiled GPU solution.
    
    Level A (always): Parse CUDA Event timing from program stdout.
    Level B (optional): Run nsys for kernel-level timing breakdown.
    
    Uses the largest available data size for benchmarking.
    """
    task = load_task(task_id)
    task_dir = get_task_dir(task_id)
    result = BenchmarkResult()
    result.device_id = device_id
    result.hardware = get_gpu_name(device_id)

    # Pick a data size for benchmarking.
    #
    # Preference order:
    #   1) ACCELEVAL_BENCHMARK_SIZES env (comma-separated list, e.g. "small,medium")
    #   2) ACCELEVAL_VALIDATE_SIZES env (to stay consistent with validation)
    #   3) Default: ["large", "medium", "small"]
    #
    # For each candidate size, require:
    #   - input.bin exists (v2 data format)
    #   - cpu_time_ms.txt exists (CPU baseline)
    import os as _os

    sizes_env = _os.environ.get("ACCELEVAL_BENCHMARK_SIZES") or _os.environ.get("ACCELEVAL_VALIDATE_SIZES")
    if sizes_env:
        size_candidates = [s.strip() for s in sizes_env.split(",") if s.strip()]
    else:
        size_candidates = ["large", "medium", "small"]

    data_dir = None
    chosen_size = None
    for size_name in size_candidates:
        candidate = _os.path.join(task_dir, "data", size_name)
        if _os.path.exists(_os.path.join(candidate, "input.bin")) and _os.path.exists(
            _os.path.join(candidate, "cpu_time_ms.txt")
        ):
            data_dir = candidate
            chosen_size = size_name
            break

    if data_dir is None:
        result.error = "No pre-generated data (input.bin + cpu_time_ms.txt) found for benchmarking"
        return result

    result.data_dir = data_dir
    result.size_name = chosen_size

    # Remove stale artifacts so a failed run can never inherit a previous
    # solution's output.txt / timing.json (audit hardening).
    for _stale in ("output.txt", "timing.json"):
        _p = _os.path.join(data_dir, _stale)
        if _os.path.exists(_p):
            try:
                _os.remove(_p)
            except OSError:
                pass

    # Timing run: warmup + timed trials only. Validation is deliberately NOT
    # done in this process — see the fresh-process validation run below (A2).
    config = get_config()
    exe_args = [
        data_dir,
        "--warmup", str(config.eval.warmup),
        "--trials", str(config.eval.num_trials),
    ]
    
    # Use config timeout if task timeout is not set or use config default
    timeout_to_use = task.timeout if task.timeout > 0 else config.eval.timeout

    # === Level A: timing from harness ===
    ok, stdout, stderr = _run_exe(exe_path, args=exe_args, device_id=device_id, timeout=timeout_to_use)
    if not ok:
        result.error = f"Execution failed: {stderr[:200]}"
        return result

    # Prefer timing.json (detailed: mean/min/max/num_trials) written by harness
    timing_json_path = _os.path.join(data_dir, "timing.json")
    if _os.path.exists(timing_json_path):
        try:
            with open(timing_json_path) as f:
                timing = json.load(f)
            result.e2e_time_ms = TimingStats(
                mean=timing["mean_ms"],
                std=0.0,
                min=timing["min_ms"],
                max=timing["max_ms"],
                num_trials=timing["num_trials"],
            )
        except Exception:
            # Fallback to stdout parsing
            t = parse_timing_output(stdout)
            if t is not None:
                result.e2e_time_ms = TimingStats(mean=t, std=0.0, min=t, max=t, num_trials=1)
    else:
        # Fallback: parse stdout (backward compat)
        t = parse_timing_output(stdout)
        if t is not None:
            result.e2e_time_ms = TimingStats(mean=t, std=0.0, min=t, max=t, num_trials=1)

    # === A2: fresh-process validation (audit hardening) ===
    # output.txt is produced by a SINGLE cold call in a NEW process. A solution
    # whose output is only correct because of state accumulated across the warm
    # calls (cross-call convergence, deferred-correctness caching) fails here,
    # while every self-contained solution is unaffected.
    ok_v, _stdout_v, stderr_v = _run_exe(
        exe_path,
        args=[data_dir, "--validate", "--warmup", "0", "--trials", "0"],
        device_id=device_id, timeout=timeout_to_use,
    )
    if not ok_v:
        result.error += f"Fresh-process validation failed: {stderr_v[:200]}. "

    # === CPU baseline ===
    result.cpu_baseline_ms = run_cpu_baseline(task_id, data_dir=data_dir)

    if result.cpu_baseline_ms > 0 and result.e2e_time_ms.mean > 0:
        result.speedup_e2e = result.cpu_baseline_ms / result.e2e_time_ms.mean

    # === Level B: nsys trace analysis (optional) ===
    if run_nsys:
        try:
            # Save nsys output next to the executable
            nsys_output_dir = os.path.dirname(exe_path)
            nsys_timeout = config.profiling.nsys_timeout
            nsys_result = run_nsys_profile(
                exe_path, exe_args=exe_args, device_id=device_id,
                timeout=nsys_timeout, output_dir=nsys_output_dir
            )

            if nsys_result is not None:
                # Unpack: (primary_csv_path, {report_name: csv_path, ...})
                if isinstance(nsys_result, tuple):
                    primary_csv, exported_csvs = nsys_result
                else:
                    # Backward compat: old version returned just a string
                    primary_csv = nsys_result
                    exported_csvs = {"cuda_gpu_trace": primary_csv} if primary_csv else {}

                # Basic analysis from gpu_trace
                if primary_csv and isinstance(primary_csv, str):
                    nsys_data = analyze_nsys_trace(primary_csv)
                    result.kernel_time_ms = nsys_data.get("total_kernel_time_ms")
                    result.num_kernel_launches = nsys_data.get("num_kernel_launches")
                    result.memcpy_overhead_ms = nsys_data.get("total_memcpy_time_ms")

                    if result.kernel_time_ms and result.e2e_time_ms.mean > 0:
                        result.gpu_utilization = result.kernel_time_ms / result.e2e_time_ms.mean

                    if result.cpu_baseline_ms > 0 and result.kernel_time_ms:
                        result.speedup_kernel = result.cpu_baseline_ms / result.kernel_time_ms

                # Full analysis from all CSVs
                if save_nsys_csv and save_nsys_csv_dir and exported_csvs:
                    import shutil
                    os.makedirs(save_nsys_csv_dir, exist_ok=True)

                    # Copy all CSV files to run directory
                    for report_name, csv_path in exported_csvs.items():
                        dst = os.path.join(save_nsys_csv_dir, f"nsys_{report_name}.csv")
                        shutil.copy2(csv_path, dst)
                    print(f"  [nsys] {len(exported_csvs)} CSV files saved to {save_nsys_csv_dir}")

                    # Generate comprehensive summary
                    full_analysis = analyze_all_nsys_csvs(exported_csvs)
                    summary_path = os.path.join(save_nsys_csv_dir, "nsys_summary.txt")
                    write_nsys_full_summary(full_analysis, summary_path)
                    print(f"  [nsys] Full summary saved to {summary_path}")

        except Exception as e:
            # nsys is optional; don't fail the benchmark
            result.error += f"nsys profiling failed: {e}. "

    return result