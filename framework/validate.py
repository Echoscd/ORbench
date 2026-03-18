"""
validate.py - ORBench v2.1 correctness validation

Two modes:
  1. validate_output(): Pure file comparison (output.txt vs expected_output.txt).
     No execution. Used by batch_eval after benchmark_solution already wrote output.txt.
  2. validate_solution(): Legacy — runs the executable, then compares.
     Kept for standalone use / backward compat.
"""

import os
import subprocess
import numpy as np
from dataclasses import dataclass, field

from .task import load_task, get_task_dir


@dataclass
class ValidationResult:
    correct: bool = False
    results_by_size: dict = field(default_factory=dict)  # size_name -> bool
    error: str = ""


def data_exists(data_dir: str) -> bool:
    required = ["input.bin", "expected_output.txt"]
    return all(os.path.exists(os.path.join(data_dir, f)) for f in required)


def validate_output(task_id: str, data_dir: str, num_requests: int = 0) -> tuple:
    """
    Compare output.txt vs expected_output.txt. No execution.

    If num_requests is 0 or not provided, the expected line count is inferred
    from expected_output.txt (supports tasks without a fixed num_requests).

    Returns:
        (passed: bool, message: str)
    """
    task = load_task(task_id)
    output_txt = os.path.join(data_dir, "output.txt")
    expected_txt = os.path.join(data_dir, "expected_output.txt")

    if not os.path.exists(output_txt):
        return False, "output.txt not found"
    if not os.path.exists(expected_txt):
        return False, "expected_output.txt not found"

    try:
        with open(output_txt, "r") as f:
            actual = [float(line.strip()) for line in f if line.strip()]
        with open(expected_txt, "r") as f:
            expected = [float(line.strip()) for line in f if line.strip()]
    except Exception as ex:
        return False, f"Failed to read output files: {ex}"

    # Infer num_requests from expected_output if not explicitly provided
    if num_requests <= 0:
        num_requests = len(expected)

    if len(actual) != num_requests or len(expected) != num_requests:
        return False, (
            f"Line count mismatch: got {len(actual)}, expected {len(expected)} "
            f"(num_requests={num_requests})"
        )

    atol = task.atol if task.atol is not None else 0.1
    for r in range(num_requests):
        if not np.isclose(actual[r], expected[r], atol=atol, equal_nan=True):
            diff = abs(actual[r] - expected[r])
            return False, (
                f"request {r}: diff {diff:.6e} > {atol} "
                f"(got {actual[r]:.6e}, expected {expected[r]:.6e})"
            )

    return True, "PASS"


# ---------------------------------------------------------------------------
# Legacy: standalone validate (runs the executable then compares)
# ---------------------------------------------------------------------------

def run_program(exe_path: str, args: list[str] = None, timeout: int = 180,
                env: dict = None, cwd: str = None) -> tuple:
    """Run a compiled executable, return (success, stdout, stderr)."""
    cmd = [exe_path] + (args or [])
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=timeout, text=True,
            env=run_env, cwd=cwd,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Execution timed out"
    except Exception as e:
        return False, "", str(e)


def generate_test_data(task_id: str, size_name: str, output_dir: str) -> bool:
    """Run tasks/<task_id>/gen_data.py to produce v2 inputs for a given size."""
    task_dir = get_task_dir(task_id)
    gen_script = os.path.join(task_dir, "gen_data.py")
    if not os.path.exists(gen_script):
        return False
    os.makedirs(output_dir, exist_ok=True)
    try:
        result = subprocess.run(
            ["python3", gen_script, size_name, output_dir],
            capture_output=True, timeout=300, text=True,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
        return result.returncode == 0
    except Exception:
        return False


def validate_solution(
    task_id: str,
    gpu_exe_path: str,
    device_id: int = 0,
) -> ValidationResult:
    """
    Legacy: Run GPU exe with --validate, then compare output files.
    Prefer using benchmark_solution + validate_output in batch_eval instead.
    """
    task = load_task(task_id)
    task_dir = get_task_dir(task_id)
    result = ValidationResult()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)

    sizes_to_test = task.input_sizes if task.input_sizes else {"default": {}}
    restrict = os.environ.get("ORBENCH_VALIDATE_SIZES")
    if restrict:
        allow = {s.strip() for s in restrict.split(",") if s.strip()}
        sizes_to_test = {k: v for k, v in sizes_to_test.items() if k in allow}
        if not sizes_to_test:
            result.correct = False
            result.error = f"No sizes match ORBENCH_VALIDATE_SIZES={restrict}"
            return result

    for size_name, size_params in sizes_to_test.items():
        data_dir = os.path.join(task_dir, "data", size_name)
        if not data_exists(data_dir):
            result.results_by_size[size_name] = False
            result.error += (
                f"Missing data for size '{size_name}'. "
                f"Run: python3 tasks/{task_id}/gen_data.py {size_name} {data_dir} --with-expected. "
            )
            continue

        num_requests = int(size_params.get("num_requests", 0))
        if num_requests <= 0:
            result.results_by_size[size_name] = False
            result.error += f"Invalid num_requests for size '{size_name}'. "
            continue

        output_txt = os.path.join(data_dir, "output.txt")
        if os.path.exists(output_txt):
            os.remove(output_txt)

        gpu_ok, gpu_stdout, gpu_stderr = run_program(
            gpu_exe_path, args=[data_dir, "--validate"], timeout=task.timeout, env=env
        )
        if not gpu_ok:
            result.results_by_size[size_name] = False
            result.error += f"GPU solution failed on size '{size_name}': {gpu_stderr[:200]}. "
            continue

        passed, msg = validate_output(task_id, data_dir, num_requests)
        result.results_by_size[size_name] = passed
        print(f"  [{size_name}] {msg}")
        if not passed:
            result.error += f"Output mismatch on size '{size_name}': {msg}. "

    result.correct = bool(result.results_by_size) and all(result.results_by_size.values())
    return result
