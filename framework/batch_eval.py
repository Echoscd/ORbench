"""
batch_eval.py - Batch evaluation across tasks, samples, and GPUs
Orchestrates the full pipeline: compile → validate → benchmark → save results
"""

import os
import sys
import json
import time
import argparse
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Optional
import re

from .task import load_task, load_all_tasks, ORBENCH_ROOT
from .compile import compile_solution
from .validate import validate_output, data_exists, ValidationResult
from .benchmark import benchmark_solution, BenchmarkResult
from .config import get_config


@dataclass
class EvalResult:
    task_id: str
    sample_id: int
    kernel_count: int = 0
    compiled: bool = False
    compile_error: str = ""
    correct: bool = False
    correctness_detail: dict = None
    benchmark: dict = None
    error: str = ""

    def to_dict(self):
        d = asdict(self)
        return d


def count_global_kernels_in_source(source_path: str) -> int:
    """
    Count the number of CUDA __global__ and __device__ kernel/function *definitions* in a .cu file.
    This is a static heuristic intended for reporting "LLM-written kernel count"
    in eval JSON; it does not attempt full C++ parsing.
    """
    try:
        with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
            s = f.read()
    except Exception:
        return 0

    # Strip comments to avoid counting "__global__"/"__device__" inside comments/strings.
    # First handle block comments (/* ... */)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    # Then handle line comments (// ...)
    s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)

    # Count both __global__ and __device__ function definitions.
    # Pattern: (__global__|__device__) followed by function signature ending with ) {
    # We use a more robust approach: find all __global__/__device__ positions,
    # then check if there's a function body (closing paren followed by opening brace)
    # within a reasonable distance (function signature length).
    count = 0
    
    # Find all positions of __global__ or __device__
    for match in re.finditer(r"\b(__global__|__device__)\b", s):
        start_pos = match.start()
        # Look ahead from this position to find the function body opening brace
        # Allow up to 1000 characters for the function signature (should be enough)
        search_end = min(start_pos + 1000, len(s))
        segment = s[start_pos:search_end]
        
        # Check if we can find a closing paren followed by an opening brace
        # This indicates a function definition (not just a declaration)
        # We look for ) followed by optional whitespace and {
        if re.search(r"\)\s*\{", segment):
            count += 1
    
    return count


def eval_single_sample(
    task_id: str,
    sample_path: str,
    sample_id: int,
    device_id: int = 0,
    arch: str = None,
    run_nsys: bool = None,
    save_nsys_csv: bool = False,
) -> EvalResult:
    """
    Evaluate a single solution sample: compile → benchmark+validate (single execution).

    benchmark_solution always runs with --validate, producing both timing.json
    and output.txt in one shot. Then validate_output does a pure file comparison.
    This halves the number of solution_compute calls (14 instead of 27).
    """
    # Use config defaults if not provided
    config = get_config()
    if arch is None:
        arch = config.gpu.arch
    if run_nsys is None:
        run_nsys = config.profiling.nsys_enabled
    
    result = EvalResult(task_id=task_id, sample_id=sample_id)
    result.kernel_count = count_global_kernels_in_source(sample_path)

    # Step 1: Compile
    compile_result = compile_solution(task_id, sample_path, arch=arch)
    result.compiled = compile_result.success

    if not compile_result.success:
        result.compile_error = compile_result.stderr[:500]
        return result

    exe_path = compile_result.executable_path

    # Step 2: Benchmark with --validate (single execution → timing + output.txt)
    try:
        nsys_csv_dir = None
        if save_nsys_csv:
            nsys_csv_dir = os.path.join(os.path.dirname(sample_path), f"nsys_sample_{sample_id}")

        bench_result = benchmark_solution(
            task_id, exe_path, device_id=device_id,
            run_nsys=run_nsys,
            save_nsys_csv=save_nsys_csv,
            save_nsys_csv_dir=nsys_csv_dir,
        )
        result.benchmark = asdict(bench_result)

        if bench_result.error:
            result.error += bench_result.error
    except Exception as e:
        result.error += f"Benchmark error: {str(e)[:200]}. "
        return result

    # Step 3: Validate correctness (pure file comparison, no extra execution)
    bench_data_dir = bench_result.data_dir
    bench_size = bench_result.size_name
    if bench_data_dir and bench_size:
        try:
            if data_exists(bench_data_dir):
                task = load_task(task_id)
                size_params = task.input_sizes.get(bench_size, {})
                num_requests = int(size_params.get("num_requests", 0))

                passed, msg = validate_output(task_id, bench_data_dir, num_requests)
                print(f"  [{bench_size}] {msg}")
                result.correct = passed
                result.correctness_detail = {bench_size: passed}
                if not passed:
                    result.error += f"Output mismatch on '{bench_size}': {msg}. "
            else:
                result.correctness_detail = {bench_size: False}
                result.error += f"Cannot validate: missing expected_output.txt in {bench_data_dir}. "
        except Exception as e:
            result.error += f"Validation error: {str(e)[:200]}. "

    return result


def _eval_worker(args):
    """Worker function for multiprocessing - already in a spawned process"""
    task_id, sample_path, sample_id, device_id, arch, run_nsys, save_nsys_csv, timeout = args

    try:
        result = eval_single_sample(
            task_id, sample_path, sample_id, device_id, arch, run_nsys, save_nsys_csv
        )
        return result
    except Exception as e:
        return EvalResult(
            task_id=task_id, sample_id=sample_id,
            error=f"Worker error: {str(e)[:200]}",
        )


def save_eval_result(result: EvalResult, eval_file_path: str):
    """Append a single eval result to the results JSON file"""
    if os.path.exists(eval_file_path):
        with open(eval_file_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    key = f"{result.task_id}_sample_{result.sample_id}"
    all_results[key] = result.to_dict()

    os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
    with open(eval_file_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


def batch_eval(
    run_name: str,
    task_ids: list[str] = None,
    arch: str = None,
    num_gpu_devices: int = None,
    timeout: int = None,
    run_nsys: bool = None,
    save_nsys_csv: bool = False,
):
    """
    Batch evaluation for all samples in a run.
    
    Args:
        run_name: Name of the run directory (under runs/)
        task_ids: List of task IDs to evaluate (None = all)
        arch: GPU architecture (None = use config default)
        num_gpu_devices: Number of GPUs to use in parallel (None = use config default)
        timeout: Per-sample timeout (None = use config default)
        run_nsys: Whether to run nsys profiling (None = use config default)
        save_nsys_csv: Whether to save nsys CSV and summary to run directory
    """
    # Use config defaults if not provided
    config = get_config()
    if arch is None:
        arch = config.gpu.arch
    if num_gpu_devices is None:
        num_gpu_devices = config.eval.num_gpu_devices
    if timeout is None:
        timeout = config.eval.timeout
    if run_nsys is None:
        run_nsys = config.profiling.nsys_enabled
    
    # Ensure a safe multiprocessing start method.
    #
    # Default to "spawn" for CUDA context isolation, BUT:
    # when the caller runs Python from stdin/heredoc (python - <<'PY' ...),
    # spawn will try to re-run __main__ from a fake path like ".../<stdin>",
    # causing FileNotFoundError in child processes.
    #
    # In that case, fall back to "fork" on Linux to keep CLI experiments smooth.
    cur_method = mp.get_start_method(allow_none=True)
    if cur_method is None:
        main_path = getattr(sys.modules.get("__main__"), "__file__", None)
        argv0 = sys.argv[0] if sys.argv else ""
        from_stdin = (main_path is None) or (argv0 in ("-c", "<stdin>")) or (isinstance(main_path, str) and main_path.endswith("<stdin>"))
        if from_stdin:
            try:
                mp.set_start_method("fork")
            except RuntimeError:
                mp.set_start_method("spawn")
        else:
            mp.set_start_method("spawn")

    run_dir = os.path.join(ORBENCH_ROOT, "runs", run_name)
    # Always write to a fresh results file for each invocation (no SKIP-by-existing).
    # Name includes date/time for easy experiment tracking.
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    eval_file = os.path.join(run_dir, f"eval_results_{ts}.json")

    if not os.path.exists(run_dir):
        print(f"Run directory not found: {run_dir}")
        return

    # Discover tasks and samples
    if task_ids is None:
        task_ids = [d for d in sorted(os.listdir(run_dir))
                    if os.path.isdir(os.path.join(run_dir, d))]

    # Build work list
    work_list = []
    for task_id in task_ids:
        task_dir = os.path.join(run_dir, task_id)
        if not os.path.isdir(task_dir):
            continue

        for filename in sorted(os.listdir(task_dir)):
            if filename.startswith("sample_") and filename.endswith(".cu"):
                sample_id = int(filename.split("_")[1].split(".")[0])
                sample_path = os.path.join(task_dir, filename)

                work_list.append((task_id, sample_path, sample_id))

    print(f"{'='*60}")
    print(f"  ORBench Batch Evaluation")
    print(f"  Run: {run_name}")
    print(f"  Tasks: {len(task_ids)}")
    print(f"  Samples to evaluate: {len(work_list)}")
    print(f"  GPUs: {num_gpu_devices}")
    print(f"{'='*60}\n")

    if not work_list:
        print("Nothing to evaluate.")
        return

    # Phase 1: Compile all (CPU parallel, no GPU needed)
    print("[Phase 1] Compiling solutions...")
    # (compilation happens inside eval_single_sample, but could be separated)

    # Phase 2: Evaluate in GPU-parallel batches
    print(f"[Phase 2] Evaluating on {num_gpu_devices} GPU(s)...\n")

    batch_size = num_gpu_devices
    total_done = 0

    for batch_start in range(0, len(work_list), batch_size):
        batch = work_list[batch_start:batch_start + batch_size]

        # Prepare work args with device assignment
        batch_args = [
            (task_id, sample_path, sample_id,
             i % batch_size,  # device_id
             arch, run_nsys, save_nsys_csv, timeout)
            for i, (task_id, sample_path, sample_id) in enumerate(batch)
        ]

        start_time = time.time()

        with mp.Pool(batch_size) as pool:
            async_results = [pool.apply_async(_eval_worker, (args,)) for args in batch_args]

            for i, ar in enumerate(async_results):
                task_id, _, sample_id = batch[i]
                try:
                    eval_result = ar.get(timeout=timeout + 30)
                except Exception as e:
                    eval_result = EvalResult(
                        task_id=task_id, sample_id=sample_id,
                        error=f"Batch error: {str(e)[:200]}",
                    )

                # Print result
                status_parts = []
                if eval_result.compiled:
                    status_parts.append("compiled")
                else:
                    status_parts.append("COMPILE_FAIL")
                if eval_result.correct:
                    status_parts.append("correct")
                if eval_result.benchmark and eval_result.benchmark.get("speedup_e2e", -1) > 0:
                    status_parts.append(f"speedup={eval_result.benchmark['speedup_e2e']:.1f}x")

                status = " | ".join(status_parts)
                print(f"  [{task_id}/sample_{sample_id}] {status}")

                # Save incrementally
                save_eval_result(eval_result, eval_file)
                total_done += 1

        elapsed = time.time() - start_time
        print(f"  Batch done in {elapsed:.1f}s ({total_done}/{len(work_list)} total)\n")

    print(f"\n{'='*60}")
    print(f"  Evaluation complete: {total_done} samples")
    print(f"  Results saved to: {eval_file}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="ORBench Batch Evaluation")
    parser.add_argument("--run", required=True, help="Run name")
    parser.add_argument("--tasks", nargs="*", default=None, help="Task IDs to evaluate")
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--no-nsys", action="store_true")
    parser.add_argument("--save-nsys", action="store_true", help="Save nsys CSV to run dir")
    args = parser.parse_args()

    batch_eval(
        run_name=args.run,
        task_ids=args.tasks,
        arch=args.arch,
        num_gpu_devices=args.gpus,
        timeout=args.timeout,
        run_nsys=not args.no_nsys,
        save_nsys_csv=args.save_nsys,
    )


if __name__ == "__main__":
    main()