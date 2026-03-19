"""
compile.py - Compile CUDA solutions using nvcc
Compilation can run on CPU-only machines (no GPU needed).
"""

import os
import subprocess
import shutil
from dataclasses import dataclass

from .task import load_task, ORBENCH_ROOT, TASKS_DIR
from .config import get_config


@dataclass
class CompileResult:
    success: bool
    executable_path: str = ""
    stdout: str = ""
    stderr: str = ""


def compile_solution(
    task_id: str,
    solution_path: str,
    build_dir: str = None,
    arch: str = None,
    timeout: int = 60,
) -> CompileResult:
    """
    Compile a single .cu solution file.
    
    Args:
        task_id: Task identifier
        solution_path: Path to the .cu source file
        build_dir: Directory for build artifacts (default: cache/{task_id}/{hash})
        arch: CUDA architecture target
        timeout: Compilation timeout in seconds
    
    Returns:
        CompileResult with success flag, executable path, and compiler output
    """
    task = load_task(task_id)
    
    # Use config default if arch not provided
    if arch is None:
        config = get_config()
        arch = config.gpu.arch

    if build_dir is None:
        # Use content hash for cache key
        with open(solution_path, "r") as f:
            content_hash = str(abs(hash(f.read())))[:12]
        build_dir = os.path.join(ORBENCH_ROOT, "cache", task_id, content_hash)

    os.makedirs(build_dir, exist_ok=True)

    # Copy solution to build directory
    src_in_build = os.path.join(build_dir, "solution.cu")
    shutil.copy2(solution_path, src_in_build)

    exe_path = os.path.join(build_dir, "solution_gpu")

    # ORBench v2.1 compilation: harness_gpu.cu + task_io.cu + solution.cu
    harness_path = os.path.join(ORBENCH_ROOT, "framework", "harness_gpu.cu")
    task_io_path = os.path.join(TASKS_DIR, task_id, "task_io.cu")
    include_dir = os.path.join(ORBENCH_ROOT, "framework")
    cmd = [
        "nvcc", "-O2", f"-arch={arch}",
        "-I", include_dir,
    ]

    # compute_only mode: pass macro to harness
    if task.interface_mode == "compute_only":
        cmd.append("-DORBENCH_COMPUTE_ONLY")

    cmd.extend([
        harness_path,
        task_io_path,
        src_in_build,
        "-o", exe_path,
    ])

    # Add extra flags from task config
    if task.extra_build_flags:
        cmd.extend(task.extra_build_flags.split())

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            text=True,
        )

        if result.returncode == 0:
            return CompileResult(
                success=True,
                executable_path=exe_path,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        else:
            return CompileResult(
                success=False,
                stdout=result.stdout,
                stderr=result.stderr,
            )

    except subprocess.TimeoutExpired:
        return CompileResult(success=False, stderr="Compilation timed out")
    except Exception as e:
        return CompileResult(success=False, stderr=str(e))


def batch_compile(
    tasks_and_solutions: list[tuple[str, str]],
    arch: str = None,
    num_workers: int = None,
) -> dict[str, CompileResult]:
    """
    Compile multiple solutions in parallel (CPU-only, no GPU needed).
    
    Args:
        tasks_and_solutions: List of (task_id, solution_path) tuples
        arch: CUDA architecture
        num_workers: Number of parallel compilation processes
    
    Returns:
        Dict mapping solution_path -> CompileResult
    """
    import multiprocessing as mp
    
    # Use config defaults if not provided
    config = get_config()
    if arch is None:
        arch = config.gpu.arch
    if num_workers is None:
        num_workers = config.eval.num_cpu_workers

    def _compile_one(args):
        task_id, solution_path = args
        return solution_path, compile_solution(task_id, solution_path, arch=arch)

    results = {}
    with mp.Pool(num_workers) as pool:
        for sol_path, result in pool.imap_unordered(_compile_one, tasks_and_solutions):
            results[sol_path] = result
            status = "OK" if result.success else "FAIL"
            print(f"  [{status}] {sol_path}")

    return results


def cleanup_build_dir(task_id: str, content_hash: str = None):
    """Remove cached build artifacts"""
    if content_hash:
        build_dir = os.path.join(ORBENCH_ROOT, "cache", task_id, content_hash)
    else:
        build_dir = os.path.join(ORBENCH_ROOT, "cache", task_id)

    if os.path.exists(build_dir):
        shutil.rmtree(build_dir, ignore_errors=True)
