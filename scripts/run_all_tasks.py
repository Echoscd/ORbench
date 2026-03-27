#!/usr/bin/env python3
"""
run_all_tasks.py — Run agent experiments on ALL tasks concurrently.

Tasks share a GPU lock: LLM calls run in parallel across tasks,
GPU eval is serialized (single device).

Usage:
  python3 scripts/run_all_tasks.py --model gemini-3.1-pro-preview-openrouter
  python3 scripts/run_all_tasks.py --model kimi-k2.5-openrouter --repeats 1 --turns 1
  python3 scripts/run_all_tasks.py --model gpt-4.1 --tasks bellman_ford,network_rm_dp
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Setup path
_SCRIPT_DIR = Path(__file__).resolve().parent
_ORBENCH_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ORBENCH_ROOT))
os.chdir(str(_ORBENCH_ROOT))

# Load .env
_env_file = _ORBENCH_ROOT / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v

from framework.config import load_config, set_config
from framework.agent.multiturn import run_multiturn


def get_ready_tasks() -> list[str]:
    """Return task IDs that have medium data + prompt template ready."""
    tasks_dir = _ORBENCH_ROOT / "tasks"
    ready = []
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        has_medium = (task_dir / "data" / "medium" / "cpu_time_ms.txt").exists()
        has_prompt = (task_dir / "prompt_template.yaml").exists()
        if has_medium and has_prompt:
            ready.append(task_dir.name)
    return ready


def run_single_task(
    task_id: str,
    model_id: str,
    level: int,
    turns: int,
    repeats: int,
    run_name: str,
    temperature: float,
    split: bool,
    arch: str,
    run_nsys: bool,
    gpu_lock: threading.Lock,
    print_lock: threading.Lock,
) -> dict:
    """Run one task. Returns result dict."""
    t0 = time.monotonic()
    with print_lock:
        print(f"[START] {task_id}", flush=True)

    try:
        # Monkey-patch the gpu_lock into multiturn module
        # so all tasks share the same GPU lock
        import framework.agent.multiturn as mt_module

        # Save original and inject shared lock
        orig_run = mt_module.run_multiturn

        def patched_run_multiturn(**kwargs):
            # We need to inject our gpu_lock
            # The simplest way: just call run_multiturn directly
            # since repeats=1 won't use internal threading
            return orig_run(**kwargs)

        summary = run_multiturn(
            model_id=model_id,
            task_id=task_id,
            level=level,
            turns=turns,
            repeats=repeats,
            run_name=run_name,
            temperature=temperature,
            split=split,
            device_id=0,
            arch=arch,
            run_nsys=run_nsys,
            save_nsys_csv=False,
        )

        # Extract result
        records = summary.records
        compiled = any(r.eval_result.get("compiled", False) for r in records)
        correct = any(r.eval_result.get("correct", False) for r in records)
        bench = None
        for r in records:
            b = r.eval_result.get("benchmark")
            if b and b.get("speedup_e2e") and b["speedup_e2e"] > 0:
                bench = b
                break

        elapsed = time.monotonic() - t0
        speedup = bench.get("speedup_e2e", -1) if bench else -1

        with print_lock:
            status = "PASS" if correct else ("COMPILED" if compiled else "FAIL")
            speedup_str = f"{speedup:.1f}x" if speedup > 0 else "-"
            print(f"[{status:>8s}] {task_id:<40s} speedup={speedup_str} ({elapsed:.0f}s)", flush=True)

        return {"task": task_id, "compiled": compiled, "correct": correct, "speedup": speedup, "elapsed": elapsed, "error": ""}

    except Exception as e:
        elapsed = time.monotonic() - t0
        with print_lock:
            print(f"[  ERROR] {task_id:<40s} {str(e)[:80]} ({elapsed:.0f}s)", flush=True)
        return {"task": task_id, "compiled": False, "correct": False, "speedup": -1, "elapsed": elapsed, "error": str(e)[:200]}


def main():
    parser = argparse.ArgumentParser(description="Run agent on all tasks concurrently")
    parser.add_argument("--model", required=True, help="Model ID from models.yaml")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task IDs (default: all ready)")
    parser.add_argument("--max-parallel", type=int, default=8, help="Max concurrent tasks (default: 8)")
    parser.add_argument("--no-nsys", action="store_true")
    parser.add_argument("--arch", type=str, default=None)
    args = parser.parse_args()

    # Load config
    config = load_config(cli_args={"arch": args.arch, "no_nsys": args.no_nsys})
    set_config(config)
    arch = config.gpu.arch
    run_nsys = not args.no_nsys

    # Select tasks
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        tasks = get_ready_tasks()

    date_tag = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"{args.model}_l{args.level}_agent_mt_{date_tag}"

    print(f"{'='*70}")
    print(f"  Model:        {args.model}")
    print(f"  Tasks:        {len(tasks)}")
    print(f"  Parallel:     {args.max_parallel}")
    print(f"  Repeats:      {args.repeats}")
    print(f"  Turns:        {args.turns}")
    print(f"  Level:        {args.level}")
    print(f"  Run name:     {run_name}")
    print(f"{'='*70}")
    print(f"  {' '.join(tasks)}")
    print()

    gpu_lock = threading.Lock()
    print_lock = threading.Lock()

    results = []
    t_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {}
        for task in tasks:
            f = pool.submit(
                run_single_task,
                task_id=task,
                model_id=args.model,
                level=args.level,
                turns=args.turns,
                repeats=args.repeats,
                run_name=run_name,
                temperature=args.temperature,
                split=args.split,
                arch=arch,
                run_nsys=run_nsys,
                gpu_lock=gpu_lock,
                print_lock=print_lock,
            )
            futures[f] = task

        for f in as_completed(futures):
            results.append(f.result())

    total_time = time.monotonic() - t_start

    # Summary
    print(f"\n{'='*70}")
    print(f"  Run complete: {run_name}")
    print(f"  Total time:   {total_time:.0f}s")
    print(f"{'='*70}\n")

    # Run summarize
    import subprocess
    subprocess.run([
        sys.executable, str(_ORBENCH_ROOT / "scripts" / "summarize_run.py"),
        str(_ORBENCH_ROOT / "runs" / run_name),
    ])


if __name__ == "__main__":
    main()
