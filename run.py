#!/usr/bin/env python3
"""
ORBench - General-Purpose CPU-to-CUDA Acceleration Benchmark for LLMs

Usage:
    # Generate solutions (legacy single model)
    python run.py generate --task bellman_ford --model claude-sonnet-4-20250514 --level 2

    # Batch generate across multiple models (new)
    python run.py generate-batch --models claude-sonnet-4 gpt-4o deepseek-v3 \
        --tasks bellman_ford --levels 2 --samples 5

    # Evaluate a run
    python run.py eval --run claude_sonnet_4_l2 --gpus 1

    # Analyze results
    python run.py analyze --run claude_sonnet_4_l2

    # Cross-model comparison (new)
    python run.py compare --runs claude-sonnet-4_l2 gpt-4o_l2 deepseek-v3_l2

    # List available tasks / models
    python run.py list
"""

import sys
import os
import argparse

# Load .env if python-dotenv is available (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from framework.config import load_config, set_config, get_config


# ═══════════════════════════════════════════════════════════════
#  list
# ═══════════════════════════════════════════════════════════════

def cmd_list(args):
    from framework.task import load_all_tasks

    tasks = load_all_tasks()
    print(f"\nAvailable tasks ({len(tasks)}):\n")
    print(f"  {'ID':<25s} {'Category':<20s} {'Diff':>4s}  Tags")
    print(f"  {'-'*70}")
    for t in tasks:
        print(f"  {t.task_id:<25s} {t.category:<20s} {'*'*t.difficulty:>4s}  {', '.join(t.tags)}")
    print()

    # Also list models if models.yaml exists
    try:
        from framework.llm.registry import LLMRegistry
        registry = LLMRegistry()
        models = registry.list_models()
        print(f"Registered models ({len(models)}):\n")
        print(f"  {'Model ID':<22s} {'Provider':<12s} {'Model String':<35s} {'$/1k in':>8s} {'$/1k out':>9s}")
        print(f"  {'-'*90}")
        for mid in models:
            mc = registry.get_model_config(mid)
            print(f"  {mid:<22s} {mc['provider']:<12s} {mc.get('model_string',''):<35s} "
                  f"{mc.get('cost_per_1k_input',0):>8.5f} {mc.get('cost_per_1k_output',0):>9.5f}")
        print()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
#  generate (legacy single-model)
# ═══════════════════════════════════════════════════════════════

def cmd_generate(args):
    from framework.generate import generate_solutions

    # Load config and merge CLI args
    cli_args = {}
    if args.model is not None:
        cli_args["model"] = args.model
    if args.api_base is not None:
        cli_args["api_base"] = args.api_base
    if args.samples is not None:
        cli_args["samples"] = args.samples

    config = load_config(cli_args=cli_args)
    set_config(config)

    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    if not api_key:
        print("ERROR: Set --api-key or LLM_API_KEY environment variable")
        sys.exit(1)

    generate_solutions(
        task_id=args.task,
        model=config.llm.model,
        level=args.level,
        num_samples=config.llm.num_samples,
        api_key=api_key,
        api_base=config.llm.api_base,
        run_name=args.run_name,
    )


# ═══════════════════════════════════════════════════════════════
#  generate-batch (multi-model, registry-based)
# ═══════════════════════════════════════════════════════════════

def cmd_generate_batch(args):
    from framework.llm.registry import LLMRegistry
    from framework.llm.scheduler import GenerationScheduler, estimate_cost
    from framework.task import load_all_tasks

    registry = LLMRegistry(args.models_yaml)

    # Determine models
    if args.models:
        model_ids = args.models
        # Validate
        for mid in model_ids:
            registry.get_model_config(mid)  # raises if unknown
    else:
        model_ids = registry.list_models()

    # Determine tasks
    if args.tasks:
        task_ids = args.tasks
    else:
        task_ids = [t.task_id for t in load_all_tasks()]

    levels = args.levels or [2]
    num_samples = args.samples

    scheduler = GenerationScheduler(registry, runs_dir=os.path.join("runs"))
    jobs = scheduler.build_jobs(model_ids, task_ids, levels, num_samples)

    print(f"\n{'='*60}")
    print(f"  ORBench Batch Generation")
    print(f"{'='*60}")
    print(f"  Models:  {len(model_ids):>3d}  {model_ids}")
    print(f"  Tasks:   {len(task_ids):>3d}  {task_ids}")
    print(f"  Levels:  {levels}")
    print(f"  Samples: {num_samples}")
    print(f"  Total:   {len(jobs)} generation jobs")

    # Cost estimate
    est = estimate_cost(registry, jobs)
    print(f"  Est. cost: ${est:.2f}")
    print(f"{'='*60}\n")

    # Confirm unless --yes
    if not args.yes and est > 0.0:
        confirm = input("  Proceed? [y/N] ")
        if confirm.strip().lower() != "y":
            print("  Aborted.")
            return

    progress_file = os.path.join("runs", f"generation_progress_{args.run_tag}.json")

    results = scheduler.run(
        jobs,
        max_workers_per_provider=args.workers,
        progress_file=progress_file,
        temperature=args.temperature,
    )

    # Print generated run names for subsequent eval
    run_names = sorted(set(scheduler._run_name(r.job) for r in results if r.success))
    if run_names:
        print(f"\n  Generated runs (for eval):")
        for rn in run_names:
            print(f"    python run.py eval --run {rn} --sizes small")
    print()


# ═══════════════════════════════════════════════════════════════
#  eval
# ═══════════════════════════════════════════════════════════════

def cmd_eval(args):
    from framework.batch_eval import batch_eval

    # Set ORBENCH_VALIDATE_SIZES environment variable if --sizes is specified
    if args.sizes:
        os.environ["ORBENCH_VALIDATE_SIZES"] = ",".join(args.sizes)

    # Load config and merge CLI args
    cli_args = {
        "arch": args.arch,
        "gpus": args.gpus,
        "timeout": args.timeout,
        "no_nsys": args.no_nsys,
    }
    config = load_config(cli_args=cli_args)
    set_config(config)

    batch_eval(
        run_name=args.run,
        task_ids=args.tasks,
        arch=config.gpu.arch,
        num_gpu_devices=config.eval.num_gpu_devices,
        timeout=config.eval.timeout,
        run_nsys=config.profiling.nsys_enabled,
        save_nsys_csv=args.save_nsys,
    )


# ═══════════════════════════════════════════════════════════════
#  analyze
# ═══════════════════════════════════════════════════════════════

def cmd_analyze(args):
    from framework.analyze import compute_summary, print_summary
    import json

    summary = compute_summary(args.run)
    print_summary(summary)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)


# ═══════════════════════════════════════════════════════════════
#  compare (cross-model)
# ═══════════════════════════════════════════════════════════════

def cmd_compare(args):
    from framework.analyze import load_eval_results
    from framework.task import load_all_tasks
    from collections import defaultdict
    import json

    run_names = args.runs
    if not run_names or len(run_names) < 2:
        print("ERROR: --runs requires at least 2 run names for comparison")
        sys.exit(1)

    # Load results for each run
    all_run_data = {}
    for rn in run_names:
        try:
            all_run_data[rn] = load_eval_results(rn)
        except FileNotFoundError:
            print(f"  [WARN] No results for run '{rn}', skipping.")

    if len(all_run_data) < 2:
        print("ERROR: Need at least 2 runs with results to compare.")
        sys.exit(1)

    # Collect all task_ids across all runs
    all_tasks = set()
    for rn, results in all_run_data.items():
        for key, res in results.items():
            all_tasks.add(res["task_id"])

    print(f"\n{'═'*70}")
    print(f"  ORBench Cross-Model Comparison")
    print(f"{'═'*70}\n")

    for task_id in sorted(all_tasks):
        # Try to load task info for difficulty display
        try:
            from framework.task import load_task
            task = load_task(task_id)
            diff_str = "*" * task.difficulty
        except Exception:
            diff_str = "?"

        print(f"  Task: {task_id} (difficulty: {diff_str})\n")
        print(f"  {'Model (run)':<30s} {'Compile':>8s} {'Pass':>6s} {'Speedup(best)':>14s} {'Cost/sample':>12s}")
        print(f"  {'─'*72}")

        for rn in run_names:
            results = all_run_data.get(rn, {})

            # Find all samples for this task in this run
            task_samples = [
                res for key, res in results.items()
                if res.get("task_id") == task_id
            ]

            if not task_samples:
                print(f"  {rn:<30s} {'—':>8s} {'—':>6s} {'—':>14s} {'—':>12s}")
                continue

            n = len(task_samples)
            compiled = sum(1 for s in task_samples if s.get("compiled", False))
            correct = sum(1 for s in task_samples if s.get("correct", False))

            speedups = []
            costs = []
            for s in task_samples:
                bench = s.get("benchmark")
                if bench:
                    sp = bench.get("speedup_e2e", -1)
                    if sp and sp > 0:
                        speedups.append(sp)

                # Load per-sample meta if available
                # (cost info from generation metadata)

            best_sp = f"{max(speedups):.1f}x" if speedups else "N/A"
            comp_str = f"{compiled}/{n}"
            pass_str = f"{correct}/{n}"

            # Try to find cost info from meta files
            total_cost = 0.0
            meta_found = False
            from framework.task import ORBENCH_ROOT
            run_dir = os.path.join(ORBENCH_ROOT, "runs", rn, task_id)
            if os.path.isdir(run_dir):
                for fn in os.listdir(run_dir):
                    if fn.endswith("_meta.json"):
                        try:
                            with open(os.path.join(run_dir, fn)) as f:
                                meta = json.load(f)
                            total_cost += meta.get("cost_usd", 0)
                            meta_found = True
                        except Exception:
                            pass

            cost_str = f"${total_cost / n:.3f}" if meta_found and n > 0 else "—"

            print(f"  {rn:<30s} {comp_str:>8s} {pass_str:>6s} {best_sp:>14s} {cost_str:>12s}")

        print()

    print(f"{'═'*70}\n")

    # Save comparison JSON if requested
    if args.output:
        comparison = {
            "runs": run_names,
            "tasks": {},
        }
        for task_id in sorted(all_tasks):
            comparison["tasks"][task_id] = {}
            for rn in run_names:
                results = all_run_data.get(rn, {})
                task_samples = [
                    res for key, res in results.items()
                    if res.get("task_id") == task_id
                ]
                n = len(task_samples)
                compiled = sum(1 for s in task_samples if s.get("compiled", False))
                correct = sum(1 for s in task_samples if s.get("correct", False))
                speedups = [
                    s.get("benchmark", {}).get("speedup_e2e", -1)
                    for s in task_samples
                    if s.get("benchmark", {}).get("speedup_e2e", -1) > 0
                ]
                comparison["tasks"][task_id][rn] = {
                    "num_samples": n,
                    "compiled": compiled,
                    "correct": correct,
                    "best_speedup": max(speedups) if speedups else None,
                }

        with open(args.output, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"  Comparison saved to {args.output}")


# ═══════════════════════════════════════════════════════════════
#  main / CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ORBench CLI")
    subparsers = parser.add_subparsers(dest="command")

    # ── list ──
    subparsers.add_parser("list", help="List available tasks and models")

    # ── generate (legacy) ──
    p_gen = subparsers.add_parser("generate", help="Generate CUDA solutions (single model, legacy)")
    p_gen.add_argument("--task", required=True)
    p_gen.add_argument("--model", default=None, help="LLM model (overrides config.yaml)")
    p_gen.add_argument("--level", type=int, default=2, choices=[1, 2, 3])
    p_gen.add_argument("--samples", type=int, default=None, help="Number of samples (overrides config.yaml)")
    p_gen.add_argument("--api-key", default=None)
    p_gen.add_argument("--api-base", default=None, help="API base URL (overrides config.yaml)")
    p_gen.add_argument("--run-name", default=None)

    # ── generate-batch (new, multi-model) ──
    p_batch = subparsers.add_parser("generate-batch",
        help="Batch-generate solutions from multiple LLMs (uses models.yaml)")
    p_batch.add_argument("--models", nargs="*", default=None,
        help="Model IDs from models.yaml (default: all)")
    p_batch.add_argument("--tasks", nargs="*", default=None,
        help="Task IDs (default: all)")
    p_batch.add_argument("--levels", type=int, nargs="*", default=[2],
        help="Prompt levels (default: [2])")
    p_batch.add_argument("--samples", type=int, default=3,
        help="Number of samples per model×task×level")
    p_batch.add_argument("--workers", type=int, default=3,
        help="Max concurrent requests per provider")
    p_batch.add_argument("--temperature", type=float, default=0.7,
        help="Sampling temperature")
    p_batch.add_argument("--models-yaml", default=None,
        help="Path to models.yaml (default: ORBENCH_ROOT/models.yaml)")
    p_batch.add_argument("--run-tag", default="batch",
        help="Tag for progress file naming")
    p_batch.add_argument("--yes", action="store_true",
        help="Skip cost confirmation prompt")

    # ── eval ──
    p_eval = subparsers.add_parser("eval", help="Evaluate generated solutions")
    p_eval.add_argument("--run", required=True)
    p_eval.add_argument("--tasks", nargs="*", default=None)
    p_eval.add_argument("--sizes", nargs="*", default=None,
        help="Data sizes to evaluate (e.g., small medium large, default: all)")
    p_eval.add_argument("--arch", default=None, help="GPU architecture (overrides config.yaml)")
    p_eval.add_argument("--gpus", type=int, default=None, help="Number of GPUs (overrides config.yaml)")
    p_eval.add_argument("--timeout", type=int, default=None, help="Timeout in seconds (overrides config.yaml)")
    p_eval.add_argument("--no-nsys", action="store_true", help="Skip nsys profiling entirely")
    p_eval.add_argument("--save-nsys", action="store_true", help="Save nsys CSV and summary to run directory")

    # ── analyze ──
    p_ana = subparsers.add_parser("analyze", help="Analyze evaluation results")
    p_ana.add_argument("--run", required=True)
    p_ana.add_argument("--output", default=None, help="Save summary JSON")

    # ── compare (new) ──
    p_cmp = subparsers.add_parser("compare", help="Cross-model comparison of evaluation results")
    p_cmp.add_argument("--runs", nargs="+", required=True,
        help="Two or more run names to compare")
    p_cmp.add_argument("--output", default=None,
        help="Save comparison JSON to file")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "generate-batch":
        cmd_generate_batch(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
