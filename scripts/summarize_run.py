#!/usr/bin/env python3
"""
summarize_run.py — Summarize agent run results: pass rate, speedup stats.

Usage:
  python3 scripts/summarize_run.py <run_dir>
  python3 scripts/summarize_run.py runs/gemini-3.1-pro-preview-openrouter_l2_agent_mt_20260323_1602
"""

import json
import math
import os
import sys
from pathlib import Path


def summarize_run(run_dir: str):
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory")
        sys.exit(1)

    run_name = run_dir.name
    tasks = sorted([d.name for d in run_dir.iterdir() if d.is_dir()])

    print(f"{'='*80}")
    print(f"  Run: {run_name}")
    print(f"  Tasks: {len(tasks)}")
    print(f"{'='*80}")
    print()

    # Header
    print(f"{'Task':<40s} {'Compiled':>8s} {'Correct':>8s} {'E2E(ms)':>10s} {'Kernel(ms)':>11s} {'Speedup':>10s} {'KSpeedup':>10s}")
    print("-" * 97)

    total = 0
    compiled_count = 0
    correct_count = 0
    speedups_e2e = []
    speedups_kernel = []
    rows = []  # for CSV output

    for task in tasks:
        summary_path = run_dir / task / "agent_multiturn_summary.json"
        if not summary_path.exists():
            print(f"{task:<40s} {'N/A':>8s}")
            total += 1
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        records = summary.get("records", [])
        if not records:
            print(f"{task:<40s} {'EMPTY':>8s}")
            total += 1
            continue

        # Use last turn's result (best attempt)
        for rec in records:
            total += 1
            ev = rec.get("eval_result", {})
            compiled = ev.get("compiled", False)
            correct = ev.get("correct", False)
            bench = ev.get("benchmark") or {}

            e2e = bench.get("e2e_time_ms", {})
            if isinstance(e2e, dict):
                e2e_ms = e2e.get("mean", -1)
            else:
                e2e_ms = e2e if e2e else -1

            kernel_ms = bench.get("kernel_time_ms", -1) or -1
            speedup_e2e = bench.get("speedup_e2e", -1) or -1
            speedup_kernel = bench.get("speedup_kernel", -1) or -1

            if compiled:
                compiled_count += 1
            if correct:
                correct_count += 1
                if speedup_e2e and speedup_e2e > 0:
                    speedups_e2e.append(speedup_e2e)
                if speedup_kernel and speedup_kernel > 0:
                    speedups_kernel.append(speedup_kernel)

            compiled_str = "Y" if compiled else "N"
            correct_str = "Y" if correct else "N"
            e2e_str = f"{e2e_ms:.2f}" if e2e_ms > 0 else "-"
            kernel_str = f"{kernel_ms:.2f}" if kernel_ms > 0 else "-"
            speedup_str = f"{speedup_e2e:.1f}x" if speedup_e2e > 0 else "-"
            kspeedup_str = f"{speedup_kernel:.1f}x" if speedup_kernel > 0 else "-"

            turn = rec.get("turn", 0)
            label = f"{task} (t{turn})"
            print(f"{label:<40s} {compiled_str:>8s} {correct_str:>8s} {e2e_str:>10s} {kernel_str:>11s} {speedup_str:>10s} {kspeedup_str:>10s}")
            rows.append([task, turn, compiled_str, correct_str, e2e_str, kernel_str, speedup_str, kspeedup_str])

    print("-" * 97)

    # Summary stats
    print()
    print(f"  Total samples:     {total}")
    print(f"  Compiled:          {compiled_count}/{total} ({100*compiled_count/total:.0f}%)" if total else "")
    print(f"  Correct (pass):    {correct_count}/{total} ({100*correct_count/total:.0f}%)" if total else "")

    geo_mean_e2e = None
    geo_mean_k = None
    if speedups_e2e:
        geo_mean_e2e = math.exp(sum(math.log(s) for s in speedups_e2e) / len(speedups_e2e))
        arith_mean_e2e = sum(speedups_e2e) / len(speedups_e2e)
        median_e2e = sorted(speedups_e2e)[len(speedups_e2e) // 2]
        print(f"  Speedup E2E (geo): {geo_mean_e2e:.1f}x  (n={len(speedups_e2e)} passing tasks)")
        print(f"  Speedup E2E (avg): {arith_mean_e2e:.1f}x")
        print(f"  Speedup E2E (med): {median_e2e:.1f}x")
        print(f"  Speedup E2E range: {min(speedups_e2e):.1f}x ~ {max(speedups_e2e):.1f}x")

    if speedups_kernel:
        geo_mean_k = math.exp(sum(math.log(s) for s in speedups_kernel) / len(speedups_kernel))
        print(f"  Speedup Kernel (geo): {geo_mean_k:.1f}x  (n={len(speedups_kernel)})")

    print()

    # Write CSV
    csv_path = run_dir / "summary.csv"
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "turn", "compiled", "correct", "e2e_ms", "kernel_ms", "speedup_e2e", "speedup_kernel"])
        for row in rows:
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(["_summary", "", "compiled_rate", f"{compiled_count}/{total}",
                         "pass_rate", f"{correct_count}/{total}",
                         "geo_speedup_e2e", f"{geo_mean_e2e:.1f}x" if geo_mean_e2e else "",
                         "geo_speedup_kernel", f"{geo_mean_k:.1f}x" if geo_mean_k else ""])
    print(f"  CSV saved: {csv_path}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/summarize_run.py <run_dir> [run_dir2 ...]")
        sys.exit(1)
    for run_dir in sys.argv[1:]:
        summarize_run(run_dir)
