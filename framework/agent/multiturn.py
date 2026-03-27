"""
multiturn.py - Multi-turn agent pipeline:

Loop:
  (turn 0) generate solution from base task prompt
  eval/benchmark/profile (compile + correctness + timing + optional nsys breakdown)
  (turn 1..T-1) generate improved solution using previous code + eval feedback

Designed to explore how different agent modes affect speedup.
"""

from __future__ import annotations

import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import traceback

from ..config import get_config
from ..generate import extract_cuda_code
from ..task import load_prompt, ORBENCH_ROOT
from ..batch_eval import eval_single_sample, EvalResult
from ..llm.registry import LLMRegistry

from .prompts import build_feedback_prompt
from .plot_metrics import load_turn_metrics, write_csv, plot_png


@dataclass
class TurnRecord:
    turn: int
    sample_id: int
    source_path: str
    raw_path: str
    prompt_path: str
    eval_result: dict


@dataclass
class MultiTurnRunSummary:
    run_name: str
    model_id: str
    task_id: str
    level: int
    turns: int
    repeats: int
    split: bool
    timestamp: str
    records: list[TurnRecord]


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def _format_eval_summary(er: EvalResult) -> str:
    """
    Convert EvalResult into a compact, LLM-friendly feedback block.
    Emphasize: compile errors, correctness errors, per-kernel time breakdown, mem time.
    """
    lines: list[str] = []

    # Compile
    if not er.compiled:
        lines.append("### Compile\nFAIL")
        if er.compile_error:
            lines.append("Compile error (truncated):")
            lines.append("```")
            lines.append(er.compile_error.strip()[:8000])
            lines.append("```")
        return "\n".join(lines) + "\n"

    # Correctness
    lines.append("### Correctness")
    lines.append("PASS" if er.correct else "FAIL")
    if er.error:
        # Could be mismatch or benchmark error
        lines.append("Error/detail (truncated):")
        lines.append("```")
        lines.append(er.error.strip()[:8000])
        lines.append("```")

    # Performance
    bench = er.benchmark or {}
    if bench:
        lines.append("### Performance (GPU)")
        for k in [
            "init_ms",
            "kernel_time_ms",
            "memcpy_overhead_ms",
            "gpu_utilization",
            "num_kernel_launches",
            "speedup_e2e",
            "speedup_kernel",
        ]:
            if k in bench and bench[k] is not None:
                lines.append(f"- {k}: {bench[k]}")

        # Hot kernels
        kern = bench.get("kernel_summary") or {}
        if isinstance(kern, dict) and kern:
            # sort by total_ms desc
            items = []
            for name, st in kern.items():
                try:
                    items.append((float(st.get("total_ms", 0.0)), name, st))
                except Exception:
                    continue
            items.sort(reverse=True)
            lines.append("### Hot Kernels (top 8 by total_ms)")
            for total_ms, name, st in items[:8]:
                lines.append(
                    f"- {name}: total_ms={st.get('total_ms')} count={st.get('count')} avg_us={st.get('avg_us')} time_pct={st.get('time_pct')}"
                )

        # Mem ops
        mem = bench.get("mem_time_summary") or {}
        if isinstance(mem, dict) and mem:
            lines.append("### Memory Ops (nsys summary)")
            # sort by total_ms desc if available
            mem_items = []
            for op, st in mem.items():
                try:
                    mem_items.append((float(st.get("total_ms", 0.0)), op, st))
                except Exception:
                    continue
            mem_items.sort(reverse=True)
            for total_ms, op, st in mem_items[:10]:
                lines.append(
                    f"- {op}: total_ms={st.get('total_ms')} count={st.get('count')} avg_us={st.get('avg_us')} time_pct={st.get('time_pct')}"
                )

    # Kernel count (static)
    lines.append("### Static Kernel Count")
    lines.append(f"- kernel_count: {er.kernel_count}")

    return "\n".join(lines) + "\n"


def run_multiturn(
    model_id: str,
    task_id: str,
    level: int = 2,
    turns: int = 2,
    repeats: int = 1,
    run_name: Optional[str] = None,
    temperature: float = 0.7,
    split: bool = False,
    device_id: int = 0,
    arch: Optional[str] = None,
    run_nsys: Optional[bool] = None,
    save_nsys_csv: bool = False,
) -> MultiTurnRunSummary:
    """
    Execute multi-turn generation+eval loops.

    Artifacts are saved under:
      ORBENCH_ROOT/runs/<run_name>/<task_id>/agent_r<rep>_t<turn>.*
    """
    if turns < 1:
        raise ValueError("turns must be >= 1")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    cfg = get_config()
    if arch is None:
        arch = cfg.gpu.arch
    if run_nsys is None:
        run_nsys = cfg.profiling.nsys_enabled

    if run_name is None:
        # Distinguish from normal runs
        date_tag = datetime.now().strftime("%Y%m%d_%H%M")
        run_name = f"{model_id}_l{level}_agent_mt_{date_tag}"

    registry = LLMRegistry()
    client = registry.get_client(model_id)

    base_prompt = load_prompt(task_id, level, split_kernels=split)

    records: list[TurnRecord] = []

    out_dir = os.path.join(ORBENCH_ROOT, "runs", run_name, task_id)
    os.makedirs(out_dir, exist_ok=True)
    progress_path = os.path.join(out_dir, "agent_progress.jsonl")

    # Concurrency controls: GPU eval is single-device, must be serialized;
    # LLM API calls are I/O-bound and can overlap across repeats.
    gpu_lock = threading.Lock()
    progress_lock = threading.Lock()

    def _safe_append_jsonl(obj: dict) -> None:
        with progress_lock:
            _append_jsonl(progress_path, obj)

    print(f"[agent] run={run_name} model={model_id} task={task_id} level={level} turns={turns} repeats={repeats} split={split}")
    print(f"[agent] artifacts_dir={out_dir}")

    def _run_single_repeat(rep: int) -> list[TurnRecord]:
        """Run all turns for one repeat. Turns are sequential; repeats run concurrently."""
        rep_records: list[TurnRecord] = []
        prev_code: Optional[str] = None
        prev_eval: Optional[EvalResult] = None

        for t in range(turns):
            _safe_append_jsonl({
                "ts": _now_ts(),
                "event": "turn_start",
                "run_name": run_name,
                "model_id": model_id,
                "task_id": task_id,
                "level": level,
                "rep": rep,
                "turn": t,
            })
            print(f"[agent] rep {rep+1}/{repeats} turn {t+1}/{turns}: generating...")

            # Build prompt
            if t == 0 or prev_code is None or prev_eval is None:
                prompt = base_prompt
            else:
                feedback = _format_eval_summary(prev_eval)
                prompt = build_feedback_prompt(base_prompt, prev_code, feedback)

            # Generate (LLM call — I/O bound, runs concurrently across repeats)
            try:
                resp = client.generate(prompt, max_tokens=client.model.get("max_tokens"), temperature=temperature)
                code = extract_cuda_code(resp.content)
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:500]}"
                _safe_append_jsonl({
                    "ts": _now_ts(),
                    "event": "generation_error",
                    "rep": rep,
                    "turn": t,
                    "error": msg,
                })
                print(f"[agent] rep {rep+1} turn {t+1}: generation FAILED: {msg}")
                sample_id = rep * 1000 + t
                rec = TurnRecord(
                    turn=t,
                    sample_id=sample_id,
                    source_path="",
                    raw_path="",
                    prompt_path="",
                    eval_result={
                        "task_id": task_id,
                        "sample_id": sample_id,
                        "kernel_count": 0,
                        "compiled": False,
                        "compile_error": "",
                        "correct": False,
                        "correctness_detail": None,
                        "benchmark": None,
                        "error": f"generation_error: {msg}",
                    },
                )
                rep_records.append(rec)
                prev_code = None
                prev_eval = None
                continue

            print(f"[agent] rep {rep+1}/{repeats} turn {t+1}/{turns}: evaluating...")

            # Save artifacts
            sample_id = rep * 1000 + t
            src_path = os.path.join(out_dir, f"agent_r{rep}_t{t}.cu")
            raw_path = os.path.join(out_dir, f"agent_r{rep}_t{t}_raw.txt")
            prompt_path = os.path.join(out_dir, f"agent_r{rep}_t{t}_prompt.txt")

            with open(src_path, "w", encoding="utf-8") as f:
                f.write(code)
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(resp.content)
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt)

            # Eval (GPU-bound — serialized via gpu_lock)
            with gpu_lock:
                try:
                    er = eval_single_sample(
                        task_id=task_id,
                        sample_path=src_path,
                        sample_id=sample_id,
                        device_id=device_id,
                        arch=arch,
                        run_nsys=run_nsys,
                        save_nsys_csv=save_nsys_csv,
                    )
                except Exception as e:
                    msg = f"{type(e).__name__}: {str(e)[:500]}"
                    _safe_append_jsonl({
                        "ts": _now_ts(),
                        "event": "eval_error",
                        "rep": rep,
                        "turn": t,
                        "error": msg,
                        "trace": traceback.format_exc(limit=5),
                    })
                    print(f"[agent] rep {rep+1} turn {t+1}: eval FAILED: {msg}")
                    er = EvalResult(task_id=task_id, sample_id=sample_id, error=f"eval_error: {msg}")

            rec = TurnRecord(
                turn=t,
                sample_id=sample_id,
                source_path=src_path,
                raw_path=raw_path,
                prompt_path=prompt_path,
                eval_result=er.to_dict(),
            )
            rep_records.append(rec)

            prev_code = code
            prev_eval = er

            _bench = er.benchmark or {}
            _init_ms = _bench.get("init_ms") if er.benchmark else None
            _e2e = _bench.get("e2e_time_ms", {}) if er.benchmark else {}
            _solve_ms = _e2e.get("mean") if isinstance(_e2e, dict) else _e2e
            _total_ms = None
            if _init_ms is not None and _solve_ms is not None:
                _total_ms = float(_init_ms) + float(_solve_ms)

            _safe_append_jsonl({
                "ts": _now_ts(),
                "event": "turn_done",
                "rep": rep,
                "turn": t,
                "compiled": bool(er.compiled),
                "correct": bool(er.correct),
                "kernel_count": int(er.kernel_count),
                "kernel_time_ms": _bench.get("kernel_time_ms") if er.benchmark else None,
                "init_ms": _init_ms,
                "solve_ms": _solve_ms,
                "total_ms": _total_ms,
                "speedup_e2e": _bench.get("speedup_e2e") if er.benchmark else None,
                "speedup_kernel": _bench.get("speedup_kernel") if er.benchmark else None,
            })
            _total_str = f"{_total_ms:.3f}" if _total_ms is not None else "N/A"
            print(f"[agent] rep {rep+1}/{repeats} turn {t+1}/{turns}: done (compiled={er.compiled} correct={er.correct} total_ms={_total_str} kernel_time_ms={_bench.get('kernel_time_ms') if er.benchmark else None})")

        return rep_records

    # Run repeats concurrently (capped to avoid API flooding)
    max_concurrent_repeats = min(repeats, 4)
    if repeats == 1:
        # Single repeat: no threading overhead
        records = _run_single_repeat(0)
    else:
        print(f"[agent] running {repeats} repeats concurrently (max {max_concurrent_repeats} threads)")
        rep_results: dict[int, list[TurnRecord]] = {}
        with ThreadPoolExecutor(max_workers=max_concurrent_repeats) as pool:
            future_to_rep = {pool.submit(_run_single_repeat, r): r for r in range(repeats)}
            for future in as_completed(future_to_rep):
                rep = future_to_rep[future]
                try:
                    rep_results[rep] = future.result()
                except Exception as e:
                    print(f"[agent] rep {rep+1} CRASHED: {e}")
                    rep_results[rep] = []
        # Merge records in rep order
        for r in range(repeats):
            records.extend(rep_results.get(r, []))

    summary = MultiTurnRunSummary(
        run_name=run_name,
        model_id=model_id,
        task_id=task_id,
        level=level,
        turns=turns,
        repeats=repeats,
        split=split,
        timestamp=_now_ts(),
        records=records,
    )

    # Write summary json
    summary_path = os.path.join(out_dir, "agent_multiturn_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, default=str)

    # Auto-generate metrics artifacts (CSV + PNG) for every agent run.
    # Best-effort: if matplotlib is not available, still write CSV.
    try:
        sp = os.path.abspath(summary_path)
        metrics = load_turn_metrics(Path(sp))
        write_csv(metrics, Path(out_dir) / "agent_metrics.csv")
        title = f"agent-multiturn: {run_name}/{task_id}"
        plot_png(metrics, Path(out_dir) / "agent_metrics.png", title)
    except Exception:
        # Don't fail the agent pipeline due to plotting issues.
        pass

    return summary


