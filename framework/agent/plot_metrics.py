#!/usr/bin/env python3
"""
plot_metrics.py - Plot multi-turn agent metrics from agent_multiturn_summary.json

Outputs:
- agent_metrics.csv
- agent_metrics.png (if matplotlib available)

Usage:
  python3 -m ORBench.framework.agent.plot_metrics /path/to/agent_multiturn_summary.json
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_get(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _to_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    xs = [v for v in vals if not math.isnan(v)]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


@dataclass
class TurnMetrics:
    rep: int
    turn: int
    compiled: bool
    correct: bool
    kernel_count: float
    init_ms: float
    solve_ms: float
    total_ms: float
    kernel_time_ms: float
    speedup_e2e: float
    speedup_kernel: float


def load_turn_metrics(summary_path: Path) -> List[TurnMetrics]:
    import json

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    recs = data.get("records", [])
    out: List[TurnMetrics] = []
    for r in recs:
        er = r.get("eval_result", {}) or {}
        sample_id = int(er.get("sample_id", r.get("sample_id", 0)) or 0)
        rep = sample_id // 1000
        turn = int(r.get("turn", 0))

        compiled = bool(er.get("compiled", False))
        correct = bool(er.get("correct", False))
        kernel_count = _to_float(er.get("kernel_count"))

        bench = er.get("benchmark") or {}
        init_ms = _to_float(bench.get("init_ms"))
        e2e_obj = bench.get("e2e_time_ms", {})
        solve_ms = _to_float(e2e_obj.get("mean") if isinstance(e2e_obj, dict) else e2e_obj)
        total_ms = init_ms + solve_ms if not (math.isnan(init_ms) or math.isnan(solve_ms)) else float("nan")
        kernel_time_ms = _to_float(bench.get("kernel_time_ms"))
        speedup_e2e = _to_float(bench.get("speedup_e2e"))
        speedup_kernel = _to_float(bench.get("speedup_kernel"))

        out.append(
            TurnMetrics(
                rep=rep,
                turn=turn,
                compiled=compiled,
                correct=correct,
                kernel_count=kernel_count,
                init_ms=init_ms,
                solve_ms=solve_ms,
                total_ms=total_ms,
                kernel_time_ms=kernel_time_ms,
                speedup_e2e=speedup_e2e,
                speedup_kernel=speedup_kernel,
            )
        )
    return out


def write_csv(metrics: List[TurnMetrics], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rep",
                "turn",
                "compiled",
                "correct",
                "kernel_count",
                "init_ms",
                "solve_ms",
                "total_ms",
                "kernel_time_ms",
                "speedup_e2e",
                "speedup_kernel",
            ]
        )
        for m in sorted(metrics, key=lambda x: (x.rep, x.turn)):
            w.writerow(
                [
                    m.rep,
                    m.turn,
                    int(m.compiled),
                    int(m.correct),
                    m.kernel_count,
                    m.init_ms,
                    m.solve_ms,
                    m.total_ms,
                    m.kernel_time_ms,
                    m.speedup_e2e,
                    m.speedup_kernel,
                ]
            )


def plot_png(metrics: List[TurnMetrics], out_path: Path, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    # Aggregate by turn across repeats: mean ± std
    turns = sorted({m.turn for m in metrics})
    by_turn = {t: [m for m in metrics if m.turn == t] for t in turns}

    def series(getter):
        means, stds = [], []
        for t in turns:
            vals = [getter(m) for m in by_turn[t]]
            mu, sd = _mean_std(vals)
            means.append(mu)
            stds.append(sd)
        return means, stds

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    fig.suptitle(title)

    # 1) kernel_count
    y, ysd = series(lambda m: m.kernel_count)
    ax = axes[0][0]
    ax.plot(turns, y, marker="o", label="kernel_count")
    ax.fill_between(turns, [a - b for a, b in zip(y, ysd)], [a + b for a, b in zip(y, ysd)], alpha=0.2)
    ax.set_xlabel("turn")
    ax.set_ylabel("kernel_count")
    ax.grid(True, alpha=0.3)

    # 2) kernel_time_ms
    y, ysd = series(lambda m: m.kernel_time_ms)
    ax = axes[0][1]
    ax.plot(turns, y, marker="o", label="kernel_time_ms")
    ax.fill_between(turns, [a - b for a, b in zip(y, ysd)], [a + b for a, b in zip(y, ysd)], alpha=0.2)
    ax.set_xlabel("turn")
    ax.set_ylabel("kernel_time_ms")
    ax.grid(True, alpha=0.3)

    # 3) total_ms (init + solve — the number that matters most)
    y, ysd = series(lambda m: m.total_ms)
    ax = axes[1][0]
    ax.plot(turns, y, marker="o", label="total_ms", color="tab:red")
    ax.fill_between(turns, [a - b for a, b in zip(y, ysd)], [a + b for a, b in zip(y, ysd)], alpha=0.2, color="tab:red")
    ax.set_xlabel("turn")
    ax.set_ylabel("total_ms (init + solve)")
    ax.grid(True, alpha=0.3)

    # 4) speedup (e2e + kernel)
    y1, s1 = series(lambda m: m.speedup_e2e)
    y2, s2 = series(lambda m: m.speedup_kernel)
    ax = axes[1][1]
    ax.plot(turns, y1, marker="o", label="speedup_e2e")
    ax.fill_between(turns, [a - b for a, b in zip(y1, s1)], [a + b for a, b in zip(y1, s1)], alpha=0.15)
    ax.plot(turns, y2, marker="o", label="speedup_kernel")
    ax.fill_between(turns, [a - b for a, b in zip(y2, s2)], [a + b for a, b in zip(y2, s2)], alpha=0.15)
    ax.set_xlabel("turn")
    ax.set_ylabel("speedup")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: python3 -m ORBench.framework.agent.plot_metrics /path/to/agent_multiturn_summary.json")
        return 2

    summary_path = Path(argv[1]).resolve()
    metrics = load_turn_metrics(summary_path)
    out_dir = summary_path.parent

    write_csv(metrics, out_dir / "agent_metrics.csv")

    title = f"agent-multiturn: {summary_path.parent.parent.name}/{summary_path.parent.name}"
    ok = plot_png(metrics, out_dir / "agent_metrics.png", title)
    if not ok:
        print("[plot] matplotlib not available; wrote agent_metrics.csv only.")
    else:
        print("[plot] wrote agent_metrics.png and agent_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


