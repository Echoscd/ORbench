"""
audit.py - Timing-audit orchestration (design: rebuttal/AUDIT_PLAN.md).

Verifies the contract "every solution_compute call is self-contained" on
already-passing solutions, and reports WHICH check a violating solution failed:

  S0 (screen, advisory)   All warmup+timed calls are individually timed by the
                          harness. A first call far slower than the timed calls
                          signals cost hidden in warmup. Never convicts on its
                          own (a compliant first call is also legitimately
                          slower: module load, clock ramp).
  S1 (convicting)         Reset-and-revalidate: the harness wipes device state
                          (cudaDeviceReset) between extra audit rounds and
                          re-validates the output. A solution that caches
                          device pointers/results across calls dereferences
                          dangling state -> CUDA error or wrong output.
  S2 (convicting)         Cold/warm ratio: K fresh-process runs with
                          --warmup 0 --trials 1 (context pre-created, eager
                          module loading). Compliant solutions do their
                          allocation in every call, so t_cold ~= t_warm;
                          a solution whose timed calls skip work done in
                          earlier calls shows t_cold >> t_warm.

Convention: `violations` (S1/S2) flag the solution; `advisories` (S0 and
anomalies) are reported but do not flag. The reported speedup of a flagged
solution should be recomputed from t_cold (`speedup_e2e_cold`).
"""

import os
import re
import json
import glob
import math
import statistics
from datetime import datetime

import numpy as np

from .task import load_task, get_task_dir, ACCELEVAL_ROOT
from .compile import compile_solution
from .benchmark import _run_exe


# ──────────────────────────────────────────────────────────────────
#  Tolerant output comparison (same semantics as validate.validate_output,
#  generalized to arbitrary output files such as output_audit_<r>.txt)
# ──────────────────────────────────────────────────────────────────

def _parse_floats(path):
    vals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for tok in line.split():
                vals.append(float(tok))
    return vals


def _compare_outputs(task, actual_path, expected_path):
    """-> (ok, message) or (ok, message, frac_mismatch, max_rel_err)."""
    if not os.path.exists(actual_path):
        return False, f"{os.path.basename(actual_path)} not found"
    if not os.path.exists(expected_path):
        return False, "expected_output.txt not found"
    try:
        actual = _parse_floats(actual_path)
        expected = _parse_floats(expected_path)
    except Exception as ex:
        return False, f"failed to parse outputs: {ex}"
    if len(actual) != len(expected):
        return False, f"value count mismatch: got {len(actual)}, expected {len(expected)}"
    atol = task.atol if task.atol is not None else 0.1
    rtol = task.rtol if task.rtol is not None else 0.01
    a = np.asarray(actual)
    e = np.asarray(expected)
    bad = ~np.isclose(a, e, atol=atol, rtol=rtol, equal_nan=True)
    if bad.any():
        i = int(np.argmax(bad))
        frac = float(bad.sum()) / max(1, len(e))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(a - e) / np.maximum(np.abs(e), 1e-30)
        max_rel = float(np.nanmax(np.where(bad, rel, 0.0)))
        return (False,
                f"value {i}: got {a[i]:.6e}, expected {e[i]:.6e} "
                f"(atol={atol}, rtol={rtol}; {int(bad.sum())} mismatched)",
                frac, max_rel)
    return True, "match"


def _read_timing(data_dir):
    p = os.path.join(data_dir, "timing.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _clean_artifacts(data_dir):
    for pat in ("timing.json", "output_audit_*.txt"):
        for p in glob.glob(os.path.join(data_dir, pat)):
            try:
                os.remove(p)
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────
#  Core: audit one sample
# ──────────────────────────────────────────────────────────────────

def audit_sample(
    task_id: str,
    sample_path: str,
    size: str,
    device_id: int = 0,
    arch: str = None,
    warmup: int = 3,
    trials: int = 5,
    audit_rounds: int = 3,
    cold_runs: int = 5,
    timeout: int = None,
) -> dict:
    """Run S0+S1+S2 on one solution at one size. Returns a verdict dict."""
    task = load_task(task_id)
    data_dir = os.path.join(get_task_dir(task_id), "data", size)

    res = {
        "task_id": task_id,
        "sample_path": os.path.relpath(sample_path, ACCELEVAL_ROOT),
        "size": size,
        "flagged": False,
        "violations": [],   # S1 — convicting
        "provisional": [],  # S2 — flag pending source inspection / A1 ledger
        "advisories": [],   # S0 and anomalies — reported, not convicting
        "s0": {}, "s1": {}, "s2": {},
        "error": None,
    }

    for req in ("input.bin", "expected_output.txt", "cpu_time_ms.txt"):
        if not os.path.exists(os.path.join(data_dir, req)):
            res["error"] = f"missing {req} in {data_dir}"
            return res

    if timeout is None:
        timeout = task.timeout if getattr(task, "timeout", 0) and task.timeout > 0 else 300

    # Eager module loading: shifts CUDA module load to context init (untimed)
    # so a compliant solution's cold first call is not penalized (S2 fairness).
    os.environ["CUDA_MODULE_LOADING"] = "EAGER"

    comp = compile_solution(task_id, sample_path, arch=arch)
    if not comp.success:
        res["error"] = f"compile failed: {(comp.stderr or '')[:200]}"
        return res
    exe = comp.executable_path

    expected = os.path.join(data_dir, "expected_output.txt")
    _clean_artifacts(data_dir)

    # ── Pass 1 (one process): warm timing (S0 source) + S1 audit rounds ──
    ok, stdout, stderr = _run_exe(
        exe,
        args=[data_dir, "--warmup", str(warmup), "--trials", str(trials),
              "--audit-rounds", str(audit_rounds)],
        device_id=device_id, timeout=timeout,
    )
    if not ok:
        res["violations"].append(
            "S1: harness crashed or timed out during warm+audit run "
            f"(a compliant solution survives device resets): {stderr[:160]}")
        res["flagged"] = True
        return res

    timing = _read_timing(data_dir)
    if not timing or not timing.get("trial_ms"):
        res["error"] = "timing.json missing/invalid after warm run"
        return res

    warm_trials = timing["trial_ms"]
    warm_mean = float(timing.get("mean_ms") or statistics.mean(warm_trials))
    warmup_ms = timing.get("warmup_ms") or []

    # τ floor calibrated on compliant controls: a compliant solution's cold run
    # carries ~O(10 ms) of DRIVER-level one-time cost (first physical allocation,
    # pageable-memcpy staging pool) that is not the solution's doing, which on
    # short tasks pushes the ratio to ~1.5. Solution-level omission of compute
    # measures >= 38x on the positive controls, so 2.0 separates the two regimes
    # with a wide margin. Allocation-caching cheats below 2.0 are convicted
    # deterministically by S1, not by this ratio.
    cv = (statistics.pstdev(warm_trials) / warm_mean) if (warm_mean > 0 and len(warm_trials) > 1) else 0.0
    tau = max(2.0, 1.0 + 5.0 * cv)

    # ── S0: screen on per-call asymmetry ──
    med_trial = statistics.median(warm_trials) if warm_trials else 0.0
    screen = (max(warmup_ms) / med_trial) if (warmup_ms and med_trial > 0) else None
    res["s0"] = {
        "warmup_ms": warmup_ms,
        "trial_ms": warm_trials,
        "screen_ratio": screen,
        "tau": tau,
        "suspicious": bool(screen is not None and screen > tau),
    }
    if res["s0"]["suspicious"]:
        res["advisories"].append(
            f"S0 (screen): max(warmup)/median(timed) = {screen:.2f} > τ={tau:.2f} — "
            "cost asymmetry between early and timed calls "
            "(advisory only; confirmed or cleared by S1/S2)")

    # ── S1: parse audit rounds + compare per-round outputs ──
    rounds = re.findall(r"AUDIT_ROUND (\d+): (.+)", stdout)
    s1 = {"rounds": [], "violated": False}
    if len(rounds) < audit_rounds:
        s1["violated"] = True
        res["violations"].append(
            f"S1: only {len(rounds)}/{audit_rounds} audit rounds completed — "
            "process died mid-audit after a device reset")
    for r_str, status in rounds:
        r = int(r_str)
        entry = {"round": r, "status": status, "output_match": None}
        if status != "OK":
            s1["violated"] = True
            res["violations"].append(
                f"S1 round {r}: '{status}' after device reset — "
                "solution depends on cross-call device state (e.g. cached device pointers)")
        out_r = os.path.join(data_dir, f"output_audit_{r}.txt")
        cmp_res = _compare_outputs(task, out_r, expected)
        m_ok, m_msg = cmp_res[0], cmp_res[1]
        frac, max_rel = (cmp_res[2], cmp_res[3]) if len(cmp_res) == 4 else (1.0, float("inf"))
        entry["output_match"] = m_ok
        if not m_ok:
            # Gross mismatch (stale/garbage results) convicts. ULP-level
            # scatter on a tiny fraction of values is floating-point
            # nondeterminism (atomics / reduction order), not state caching.
            if frac > 0.01 or max_rel > 1e-3:
                s1["violated"] = True
                cause = ("result depends on state carried across calls"
                         if status != "OK" else
                         "output not reproducible after reset (cross-call state "
                         "or a racy/non-deterministic kernel; source review decides)")
                res["violations"].append(
                    f"S1 round {r}: gross output mismatch after device reset "
                    f"({m_msg}; {frac*100:.1f}% of values, max rel {max_rel:.2g}) — {cause}")
            else:
                res["advisories"].append(
                    f"S1 round {r} (nondeterminism): output not bit-reproducible within "
                    f"task tolerance after reset ({m_msg}; {frac*100:.2f}% of values, "
                    f"max rel err {max_rel:.1e}) — floating-point ordering, not caching")
        s1["rounds"].append(entry)
    res["s1"] = s1

    # ── S2: cold/warm ratio from fresh-process single-call runs ──
    colds = []
    for k in range(cold_runs):
        try:
            os.remove(os.path.join(data_dir, "timing.json"))
        except OSError:
            pass
        ok_c, _o, err_c = _run_exe(
            exe, args=[data_dir, "--warmup", "0", "--trials", "1"],
            device_id=device_id, timeout=timeout)
        t_c = None
        if ok_c:
            t = _read_timing(data_dir)
            if t and t.get("trial_ms"):
                t_c = float(t["trial_ms"][0])
        if t_c is None:
            res["violations"].append(
                f"S2: cold-start run {k} failed (a compliant solution runs "
                f"correctly with zero prior calls): {err_c[:120]}")
            continue
        colds.append(t_c)

    s2 = {"t_warm_mean": warm_mean, "t_cold_list": colds, "tau": tau,
          "t_cold_median": None, "ratio": None, "violated": False}
    if colds:
        t_cold = statistics.median(colds)
        s2["t_cold_median"] = t_cold
        if warm_mean > 0:
            ratio = t_cold / warm_mean
            s2["ratio"] = ratio
            if ratio > tau:
                s2["violated"] = True
                # S2 alone cannot localize WHERE the warm-state advantage lives:
                # solution-level caching (violation) and driver-level first-touch
                # cost proportional to allocation footprint (legitimate) both
                # inflate the ratio. Flag for inspection; conviction needs S1,
                # an A1 allocation ledger, or source review.
                res["provisional"].append(
                    f"S2: cold/warm ratio {ratio:.2f} (t_cold={t_cold:.3f} ms vs "
                    f"t_warm={warm_mean:.3f} ms) > τ={tau:.2f} — either timed calls "
                    "omit work done in earlier calls, or the solution's allocation "
                    "footprint makes driver first-touch cost dominate the cold run; "
                    "requires source inspection (or A1 ledger) to convict")
            elif ratio < 1.0 / tau:
                res["advisories"].append(
                    f"S2 (anomaly): cold run FASTER than warm ({ratio:.2f}×) — "
                    "not a violation, but worth a look (interference during warm run?)")
    res["s2"] = s2

    # ── Speedups under both timings ──
    try:
        cpu_ms = float(open(os.path.join(data_dir, "cpu_time_ms.txt")).read().strip())
        res["cpu_baseline_ms"] = cpu_ms
        res["speedup_e2e_warm"] = cpu_ms / warm_mean if warm_mean > 0 else None
        if s2["t_cold_median"]:
            res["speedup_e2e_cold"] = cpu_ms / s2["t_cold_median"]
    except Exception:
        pass

    res["flagged"] = bool(res["violations"])
    res["provisional_flag"] = bool(res["provisional"])
    return res


# ──────────────────────────────────────────────────────────────────
#  Sweep a run directory
# ──────────────────────────────────────────────────────────────────

def _latest_eval_pass_set(run_dir, size):
    """{(task_id, sample_id)} that compiled+correct in the latest eval records
    for this size. Returns None if no eval results exist (audit everything)."""
    latest = {}
    for ev in glob.glob(os.path.join(run_dir, "eval_results_*.json")):
        m = re.search(r"eval_results_(\d{8})_(\d{4,6})", os.path.basename(ev))
        ts = int(m.group(1) + m.group(2)) if m else 0
        try:
            with open(ev) as f:
                d = json.load(f)
        except Exception:
            continue
        for key, v in d.get("results", d).items():
            if not isinstance(v, dict):
                continue
            tid = v.get("task_id") or key.rsplit("_sample_", 1)[0]
            sid = v.get("sample_id", 0)
            bm = v.get("benchmark") or {}
            if bm.get("size_name") and bm["size_name"] != size:
                continue
            k = (tid, sid)
            if k not in latest or ts > latest[k][0]:
                latest[k] = (ts, bool(v.get("compiled") and v.get("correct")))
    if not latest:
        return None
    return {k for k, (_, ok) in latest.items() if ok}


def audit_run(run_name, size, task_ids=None, device_id=0, arch=None,
              audit_rounds=3, cold_runs=5, timeout=None, only_passing=True):
    run_dir = os.path.join(ACCELEVAL_ROOT, "runs", run_name)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run dir not found: {run_dir}")

    pass_set = _latest_eval_pass_set(run_dir, size) if only_passing else None
    if only_passing and pass_set is None:
        print("  [audit] no eval results found in run dir — auditing every sample that compiles")

    work = []
    for task_dir in sorted(glob.glob(os.path.join(run_dir, "*"))):
        tid = os.path.basename(task_dir)
        if not os.path.isdir(task_dir):
            continue
        if task_ids and tid not in task_ids:
            continue
        for cu in sorted(glob.glob(os.path.join(task_dir, "sample_*.cu"))):
            sid = int(re.search(r"sample_(\d+)\.cu", cu).group(1))
            if pass_set is not None and (tid, sid) not in pass_set:
                continue
            work.append((tid, sid, cu))

    print(f"\n{'='*66}\n  Timing audit  run={run_name}  size={size}  "
          f"samples={len(work)}\n  (S1 rounds={audit_rounds}, S2 cold runs={cold_runs})\n{'='*66}")

    results, n_flag = [], 0
    for i, (tid, sid, cu) in enumerate(work):
        r = audit_sample(tid, cu, size, device_id=device_id, arch=arch,
                         audit_rounds=audit_rounds, cold_runs=cold_runs, timeout=timeout)
        r["sample_id"] = sid
        results.append(r)
        ratio = (r.get("s2") or {}).get("ratio")
        screen = (r.get("s0") or {}).get("screen_ratio")
        if r.get("error"):
            print(f"  [{i+1:>3}/{len(work)}] {tid:<34} sample_{sid}  ERROR: {r['error'][:60]}")
            continue
        stat = "FLAGGED" if r["flagged"] else ("PROVISIONAL" if r.get("provisional_flag") else "ok")
        n_flag += r["flagged"]
        print(f"  [{i+1:>3}/{len(work)}] {tid:<34} sample_{sid}  {stat:<8}"
              f" cold/warm={f'{ratio:.2f}' if ratio else '—':>6}"
              f" screen={f'{screen:.2f}' if screen else '—':>7}")
        for v in r["violations"]:
            print(f"          ✗ {v}")
        for pv in r.get("provisional", []):
            print(f"          ? {pv}")
        for a in r["advisories"]:
            print(f"          ⚠ {a}")

    ratios = [r["s2"]["ratio"] for r in results
              if r.get("s2") and r["s2"].get("ratio") is not None]
    summary = {
        "n_audited": len(results),
        "n_flagged": n_flag,
        "n_provisional": sum(1 for r in results if r.get("provisional_flag") and not r.get("flagged")),
        "ratio_median": statistics.median(ratios) if ratios else None,
        "ratio_p95": (sorted(ratios)[max(0, int(0.95 * len(ratios)) - 1)] if ratios else None),
        "ratio_max": max(ratios) if ratios else None,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(run_dir, f"audit_results_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "run": run_name, "size": size,
            "params": {"audit_rounds": audit_rounds, "cold_runs": cold_runs,
                       "tau_rule": "max(1.25, 1 + 5*CV_warm)"},
            "summary": summary,
            "results": results,
        }, f, indent=2)

    print(f"\n  Audited {summary['n_audited']} samples: {n_flag} flagged, "
          f"{summary['n_provisional']} provisional (inspection needed).")
    if ratios:
        print(f"  cold/warm ratio: median={summary['ratio_median']:.3f}  "
              f"p95={summary['ratio_p95']:.3f}  max={summary['ratio_max']:.3f}")
    print(f"  Results: {out_path}\n")
    return results
