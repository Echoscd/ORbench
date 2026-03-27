"""
integration.py — Integration with the eval pipeline.

Called after each eval_single_sample to analyze the result,
match patterns, optionally trigger LLM deep analysis, and record observations.
"""

from __future__ import annotations

import time
import json
import os
from dataclasses import asdict

from .store import KnowledgeBase, Observation, Evidence, StagingCandidate
from .auto_detect import extract_auto_features, extract_ptxas_info, ptxas_summary
from .agent_analyzer import analyze_sample

# Trigger LLM deep analysis when:
AGENT_TRIGGER_SPEEDUP = 5.0        # speedup > 5x
AGENT_TRIGGER_COVERAGE = 0.6       # known pattern coverage < 60%

# Check promotion every N observations
PROMOTE_CHECK_INTERVAL = 10
_observation_counter = 0


def analyze_eval_result(
    eval_result,
    source_path: str,
    task_id: str,
    model_id: str,
    run_name: str,
    sample_id: int,
    compile_stderr: str = "",
    knowledge_base: KnowledgeBase = None,
    enable_agent: bool = True,
    agent_model_id: str = "gemini-3.1-pro-preview-openrouter",
) -> Observation:
    """
    Full knowledge base analysis for one eval result.

    Steps:
    1. Extract auto features from source code
    2. Match against known patterns
    3. Optionally trigger LLM deep analysis
    4. Record observation
    5. Periodically check candidate promotion
    """
    global _observation_counter

    if knowledge_base is None:
        knowledge_base = KnowledgeBase()

    benchmark = eval_result.benchmark or {}
    speedup = benchmark.get("speedup_e2e", 0)

    # Step 1: Auto feature extraction
    auto_features = {}
    if os.path.exists(source_path):
        auto_features = extract_auto_features(source_path)

    ptxas_info = extract_ptxas_info(compile_stderr) if compile_stderr else []
    ptxas_sum = ptxas_summary(ptxas_info)

    # Step 2: Match known patterns
    source_code = ""
    if os.path.exists(source_path):
        with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
            source_code = f.read()

    match_result = knowledge_base.match_by_features(
        auto_features, ptxas_sum, source_code
    )

    # Step 3: Build observation
    obs = Observation(
        task_id=task_id,
        model_id=model_id,
        run_name=run_name,
        sample_id=sample_id,
        source_path=source_path,
        auto_features=auto_features,
        ptxas_info=ptxas_info,
        matched_patterns=[m["pattern_id"] for m in match_result["matched"]],
        speedup_e2e=speedup,
        speedup_kernel=benchmark.get("speedup_kernel", 0),
        kernel_time_ms=benchmark.get("kernel_time_ms", 0),
        compiled=bool(eval_result.compiled),
        correct=bool(eval_result.correct),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    # Step 4: Decide whether to trigger LLM deep analysis
    # Always trigger if agent is enabled and sample is correct with meaningful speedup
    should_trigger_agent = (
        enable_agent
        and eval_result.correct
        and speedup > AGENT_TRIGGER_SPEEDUP
    )

    if should_trigger_agent:
        auto_matched = match_result["matched"]
        print(f"  [KB] Triggering agent analysis: speedup={speedup:.1f}x, "
              f"auto-matched={[m['pattern_id'] for m in auto_matched]}")

        try:
            agent_result = analyze_sample(
                task_id=task_id,
                source_path=source_path,
                auto_features=auto_features,
                ptxas_info=ptxas_info,
                benchmark=benchmark,
                knowledge_base=knowledge_base,
                auto_matched=auto_matched,
                agent_model_id=agent_model_id,
            )

            obs.agent_analysis = agent_result

            # Add evidence to auto-matched patterns (with agent summaries)
            pattern_summaries = {
                ps.get("pattern_id"): ps
                for ps in agent_result.get("pattern_summaries", [])
            }
            for m in auto_matched:
                pid = m["pattern_id"]
                if knowledge_base.get_pattern(pid):
                    knowledge_base.add_evidence_to_pattern(pid, Evidence(
                        task_id=task_id,
                        model_id=model_id,
                        run_name=run_name,
                        sample_id=sample_id,
                        speedup_e2e=speedup,
                        speedup_kernel=benchmark.get("speedup_kernel", 0),
                        kernel_time_ms=benchmark.get("kernel_time_ms", 0),
                        timestamp=obs.timestamp,
                    ))

            # Process new candidates
            for new_cand in agent_result.get("new_candidates", []):
                cand = StagingCandidate(
                    id=knowledge_base._next_candidate_id(),
                    raw_description=new_cand.get("raw_description", ""),
                    mechanism_hypothesis=new_cand.get("mechanism_hypothesis", ""),
                    code_snippet=new_cand.get("code_snippet", ""),
                    evidence=[Evidence(
                        task_id=task_id,
                        model_id=model_id,
                        run_name=run_name,
                        sample_id=sample_id,
                        speedup_e2e=speedup,
                        timestamp=obs.timestamp,
                    )],
                    auto_features=auto_features,
                    ptxas_info=ptxas_sum,
                    created_at=obs.timestamp,
                )

                merged = _try_merge_with_existing_candidate(cand, knowledge_base)
                if not merged:
                    knowledge_base.add_candidate(cand)
                    obs.new_candidates.append(cand.id)
                    print(f"  [KB] New candidate: {cand.id} - {cand.raw_description[:80]}")

        except Exception as e:
            print(f"  [KB] Agent analysis failed: {e}")

    # Step 5: Record observation
    knowledge_base.record_observation(obs)

    # Step 6: Periodic promotion check
    _observation_counter += 1
    if _observation_counter % PROMOTE_CHECK_INTERVAL == 0:
        from .promotion import try_promote_candidates
        promoted = try_promote_candidates(knowledge_base)
        if promoted:
            print(f"  [KB] Promoted {len(promoted)} patterns: {promoted}")

    return obs


def _try_merge_with_existing_candidate(
    new_cand: StagingCandidate,
    knowledge_base: KnowledgeBase
) -> bool:
    """Check if new candidate duplicates an existing staging candidate."""
    for existing_id, existing in knowledge_base._staging.items():
        new_tokens = set(new_cand.raw_description.lower().split())
        exist_tokens = set(existing.raw_description.lower().split())

        if not (new_tokens | exist_tokens):
            continue

        overlap = len(new_tokens & exist_tokens) / len(new_tokens | exist_tokens)

        if overlap > 0.5:
            for ev in new_cand.evidence:
                existing.evidence.append(ev)
            print(f"  [KB] Merged new candidate into existing {existing_id}")
            return True

    return False


def analyze_run(
    run_dir: str,
    knowledge_base: KnowledgeBase = None,
    enable_agent: bool = True,
    agent_model_id: str = "gemini-3.1-pro-preview-openrouter",
):
    """
    Analyze all samples in a completed run directory.

    Scans run_dir/<task>/agent_multiturn_summary.json for eval results
    and runs KB analysis on each passing sample.
    """
    if knowledge_base is None:
        knowledge_base = KnowledgeBase()

    run_name = os.path.basename(run_dir)
    # Extract model_id from run_name (e.g., "kimi-k2.5-openrouter_l2_agent_mt_20260323_1922")
    parts = run_name.split("_l")
    model_id = parts[0] if parts else run_name

    analyzed = 0
    skipped = 0
    run_observations = []  # Collect per-task results for the run report

    for task_dir in sorted(os.listdir(run_dir)):
        task_path = os.path.join(run_dir, task_dir)
        if not os.path.isdir(task_path):
            continue

        summary_path = os.path.join(task_path, "agent_multiturn_summary.json")
        if not os.path.exists(summary_path):
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        for rec in summary.get("records", []):
            ev = rec.get("eval_result", {})
            source_path = rec.get("source_path", "")

            if not source_path or not os.path.exists(source_path):
                skipped += 1
                run_observations.append({
                    "task_id": task_dir,
                    "sample_id": rec.get("sample_id", 0),
                    "status": "skipped",
                    "reason": "no source" if not source_path else "file missing",
                })
                continue

            if not ev.get("compiled") or not ev.get("correct"):
                skipped += 1
                run_observations.append({
                    "task_id": task_dir,
                    "sample_id": rec.get("sample_id", 0),
                    "status": "skipped",
                    "compiled": ev.get("compiled", False),
                    "correct": ev.get("correct", False),
                    "reason": "compile_fail" if not ev.get("compiled") else "incorrect",
                })
                continue

            # Create a minimal EvalResult-like object
            class _EvalLike:
                pass
            er = _EvalLike()
            er.compiled = ev.get("compiled", False)
            er.correct = ev.get("correct", False)
            er.benchmark = ev.get("benchmark")
            er.compile_error = ev.get("compile_error", "")

            sample_id = rec.get("sample_id", 0)
            print(f"[KB] Analyzing {task_dir} sample={sample_id}...")

            try:
                obs = analyze_eval_result(
                    eval_result=er,
                    source_path=source_path,
                    task_id=task_dir,
                    model_id=model_id,
                    run_name=run_name,
                    sample_id=sample_id,
                    compile_stderr=er.compile_error,
                    knowledge_base=knowledge_base,
                    enable_agent=enable_agent,
                    agent_model_id=agent_model_id,
                )
                matched = obs.matched_patterns
                speedup = obs.speedup_e2e
                print(f"  matched={matched} speedup={speedup:.1f}x")
                analyzed += 1

                agent = obs.agent_analysis or {}
                run_observations.append({
                    "task_id": task_dir,
                    "sample_id": sample_id,
                    "status": "analyzed",
                    "compiled": True,
                    "correct": True,
                    "speedup_e2e": obs.speedup_e2e,
                    "speedup_kernel": obs.speedup_kernel,
                    "kernel_time_ms": obs.kernel_time_ms,
                    "matched_patterns": agent.get("pattern_summaries", [
                        {"pattern_id": pid, "pattern_name": knowledge_base.get_pattern(pid).name if knowledge_base.get_pattern(pid) else pid}
                        for pid in obs.matched_patterns
                    ]),
                    "new_candidates": agent.get("new_candidates", []),
                    "strategy_summary": agent.get("strategy_summary", ""),
                    "bottleneck_analysis": agent.get("bottleneck_analysis", ""),
                    "auto_features": {k: v for k, v in obs.auto_features.items() if v},
                })
            except Exception as e:
                print(f"  FAILED: {e}")
                skipped += 1
                run_observations.append({
                    "task_id": task_dir,
                    "sample_id": rec.get("sample_id", 0),
                    "status": "error",
                    "error": str(e)[:200],
                })

    # Save run-level analysis report to the run directory
    report = {
        "run_name": run_name,
        "model_id": model_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analyzed": analyzed,
        "skipped": skipped,
        "total_patterns_in_kb": knowledge_base.num_patterns(),
        "total_staging_in_kb": knowledge_base.num_staging(),
        "tasks": run_observations,
    }
    report_path = os.path.join(run_dir, "kb_analysis.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[KB] Report saved: {report_path}")

    print(f"[KB] Run analysis complete: analyzed={analyzed} skipped={skipped}")
    print(f"[KB] Patterns: {knowledge_base.num_patterns()}, Staging: {knowledge_base.num_staging()}")

    return knowledge_base
