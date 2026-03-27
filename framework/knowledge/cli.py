#!/usr/bin/env python3
"""
cli.py — Knowledge Base command-line interface.

Usage:
  python -m framework.knowledge.cli status
  python -m framework.knowledge.cli seed
  python -m framework.knowledge.cli analyze-run <run_dir> [--agent-model MODEL]
  python -m framework.knowledge.cli promote
  python -m framework.knowledge.cli export [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

# Load .env for API keys
_env_path = _ORBENCH_ROOT / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                value = value.strip().strip("'\"")
                os.environ.setdefault(key.strip(), value)

from framework.knowledge.store import KnowledgeBase
from framework.knowledge.promotion import try_promote_candidates


def cmd_status(args):
    """Show knowledge base status."""
    kb = KnowledgeBase()
    print(f"Knowledge Base Status")
    print(f"  Data dir:   {kb.data_dir}")
    print(f"  Patterns:   {kb.num_patterns()}")
    print(f"  Staging:    {kb.num_staging()}")

    # Count observations
    obs_count = 0
    if os.path.exists(kb.observations_file):
        with open(kb.observations_file) as f:
            obs_count = sum(1 for line in f if line.strip())
    print(f"  Observations: {obs_count}")

    if kb.num_patterns() > 0:
        print(f"\nPatterns (top 20):")
        patterns = sorted(kb.all_patterns(), key=lambda p: len(p.evidence), reverse=True)
        for p in patterns[:20]:
            status = f" [{p.status}]" if p.status != "active" else ""
            print(f"  [{p.id}] {p.name}{status} "
                  f"({len(p.evidence)} evidence, lift={p.avg_speedup_lift:.1f}x)")

    if kb.num_staging() > 0:
        print(f"\nStaging candidates:")
        for cid, c in kb._staging.items():
            print(f"  [{cid}] {c.raw_description[:70]}... "
                  f"({len(c.evidence)} evidence)")


def cmd_seed(args):
    """Load seed patterns from seed_knowledge_base.json."""
    kb = KnowledgeBase()
    seed_path = _ORBENCH_ROOT / "Library" / "seed_knowledge_base.json"
    if not seed_path.exists():
        print(f"ERROR: Seed file not found: {seed_path}")
        sys.exit(1)
    kb.load_seed(str(seed_path))
    print(f"Knowledge base now has {kb.num_patterns()} patterns")


def cmd_analyze_run(args):
    """Analyze all passing samples in a run directory."""
    from framework.knowledge.integration import analyze_run

    run_dir = args.run_dir
    if not os.path.isabs(run_dir):
        run_dir = str(_ORBENCH_ROOT / "runs" / run_dir)

    if not os.path.isdir(run_dir):
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    kb = KnowledgeBase()
    analyze_run(
        run_dir,
        knowledge_base=kb,
        enable_agent=not args.no_agent,
        agent_model_id=args.agent_model,
    )


def cmd_promote(args):
    """Manually trigger promotion check."""
    kb = KnowledgeBase()
    promoted = try_promote_candidates(kb)
    if promoted:
        print(f"Promoted {len(promoted)} patterns: {promoted}")
    else:
        print("No candidates ready for promotion")


def cmd_diff_analyze(args):
    """Analyze pairwise diffs for a run using its kb_analysis.json."""
    from framework.knowledge.diff_analysis import analyze_diffs_for_run

    run_dir = args.run_dir
    if not os.path.isabs(run_dir):
        run_dir = str(_ORBENCH_ROOT / "runs" / run_dir)

    kb_analysis = os.path.join(run_dir, "kb_analysis.json")
    if not os.path.exists(kb_analysis):
        print(f"ERROR: kb_analysis.json not found in {run_dir}")
        print(f"  Run 'analyze-run' first to generate per-sample analysis.")
        sys.exit(1)

    kb = KnowledgeBase()
    diffs = analyze_diffs_for_run(
        kb_analysis_path=kb_analysis,
        knowledge_base=kb,
        enable_agent=not args.no_agent,
        agent_model_id=args.agent_model,
    )

    print(f"\n=== Diff Summary ===")
    for d in diffs:
        n_changes = len(d.pattern_changes)
        print(f"  {d.diff_id}: {d.task_id} {d.version_a_id}->{d.version_b_id} "
              f"{d.direction} ({d.speedup_ratio:.2f}x) {n_changes} changes")
        if d.agent_summary:
            print(f"    {d.agent_summary[:150]}")


def cmd_diff_status(args):
    """Show summary of all collected diffs."""
    diffs_dir = _ORBENCH_ROOT / "Library" / "knowledge_data" / "diffs"
    if not diffs_dir.exists():
        print("No diffs collected yet.")
        return

    total = 0
    improvements = 0
    regressions = 0
    for f in sorted(diffs_dir.glob("*.diffs.jsonl")):
        run_name = f.stem.replace(".diffs", "")
        count = 0
        imp = 0
        reg = 0
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                count += 1
                if d.get("direction") == "improvement":
                    imp += 1
                elif d.get("direction") == "regression":
                    reg += 1
        total += count
        improvements += imp
        regressions += reg
        print(f"  {run_name}: {count} diffs ({imp} improvements, {reg} regressions)")

    print(f"\nTotal: {total} diffs ({improvements} improvements, {regressions} regressions)")


def cmd_export(args):
    """Export knowledge base to readable JSON."""
    kb = KnowledgeBase()
    output = {
        "num_patterns": kb.num_patterns(),
        "num_staging": kb.num_staging(),
        "patterns": [],
    }
    for p in kb.all_patterns():
        output["patterns"].append({
            "id": p.id,
            "name": p.name,
            "mechanism": p.mechanism,
            "description": p.description,
            "category": p.category,
            "auto_detectable": p.auto_detectable,
            "status": p.status,
            "evidence_count": len(p.evidence),
            "avg_speedup_lift": p.avg_speedup_lift,
            "co_occurs_with": p.co_occurs_with,
            "tasks_seen_in": list(set(e.task_id for e in p.evidence)),
            "models_seen_in": list(set(e.model_id for e in p.evidence)),
        })

    out_path = args.output or str(_ORBENCH_ROOT / "Library" / "knowledge_export.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Exported to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="ORBench Knowledge Base CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show KB status")
    sub.add_parser("seed", help="Load seed patterns from seed_knowledge_base.json")

    p_analyze = sub.add_parser("analyze-run", help="Analyze a completed run")
    p_analyze.add_argument("run_dir", help="Run directory name or path")
    p_analyze.add_argument("--agent-model", default="gemini-3.1-pro-preview-openrouter",
                          help="Model for agent analysis")
    p_analyze.add_argument("--no-agent", action="store_true",
                          help="Skip LLM agent analysis, only use auto-detection")

    sub.add_parser("promote", help="Check and promote candidates")

    p_diff = sub.add_parser("diff-analyze", help="Pairwise diff analysis for a run")
    p_diff.add_argument("run_dir", help="Run directory (must have kb_analysis.json)")
    p_diff.add_argument("--agent-model", default="gemini-3.1-pro-preview-openrouter",
                        help="Model for diff agent")
    p_diff.add_argument("--no-agent", action="store_true",
                        help="Skip LLM agent, only record deltas")

    sub.add_parser("diff-status", help="Show pairwise diff summary")

    p_export = sub.add_parser("export", help="Export KB")
    p_export.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args)
    elif args.command == "seed":
        cmd_seed(args)
    elif args.command == "analyze-run":
        cmd_analyze_run(args)
    elif args.command == "promote":
        cmd_promote(args)
    elif args.command == "diff-analyze":
        cmd_diff_analyze(args)
    elif args.command == "diff-status":
        cmd_diff_status(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
