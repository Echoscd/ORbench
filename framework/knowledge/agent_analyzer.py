"""
agent_analyzer.py — Unified CUDA Optimization Analyzer.

Single LLM call per sample:
  Phase 1: Summarize each auto-detected pattern (target, method, intensity)
  Phase 2: Discover new patterns (capped by discovery budget)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


# ---------------------------------------------------------------------------
#  Unified CUDA Optimization Analyzer
# ---------------------------------------------------------------------------

ANALYZER_SYSTEM = """\
You are a CUDA optimization analyst. You will receive:
1. CPU reference code (baseline)
2. LLM-generated CUDA code (optimized)
3. Auto-detected optimization patterns (from keyword matching)
4. Known pattern knowledge base (for deduplication)
5. A discovery budget: the MAXIMUM number of new patterns you may report

Your job has three phases:

━━━ Phase 1a: Summarize Auto-Detected Patterns ━━━
For EACH auto-detected pattern, produce a short concrete summary:
- "target": WHAT data/computation is optimized (be specific: name the array,
  the loop, the data structure)
- "method": HOW it is applied (quote variable names, sizes, template params)
- "code_evidence": copy 5-10 lines from the CUDA code that implement this
  optimization, adding inline `// <--` comments to mark the key lines
- "intensity_note": calibrate by the intensity count

━━━ Phase 1b: Check Non-Auto-Detectable Patterns ━━━
Some patterns CANNOT be detected by keyword matching (e.g., memory coalescing,
loop reordering, algebraic reformulation). These are listed under
"Patterns Requiring Manual Check" in the input.

For EACH of these patterns, read the CUDA code carefully and determine:
- Does this code use this technique? (yes/no)
- If yes, add it to pattern_summaries with the same format as Phase 1a,
  including "code_evidence" that quotes the specific lines proving it
- If no, skip it (do NOT include it)

Be strict: only report a pattern if you can point to specific code that
implements it. The "code_evidence" field is mandatory — no evidence, no report.

━━━ Phase 2: Discover New Patterns (budget = {max_new}) ━━━
Scan the CUDA code for techniques NOT already covered by Phase 1a/1b and NOT
a renamed duplicate of a knowledge-base entry. Prioritize by estimated
performance impact (highest first). You MUST return at most {max_new} items.
If nothing new exists, return an empty list.

Look for: algorithm restructuring, domain-specific math tricks, memory layout
changes, warp-level tricks, branch elimination, or any technique that
contributes to the speedup but was missed by keyword matching and Phase 1b.

Return 0 items if nothing qualifies. Never pad the list to fill the budget.

━━━ Output (strict JSON, no markdown fences, no trailing commas) ━━━
{{
  "pattern_summaries": [
    {{
      "pattern_id": "PAT-XXX",
      "pattern_name": "...",
      "target": "...",
      "method": "...",
      "code_evidence": "5-10 lines of the CUDA code showing this optimization, with inline // <-- comments marking the specific optimization technique",
      "intensity_note": "...",
      "source": "auto_detected | manual_check"
    }}
  ],
  "new_candidates": [
    {{
      "raw_description": "2-3 sentences",
      "mechanism_hypothesis": "1 sentence on WHY it helps",
      "estimated_impact": "high | medium | low",
      "code_snippet": "5-15 lines"
    }}
  ],
  "strategy_summary": "2-3 sentences: overall GPU strategy",
  "bottleneck_analysis": "1-2 sentences: remaining bottleneck"
}}
"""

ANALYZER_USER = """\
## CPU Reference
```c
{cpu_reference}
```

## CUDA Code
```cuda
{cuda_code}
```

## Auto-Detected Patterns (Phase 1a — summarize each)
{auto_matched_str}

## Patterns Requiring Manual Check (Phase 1b — read code to verify each)
{manual_check_str}

## Full Knowledge Base (Phase 2 dedup reference)
{knowledge_base_str}

Discovery budget: {max_new}
"""


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences and common errors."""
    if not text or not text.strip():
        raise ValueError("Empty response")

    text = text.strip()

    # Try markdown fence first (greedy — handles truncated closing fence)
    m = re.search(r"```(?:json)?\s*\n(.*?)(?:\n\s*```|$)", text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    # Extract { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    elif start != -1:
        # Truncated — try to close it
        text = text[start:] + "}"

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix common LLM JSON errors: trailing commas
    fixed = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Last resort: extract arrays/strings by key
    result = {}
    for key in ["pattern_summaries", "new_candidates"]:
        m = re.search(rf'"{key}"\s*:\s*(\[.*?\])', text, re.DOTALL)
        if m:
            try:
                result[key] = json.loads(m.group(1))
            except json.JSONDecodeError:
                result[key] = []

    for key in ["strategy_summary", "bottleneck_analysis"]:
        m = re.search(rf'"{key}"\s*:\s*"(.*?)"', text, re.DOTALL)
        if m:
            result[key] = m.group(1)

    if result:
        return result

    raise ValueError(f"Cannot parse JSON from response: {text[:300]}")


def _format_auto_matched(auto_matched: list[dict]) -> str:
    if not auto_matched:
        return "(none detected)"
    lines = []
    for m in auto_matched:
        intensity = m.get("intensity", "present")
        count = m.get("intensity_count", 0)
        label = m.get("intensity_label", "")
        count_str = f", count={count} ({label})" if label else ""
        lines.append(f"- [{m['pattern_id']}] {m['pattern_name']}: intensity={intensity}{count_str}")
    return "\n".join(lines)


def _load_sources(task_id: str, source_path: str):
    """Load CPU reference and CUDA source."""
    _root = Path(__file__).resolve().parents[2]
    ref_path = _root / "tasks" / task_id / "cpu_reference.c"
    cpu_ref = ref_path.read_text(encoding="utf-8", errors="ignore") if ref_path.exists() else "(not found)"

    with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
        cuda_code = f.read()

    return cpu_ref[:6000], cuda_code[:8000]


# ---------------------------------------------------------------------------
#  Main analysis function
# ---------------------------------------------------------------------------

def analyze_sample(
    task_id: str,
    source_path: str,
    auto_features: dict,
    ptxas_info: list[dict],
    benchmark: dict,
    knowledge_base,
    auto_matched: list[dict] = None,
    llm_client=None,
    agent_model_id: str = "gemini-3.1-pro-preview-openrouter",
    max_new: int = 1,
) -> dict:
    """
    Single LLM call: summarize auto-detected patterns + discover new ones.

    Args:
        auto_matched: list from KnowledgeBase.match_by_features()["matched"]
        max_new: discovery budget (max new candidates to report)

    Returns dict with pattern_summaries, new_candidates, strategy_summary, bottleneck_analysis.
    """
    cpu_ref, cuda_code = _load_sources(task_id, source_path)
    auto_matched_str = _format_auto_matched(auto_matched or [])

    # Build manual check list: non-auto-detectable patterns NOT already matched
    auto_matched_ids = {m["pattern_id"] for m in (auto_matched or [])}
    manual_lines = []
    for p in knowledge_base.all_patterns():
        if not p.auto_detectable and p.id not in auto_matched_ids and p.status in ("active", "seed"):
            hint = p.signature.representative_snippet[:80] if p.signature.representative_snippet else ""
            hint_str = f'  Example: `{hint}`' if hint else ""
            manual_lines.append(f"- [{p.id}] {p.name}: {p.mechanism}{hint_str}")
    manual_check_str = "\n".join(manual_lines) if manual_lines else "(none)"

    # Get LLM client
    if llm_client is None:
        from ..llm.registry import LLMRegistry
        registry = LLMRegistry()
        llm_client = registry.get_client(agent_model_id)

    # Build prompt
    system = ANALYZER_SYSTEM.format(max_new=max_new)
    user = ANALYZER_USER.format(
        cpu_reference=cpu_ref,
        cuda_code=cuda_code,
        auto_matched_str=auto_matched_str,
        manual_check_str=manual_check_str,
        knowledge_base_str=knowledge_base.summary_for_agent(),
        max_new=max_new,
    )

    full_prompt = f"<system>\n{system}\n</system>\n\n{user}"
    resp = llm_client.generate(full_prompt, max_tokens=16000, temperature=0.0)

    try:
        result = _parse_json(resp.content)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  [KB] Agent parse error: {e}")
        result = {
            "pattern_summaries": [],
            "new_candidates": [],
            "strategy_summary": "",
            "bottleneck_analysis": "",
            "parse_error": True,
            "raw_response": resp.content[:500],
        }

    return result
