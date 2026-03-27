"""
promotion.py — Logic for promoting staging candidates to confirmed patterns.
"""

from __future__ import annotations

import time
from .store import PatternEntry, CodeSignature


def try_promote_candidates(knowledge_base) -> list[str]:
    """
    Check staging candidates and promote those meeting criteria.
    Returns list of newly promoted pattern IDs.
    """
    promoted = []

    for cid, candidate in list(knowledge_base._staging.items()):
        decision = evaluate_candidate(candidate, knowledge_base)

        if decision["action"] == "promote":
            new_pattern = PatternEntry(
                id=knowledge_base._next_pattern_id(),
                name=decision.get("suggested_name", candidate.raw_description[:60]),
                mechanism=candidate.mechanism_hypothesis,
                description=candidate.raw_description,
                signature=CodeSignature(
                    representative_snippet=candidate.code_snippet,
                ),
                evidence=candidate.evidence,
                avg_speedup_lift=decision.get("speedup_lift", 0),
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                promoted_from_staging=True,
            )
            knowledge_base.add_pattern(new_pattern)
            del knowledge_base._staging[cid]
            promoted.append(new_pattern.id)
            print(f"  [KB] Promoted {cid} -> {new_pattern.id}: {new_pattern.name}")

        elif decision["action"] == "merge":
            target_id = decision["merge_into"]
            target = knowledge_base.get_pattern(target_id)
            if target:
                target.evidence.extend(candidate.evidence)
                del knowledge_base._staging[cid]
                print(f"  [KB] Merged {cid} into {target_id}")

    if promoted:
        knowledge_base.save()

    return promoted


def evaluate_candidate(candidate, knowledge_base) -> dict:
    """
    Evaluate whether a staging candidate should be promoted.

    Returns: {"action": "promote" | "merge" | "wait", ...}
    """
    evidence = candidate.evidence

    # Need at least 2 pieces of evidence
    if len(evidence) < 2:
        return {"action": "wait", "reason": "insufficient evidence"}

    unique_models = set(e.model_id for e in evidence)
    unique_tasks = set(e.task_id for e in evidence)

    multi_model = len(unique_models) >= 2
    multi_sample = len(evidence) >= 3

    if not (multi_model or multi_sample):
        return {"action": "wait", "reason": "need more independent observations"}

    # Speedup stats
    candidate_speedups = [e.speedup_e2e for e in evidence if e.speedup_e2e > 0]
    if len(candidate_speedups) < 2:
        return {"action": "wait", "reason": "insufficient speedup data"}

    avg_lift = sum(candidate_speedups) / len(candidate_speedups)

    # Check similarity with existing patterns
    best_similarity = 0.0
    best_match_id = ""

    for pid, pattern in knowledge_base._patterns.items():
        sim = _compute_similarity(candidate, pattern)
        if sim > best_similarity:
            best_similarity = sim
            best_match_id = pid

    if best_similarity > 0.8:
        return {
            "action": "merge",
            "merge_into": best_match_id,
            "similarity": best_similarity,
            "reason": f"too similar to {best_match_id}"
        }

    return {
        "action": "promote",
        "speedup_lift": avg_lift,
        "unique_models": len(unique_models),
        "unique_tasks": len(unique_tasks),
        "total_evidence": len(evidence),
        "suggested_name": candidate.raw_description[:60],
    }


def _compute_similarity(candidate, pattern) -> float:
    """Compute similarity between a candidate and existing pattern."""
    score = 0.0
    weights = 0.0

    # Code snippet token overlap
    if candidate.code_snippet and pattern.signature.representative_snippet:
        tokens_a = set(candidate.code_snippet.split())
        tokens_b = set(pattern.signature.representative_snippet.split())
        if tokens_a | tokens_b:
            jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
            score += jaccard * 3.0
            weights += 3.0

    # Description text overlap
    if candidate.raw_description and pattern.description:
        words_a = set(candidate.raw_description.lower().split())
        words_b = set(pattern.description.lower().split())
        if words_a | words_b:
            jaccard = len(words_a & words_b) / len(words_a | words_b)
            score += jaccard * 2.0
            weights += 2.0

    return score / weights if weights > 0 else 0.0
