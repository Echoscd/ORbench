"""
store.py — Knowledge Base storage layer.

Manages patterns (confirmed optimization techniques), staging candidates,
and raw observations. All data persisted as JSON/JSONL files.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------

@dataclass
class IntensitySpec:
    """How to measure depth-of-use for a pattern (shallow vs deep)."""
    # What to count in source code (regex pattern)
    count_pattern: str = ""
    # Thresholds: [low_max, mid_max] → low/medium/high
    # e.g. [1, 5] means: 1=low, 2-5=medium, 6+=high
    thresholds: list[int] = field(default_factory=lambda: [1, 5])
    # Human-readable label for what's being counted
    label: str = ""  # e.g. "number of __shared__ declarations"


@dataclass
class CodeSignature:
    """Identifiable features of a pattern in source code."""
    grep_indicators: list[str] = field(default_factory=list)
    grep_excludes: list[str] = field(default_factory=list)
    ptxas_conditions: dict = field(default_factory=dict)
    representative_snippet: str = ""
    anti_snippet: str = ""
    # How to quantify intensity (None = binary presence only)
    intensity: IntensitySpec = field(default_factory=IntensitySpec)


@dataclass
class Evidence:
    """One occurrence of a pattern in a specific sample."""
    task_id: str
    model_id: str
    run_name: str
    sample_id: int
    speedup_e2e: float
    speedup_kernel: float = 0.0
    kernel_time_ms: float = 0.0
    timestamp: str = ""


@dataclass
class PatternEntry:
    """A confirmed optimization pattern in the knowledge base."""
    id: str
    name: str
    mechanism: str
    description: str

    category: str = ""
    signature: CodeSignature = field(default_factory=CodeSignature)
    auto_detectable: bool = False

    evidence: list[Evidence] = field(default_factory=list)
    co_occurs_with: list[str] = field(default_factory=list)
    avg_speedup_lift: float = 0.0

    created_at: str = ""
    promoted_from_staging: bool = True
    status: str = "active"  # active / seed / deprecated / merged_into:PAT-xxx


@dataclass
class StagingCandidate:
    """A candidate pattern awaiting validation."""
    id: str
    raw_description: str
    mechanism_hypothesis: str
    code_snippet: str

    evidence: list[Evidence] = field(default_factory=list)
    auto_features: dict = field(default_factory=dict)
    ptxas_info: dict = field(default_factory=dict)

    created_at: str = ""
    most_similar_pattern: str = ""
    similarity_score: float = 0.0


@dataclass
class Observation:
    """Raw record of one analysis run."""
    task_id: str
    model_id: str
    run_name: str
    sample_id: int
    source_path: str

    auto_features: dict = field(default_factory=dict)
    ptxas_info: list[dict] = field(default_factory=list)

    matched_patterns: list[str] = field(default_factory=list)
    unmatched_speedup_ratio: float = 0.0

    agent_analysis: Optional[dict] = None
    new_candidates: list[str] = field(default_factory=list)

    speedup_e2e: float = 0.0
    speedup_kernel: float = 0.0
    kernel_time_ms: float = 0.0

    compiled: bool = False
    correct: bool = False

    timestamp: str = ""


# ---------------------------------------------------------------------------
#  Helper: strip C/C++ comments
# ---------------------------------------------------------------------------

def strip_comments(src: str) -> str:
    import re
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"//.*?$", "", src, flags=re.MULTILINE)
    return src


# ---------------------------------------------------------------------------
#  KnowledgeBase class
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """Read/write layer for the optimization knowledge base."""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            _root = Path(__file__).resolve().parents[2]
            data_dir = str(_root / "Library" / "knowledge_data")
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.patterns_file = os.path.join(data_dir, "patterns.json")
        self.staging_file = os.path.join(data_dir, "staging.jsonl")
        self.observations_file = os.path.join(data_dir, "observations.jsonl")

        self._patterns: dict[str, PatternEntry] = {}
        self._staging: dict[str, StagingCandidate] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.patterns_file):
            with open(self.patterns_file) as f:
                data = json.load(f)
            for entry in data.get("patterns", []):
                # Handle nested dataclasses
                if "signature" in entry and isinstance(entry["signature"], dict):
                    sig = entry["signature"]
                    if "intensity" in sig and isinstance(sig["intensity"], dict):
                        sig["intensity"] = IntensitySpec(**sig["intensity"])
                    entry["signature"] = CodeSignature(**sig)
                if "evidence" in entry:
                    entry["evidence"] = [
                        Evidence(**e) if isinstance(e, dict) else e
                        for e in entry["evidence"]
                    ]
                p = PatternEntry(**entry)
                self._patterns[p.id] = p

        if os.path.exists(self.staging_file):
            with open(self.staging_file) as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        if "evidence" in d:
                            d["evidence"] = [
                                Evidence(**e) if isinstance(e, dict) else e
                                for e in d["evidence"]
                            ]
                        c = StagingCandidate(**d)
                        self._staging[c.id] = c

    def save(self):
        with open(self.patterns_file, "w") as f:
            json.dump({
                "version": 1,
                "num_patterns": len(self._patterns),
                "patterns": [asdict(p) for p in self._patterns.values()]
            }, f, indent=2, ensure_ascii=False)

    def load_seed(self, seed_path: str):
        """Load seed patterns from seed_knowledge_base.json."""
        with open(seed_path) as f:
            data = json.load(f)
        for entry in data.get("patterns", []):
            if "signature" in entry and isinstance(entry["signature"], dict):
                sig = entry["signature"]
                if "intensity" in sig and isinstance(sig["intensity"], dict):
                    sig["intensity"] = IntensitySpec(**sig["intensity"])
                entry["signature"] = CodeSignature(**sig)
            else:
                entry["signature"] = CodeSignature()
            if "evidence" not in entry:
                entry["evidence"] = []
            # Filter to PatternEntry fields only
            valid_fields = set(PatternEntry.__dataclass_fields__.keys())
            filtered = {k: v for k, v in entry.items() if k in valid_fields}
            p = PatternEntry(**filtered)
            if p.id not in self._patterns:
                self._patterns[p.id] = p
        self.save()
        print(f"[KB] Loaded {len(data.get('patterns', []))} seed patterns")

    # ── Queries ──

    def get_pattern(self, pattern_id: str) -> Optional[PatternEntry]:
        return self._patterns.get(pattern_id)

    def all_patterns(self) -> list[PatternEntry]:
        return list(self._patterns.values())

    def num_patterns(self) -> int:
        return len(self._patterns)

    def num_staging(self) -> int:
        return len(self._staging)

    def summary_for_agent(self) -> str:
        """Compact summary of known patterns for LLM context."""
        lines = []
        for p in self._patterns.values():
            lines.append(
                f"[{p.id}] {p.name}: {p.mechanism} "
                f"(seen {len(p.evidence)} times, avg lift {p.avg_speedup_lift:.1f}x)"
            )
        return "\n".join(lines) if lines else "(knowledge base is empty)"

    # ── Writes ──

    def add_pattern(self, pattern: PatternEntry):
        self._patterns[pattern.id] = pattern
        self.save()

    def add_candidate(self, candidate: StagingCandidate):
        self._staging[candidate.id] = candidate
        with open(self.staging_file, "a") as f:
            f.write(json.dumps(asdict(candidate), ensure_ascii=False) + "\n")

    def add_evidence_to_pattern(self, pattern_id: str, evidence: Evidence):
        if pattern_id in self._patterns:
            self._patterns[pattern_id].evidence.append(evidence)
            self.save()

    def add_evidence_to_candidate(self, candidate_id: str, evidence: Evidence):
        if candidate_id in self._staging:
            self._staging[candidate_id].evidence.append(evidence)

    def record_observation(self, obs: Observation):
        with open(self.observations_file, "a") as f:
            f.write(json.dumps(asdict(obs), ensure_ascii=False) + "\n")

    def _next_pattern_id(self) -> str:
        existing = [int(p.id.split("-")[1]) for p in self._patterns.values()
                    if "-" in p.id and p.id.split("-")[1].isdigit()]
        next_num = max(existing, default=0) + 1
        return f"PAT-{next_num:03d}"

    def _next_candidate_id(self) -> str:
        existing = [int(c.id.split("-")[1]) for c in self._staging.values()
                    if "-" in c.id and c.id.split("-")[1].isdigit()]
        next_num = max(existing, default=0) + 1
        return f"CAND-{next_num:03d}"

    # ── Auto-matching ──

    def match_by_features(
        self,
        auto_features: dict,
        ptxas: dict,
        source_code: str = ""
    ) -> dict:
        """Match known patterns using auto-detected features."""
        matched = []
        src = strip_comments(source_code) if source_code else ""

        for pid, pattern in self._patterns.items():
            sig = pattern.signature

            # Check grep indicators
            if sig.grep_indicators:
                if not all(ind in src for ind in sig.grep_indicators):
                    continue
                if any(exc in src for exc in sig.grep_excludes):
                    continue
            elif not pattern.auto_detectable:
                continue

            # Check ptxas conditions
            ptxas_ok = True
            for key, expected in sig.ptxas_conditions.items():
                actual = ptxas.get(key)
                if actual is None:
                    continue
                if isinstance(expected, dict):
                    op = expected.get("op", "==")
                    val = expected.get("value")
                    if op == "==" and actual != val:
                        ptxas_ok = False
                    elif op == ">" and actual <= val:
                        ptxas_ok = False
                    elif op == "<" and actual >= val:
                        ptxas_ok = False
                elif actual != expected:
                    ptxas_ok = False

            if not ptxas_ok:
                continue

            confidence = "high" if pattern.auto_detectable else "medium"

            # Compute intensity level if spec is defined
            intensity_info = self._compute_intensity(sig.intensity, src)

            matched.append({
                "pattern_id": pid,
                "pattern_name": pattern.name,
                "confidence": confidence,
                "method": "auto_detect",
                **intensity_info,
            })

        return {
            "matched": matched,
            "num_matched": len(matched),
            "coverage": len(matched) / max(len(self._patterns), 1)
        }

    @staticmethod
    def _compute_intensity(spec: IntensitySpec, src: str) -> dict:
        """Compute intensity level for a matched pattern."""
        import re
        if not spec.count_pattern or not src:
            return {"intensity": "present", "intensity_count": 0}

        count = len(re.findall(spec.count_pattern, src))
        thresholds = spec.thresholds or [1, 5]

        if count <= thresholds[0]:
            level = "low"
        elif len(thresholds) > 1 and count <= thresholds[1]:
            level = "medium"
        else:
            level = "high"

        return {
            "intensity": level,
            "intensity_count": count,
            "intensity_label": spec.label,
        }
