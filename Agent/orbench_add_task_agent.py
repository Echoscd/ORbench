#!/usr/bin/env python3
"""
orbench_add_task_agent.py — ORBench Auto-Add-Task Agent Pipeline

Two-stage pipeline:
  Agent 1 (Feasibility Checker):  Reads papers/code from a folder, determines
      if the OR problem is suitable for GPU acceleration in ORBench.
  Agent 2 (Task Assembler):  If suitable, generates all task folder artifacts.

Usage:
  # Process a single paper folder
  python orbench_add_task_agent.py Paper/paper1/ --model gemini-3.1-pro-preview

  # Process all subfolders under Paper/
  python orbench_add_task_agent.py Paper/ --model gemini-3.1-pro-preview --batch

  # Process individual PDF files directly
  python orbench_add_task_agent.py Paper/ORpaper1.pdf --model gemini-3.1-pro-preview
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
#  Ensure ORBench framework is importable
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_ORBENCH_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ORBENCH_ROOT))

# Load .env file if present (for API keys)
_env_file = _ORBENCH_ROOT / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                _k, _v = _k.strip(), _v.strip().strip("'\"")
                if _k and _k not in os.environ:
                    os.environ[_k] = _v

from framework.llm.registry import LLMRegistry
from framework.llm.base import LLMResponse

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

SUPPORTED_CODE_EXTS = {
    ".c", ".cpp", ".cu", ".h", ".hpp", ".py", ".jl", ".m", ".java", ".rs",
}
SUPPORTED_DOC_EXTS = {".pdf", ".md", ".txt", ".tex"}

MAX_FILE_CHARS = 50_000     # ~12K tokens, safe for most models
MAX_TOTAL_CHARS = 100_000

# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeasibilityResult:
    """Output of Agent 1."""
    suitability_score: int = 5            # 1-10 score (see AGENT1_SYSTEM for rubric)
    task_id: str = ""
    task_name: str = ""
    category: str = ""
    problem_summary: str = ""
    parallelism_analysis: str = ""
    scale_analysis: str = ""
    concerns: list[str] = field(default_factory=list)  # issues that lower the score
    gpu_optimization_points: list[str] = field(default_factory=list)
    suggested_sizes: dict = field(default_factory=dict)
    interface_mode: str = ""  # "compute_only" or "init_compute", determined by Agent 1
    difficulty: int = 2
    tags: list[str] = field(default_factory=list)
    algorithm_description: str = ""
    input_data_description: str = ""
    output_data_description: str = ""
    reference_code_snippet: str = ""


@dataclass
class TaskFiles:
    """Output of Agent 2 — all files needed under tasks/{task_id}/."""
    task_json: str
    prompt_template_yaml: str
    cpu_reference_c: str
    gen_data_py: str
    task_io_cu: str
    task_io_cpu_c: str


# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------

class AgentLogger:
    """Logs all LLM interactions to a log file with full input/output."""

    def __init__(self, log_dir: Path, paper_name: str):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"{paper_name}_{ts}.log"
        self._f = open(self.log_path, "w", encoding="utf-8")
        self._write(f"=== ORBench Add-Task Agent Log ===")
        self._write(f"Paper: {paper_name}")
        self._write(f"Time: {ts}")
        self._write("")

    def _write(self, msg: str):
        self._f.write(msg + "\n")
        self._f.flush()

    def log_agent_start(self, agent_name: str, model_id: str):
        self._write(f"\n{'='*80}")
        self._write(f"  {agent_name} — model: {model_id}")
        self._write(f"  Time: {datetime.now().isoformat()}")
        self._write(f"{'='*80}\n")

    def log_prompt(self, prompt: str):
        self._write(f"--- PROMPT ({len(prompt)} chars) ---")
        self._write(prompt)
        self._write(f"--- END PROMPT ---\n")

    def log_response(self, resp: LLMResponse):
        self._write(f"--- RESPONSE ---")
        self._write(f"  Model: {resp.model}")
        self._write(f"  Input tokens: {resp.input_tokens}")
        self._write(f"  Output tokens: {resp.output_tokens}")
        self._write(f"  Latency: {resp.latency_ms:.0f} ms")
        self._write(f"  Cost: ${resp.cost_usd:.4f}")
        self._write(f"--- CONTENT ({len(resp.content)} chars) ---")
        self._write(resp.content)
        self._write(f"--- END RESPONSE ---\n")

    def log_error(self, msg: str):
        self._write(f"[ERROR] {msg}\n")

    def log_info(self, msg: str):
        self._write(f"[INFO] {msg}")

    def log_result(self, agent_name: str, result_dict: dict):
        self._write(f"\n--- {agent_name} RESULT ---")
        self._write(json.dumps(result_dict, indent=2, ensure_ascii=False, default=str))
        self._write(f"--- END RESULT ---\n")

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
#  File ingestion
# ---------------------------------------------------------------------------

def read_folder(folder: str) -> dict[str, str]:
    """
    Read all relevant files from a folder (or a single file).
    Returns {relative_path: content_string}.
    """
    folder = Path(folder)

    if folder.is_file():
        # Single file mode
        try:
            if folder.suffix == ".pdf":
                content = _extract_pdf_text(folder)
            else:
                content = folder.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            content = f"[Error reading file: {e}]"
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + "\n\n... [TRUNCATED] ..."
        return {folder.name: content}

    if not folder.is_dir():
        raise FileNotFoundError(f"Input path not found: {folder}")

    files: dict[str, str] = {}
    total_chars = 0

    candidates = []
    for ext in SUPPORTED_CODE_EXTS | SUPPORTED_DOC_EXTS:
        candidates.extend(folder.rglob(f"*{ext}"))
    candidates.sort(key=lambda p: (0 if p.suffix == ".pdf" else 1, p.name))

    for path in candidates:
        if total_chars >= MAX_TOTAL_CHARS:
            break
        rel = str(path.relative_to(folder))
        try:
            if path.suffix == ".pdf":
                content = _extract_pdf_text(path)
            else:
                content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            content = f"[Error reading file: {e}]"

        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + "\n\n... [TRUNCATED] ..."

        files[rel] = content
        total_chars += len(content)

    return files


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using pymupdf (fitz)."""
    try:
        import fitz  # pymupdf
    except ImportError:
        return "[pymupdf not installed — cannot extract PDF text. pip install pymupdf]"

    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append(f"--- Page {i+1} ---\n{text}")
    doc.close()
    return "\n".join(pages)


def format_files_for_prompt(files: dict[str, str]) -> str:
    parts = []
    for rel_path, content in files.items():
        parts.append(f"<file path=\"{rel_path}\">\n{content}\n</file>")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
#  Agent 1 — Feasibility Checker
# ---------------------------------------------------------------------------

AGENT1_SYSTEM = textwrap.dedent("""\
You are an expert in Operations Research (OR), GPU/CUDA programming, and benchmark design.

Your job: Given a set of papers and code files describing an OR problem or algorithm,
SCORE this problem's suitability for inclusion in ORBench on a 1-10 scale.

## What is ORBench?

ORBench is a benchmark that tests LLMs' ability to generate CUDA code that accelerates
OR computations. Think of it as a **code generation test corpus** — we want diverse,
representative OR problems where an LLM must translate CPU logic into efficient GPU code.

The primary use case: **OR researchers run simulations** (Monte Carlo, DP sweeps,
batch evaluations, parameter scans). These simulations are CPU-bottlenecked.
ORBench tests whether LLMs can automatically GPU-accelerate such workloads.

A problem does NOT need to have "practical" GPU speedup in production to be useful
in ORBench. Even modest parallelism is fine — what matters is that the problem is
a **meaningful code generation challenge** that exercises CUDA skills (memory management,
kernel design, synchronization, etc.) in an OR context.

## Scoring Rubric (1-10)

Rate each dimension, then give an overall score:

**Parallelism (0-7 points)** — 70% weight:
- 0: Purely sequential, no way to parallelize (e.g., simplex pivots)
- 1: Minimal parallelism (e.g., one small parallel inner loop)
- 2: Minor parallelism (e.g., parallel inner loop, but outer loop is sequential)
- 3: Some parallelism (e.g., independent iterations in a batch)
- 4: Moderate parallelism (e.g., batch independent simulations, layer-parallel DP)
- 5: Good parallelism (e.g., per-state DP, batch Monte Carlo)
- 6: Strong parallelism (e.g., per-edge relaxation, large-scale element-wise ops)
- 7: Massive parallelism (e.g., millions of independent work items, embarrassingly parallel)

**Code Generation Challenge (0-3 points)** — 30% weight:
- 0: Trivial CUDA translation (just memcpy + one kernel)
- 1: Straightforward but requires proper memory management and kernel design
- 2: Requires non-trivial optimization (shared memory, synchronization, reductions)
- 3: Complex optimization landscape (multiple kernels, tiling, warp-level ops, persistent kernels)

**Overall score** = Parallelism + Code Generation Challenge (max 10).

**Score interpretation**:
- 1-3: Poor fit — skip unless forced
- 4-5: Marginal — usable but not ideal
- 6-7: Good fit — solid ORBench candidate
- 8-10: Excellent fit — high priority

## Important Notes
- A paper may contain MULTIPLE algorithms or problems. Pick the ONE most suitable.
- If the main algorithm is sequential but a sub-routine is parallel, focus on that sub-routine.
- Simulations (Monte Carlo, batch DP, parameter sweeps) are ALWAYS worth considering —
  even if each instance is small, batching many instances creates GPU parallelism.
- Non-deterministic algorithms can be made deterministic by fixing seeds or using
  pseudo-random generators with known state — don't reject just for randomness.
- **You do NOT need to use the exact algorithm proposed in the paper.** The paper defines
  a problem — you can choose ANY algorithm to solve it as the ORBench task. For example:
  - If the paper proposes an approximate/heuristic method for a DP problem, the underlying
    exact DP formulation itself can be a great GPU acceleration task.
  - If the paper uses a sequential solver but the problem has a natural parallel formulation
    (e.g., value iteration, policy evaluation, simulation-based optimization), use that instead.
  - The goal is to find a computationally intensive, parallelizable formulation of the
    problem described in the paper — not necessarily the paper's proposed solution method.

## Output Format

Respond with a JSON object (no markdown fences, just raw JSON) with these fields:
{
  "suitability_score": 7,
  "task_id": "snake_case_name",
  "task_name": "Human-Readable Name",
  "category": "category_name",
  "problem_summary": "One paragraph summary of the problem",
  "parallelism_analysis": "Analysis of available parallelism and score justification",
  "scale_analysis": "Analysis of problem scale and computational intensity",
  "concerns": ["concern1", "concern2"],
  "gpu_optimization_points": ["point1", "point2"],
  "suggested_sizes": {
    "small":  {"param1": 100, "param2": 500},
    "medium": {"param1": 10000, "param2": 50000},
    "large":  {"param1": 100000, "param2": 500000}
  },
  "interface_mode": "compute_only or init_compute (choose based on whether the problem has multi-query requests)",
  "difficulty": 2,
  "tags": ["tag1", "tag2"],
  "algorithm_description": "Detailed description of the algorithm for CPU implementation",
  "input_data_description": "Description of input data format and how to generate it",
  "output_data_description": "Description of output data format",
  "reference_code_snippet": "Key CPU code from the paper if available"
}
""")


AGENT2_SYSTEM = textwrap.dedent("""\
You are an expert systems programmer who writes benchmark tasks for ORBench v2.

Given a feasibility analysis of an OR problem, you generate ALL the files needed
for a complete ORBench task. You have deep knowledge of:
- C programming (the CPU reference must be pure C, no C++ features)
- CUDA programming conventions (the task_io.cu bridges harness and solution)
- Python data generation with numpy
- The ORBench binary I/O format (orbench_io_py.write_input_bin)

## ORBench Task Architecture

Each task lives in tasks/{task_id}/ with these files:

### 1. task.json — Metadata
```json
{
  "task_id": "example_task",
  "name": "Human-Readable Name",
  "category": "category",
  "difficulty": 2,
  "tags": ["tag1", "tag2"],
  "interface_mode": "compute_only",
  "input_sizes": {
    "small":  {"param1": 100,  "seed": 42},
    "medium": {"param1": 10000, "seed": 42},
    "large":  {"param1": 100000, "seed": 42}
  },
  "correctness": {
    "mode": "numerical",
    "atol": 0.01,
    "rtol": 0.01
  },
  "timing": {"warmup": 3, "trials": 10, "timeout": 180},
  "build_command": "",
  "extra_build_flags": "",
  "gpu_optimization_points": ["point1", "point2"]
}
```

### 2. prompt_template.yaml — LLM prompt template
Contains: task_description, interface (function signatures), input_size_notes,
output_contract, algorithm_background (L1 only), hints_l2, hints_l1.
The framework assembles these with the CPU reference into level-specific prompts.

### 3. cpu_reference.c — CPU baseline
Pure C. No file I/O. No external libraries.
- compute_only mode: implements solution_compute() + solution_free().
- init_compute mode: implements solution_init() + solution_compute() + solution_free().
Header comment block, then #include <stdlib.h> etc.

### 4. gen_data.py — Data generation
Generates input.bin (via orbench_io_py.write_input_bin), requests.txt,
and optionally expected_output.txt + cpu_time_ms.txt (with --with-expected).
Defines SIZES dict with small/medium/large configurations.
IMPORTANT: The script must add the ORBench root to sys.path so it can import
from framework.orbench_io_py:
```python
_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))
from framework.orbench_io_py import write_input_bin
```

### 5. task_io.cu — GPU I/O adapter
Bridges harness_gpu.cu <-> solution.cu. Includes cuda_runtime.h.
Implements: task_setup, task_run, task_write_output, task_cleanup.
task_setup extracts tensors/params from TaskData and stores them in a context struct.
task_run calls solution_compute with all data. task_cleanup calls solution_free.

### 6. task_io_cpu.c — CPU I/O adapter
Same as task_io.cu but pure C (no cuda_runtime.h).
Used to compile and run the CPU baseline for timing/validation.

## Interface Modes

Choose the appropriate mode based on the problem structure:

**compute_only** (preferred): No solution_init(). All work happens in solution_compute().
  - solution_compute() receives all input data (HOST pointers) and output buffers.
  - solution_free() is called after to release any persistent resources.
  - Use this when there are no multi-query requests on the same data.

**init_compute**: solution_init() called once, solution_compute() called many times.
  - init does cudaMalloc + H2D copies for shared data structures.
  - compute is timed per-request, must be idempotent.
  - Use this when the problem has multiple queries/requests on the same input data
    (e.g., multiple source vertices for shortest path, multiple patterns to match).

## Key Conventions
- All pointers in solution_compute are HOST pointers; the LLM manages H2D/D2H
- Use extern "C" linkage for both functions
- Unreachable/invalid values: 1e30f
- No file I/O in solution code
- requests.txt: one request per line, parsed by task_io
- output: one value per line in expected_output.txt

## Output Format

Respond with a JSON object containing all six files as string values:
{
  "task_json": "...",
  "prompt_template_yaml": "...",
  "cpu_reference_c": "...",
  "gen_data_py": "...",
  "task_io_cu": "...",
  "task_io_cpu_c": "..."
}
""")


# ---------------------------------------------------------------------------
#  Prompt builders
# ---------------------------------------------------------------------------

def build_agent1_prompt(files_block: str) -> str:
    return (
        f"Below are the files (papers and/or code) from a candidate OR problem.\n"
        f"Analyze them and determine if this problem is suitable for ORBench.\n\n"
        f"{files_block}\n\n"
        f"Provide your analysis as a JSON object following the schema in your instructions."
    )


def build_agent2_prompt(feasibility: FeasibilityResult, files_block: str) -> str:
    feas_json = json.dumps(asdict(feasibility), indent=2, ensure_ascii=False)
    return (
        f"Below is the feasibility analysis for an OR problem that has been deemed suitable\n"
        f"for ORBench. Also included are the original files (papers/code).\n\n"
        f"## Feasibility Analysis\n```json\n{feas_json}\n```\n\n"
        f"## Original Files\n{files_block}\n\n"
        f"Generate all 6 task files following the ORBench conventions in your instructions.\n"
        f"Make sure:\n"
        f"1. cpu_reference.c compiles with gcc -O2 -lm (pure C, no C++ features)\n"
        f"2. gen_data.py uses orbench_io_py.write_input_bin correctly\n"
        f"3. task_io.cu and task_io_cpu.c match the interface in cpu_reference.c exactly\n"
        f"4. prompt_template.yaml describes the problem clearly for an LLM\n"
        f"5. Suggested sizes produce meaningful computation (seconds on CPU for large)\n"
        f"6. The algorithm is correctly implemented in cpu_reference.c\n\n"
        f"Respond with a JSON object containing all six files."
    )


# ---------------------------------------------------------------------------
#  LLM call via ORBench registry
# ---------------------------------------------------------------------------

def call_llm(
    client,
    system: str,
    user_prompt: str,
    logger: AgentLogger,
    max_tokens: int = 65536,
    temperature: float = 0.4,
) -> LLMResponse:
    """
    Call LLM via ORBench registry client.
    System prompt is prepended to user prompt (not all providers support
    separate system prompts).
    """
    # Combine system + user into a single prompt
    full_prompt = f"<system>\n{system}\n</system>\n\n{user_prompt}"

    prompt_chars = len(full_prompt)
    # Rough token estimate: ~4 chars per token for English
    est_tokens = prompt_chars // 4
    print(f"    Prompt: {prompt_chars:,} chars (~{est_tokens:,} tokens est.)")
    logger.log_prompt(full_prompt)
    logger.log_info(f"Prompt stats: {prompt_chars:,} chars, ~{est_tokens:,} tokens (estimated)")

    t0 = time.monotonic()
    resp = client.generate(full_prompt, max_tokens=max_tokens, temperature=temperature)
    elapsed = (time.monotonic() - t0) * 1000

    print(f"    Response: {len(resp.content):,} chars, "
          f"in={resp.input_tokens} out={resp.output_tokens} tokens, "
          f"{resp.latency_ms:.0f}ms, ${resp.cost_usd:.4f}")
    logger.log_response(resp)

    if not resp.content or not resp.content.strip():
        raise RuntimeError(
            f"LLM returned empty response (input_tokens={resp.input_tokens}, "
            f"output_tokens={resp.output_tokens}, latency={resp.latency_ms:.0f}ms). "
            f"The prompt may be too large or the model may have timed out."
        )

    return resp


def parse_json_response(text: str) -> dict:
    """
    Extract a JSON object from the LLM response.
    Handles markdown fences and leading/trailing text.
    """
    # Try code fences first
    m = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]

    return json.loads(text)


# ---------------------------------------------------------------------------
#  Pipeline orchestration
# ---------------------------------------------------------------------------

def run_agent1(
    files: dict[str, str],
    client,
    model_id: str,
    logger: AgentLogger,
) -> FeasibilityResult:
    """Run Agent 1: Feasibility Checker."""
    logger.log_agent_start("Agent 1: Feasibility Checker", model_id)

    print(f"  [Agent 1] Files loaded: {len(files)}")
    for f in files:
        print(f"    - {f} ({len(files[f]):,} chars)")
    print()

    files_block = format_files_for_prompt(files)
    prompt = build_agent1_prompt(files_block)

    print(f"  [Agent 1] Calling {model_id} for feasibility analysis...")
    resp = call_llm(client, AGENT1_SYSTEM, prompt, logger)

    print(f"  [Agent 1] Response received ({resp.output_tokens} tokens, {resp.latency_ms:.0f}ms)")

    try:
        data = parse_json_response(resp.content)
    except json.JSONDecodeError as e:
        logger.log_error(f"Failed to parse Agent 1 response as JSON: {e}")
        print(f"  [Agent 1] ERROR: Failed to parse JSON response")
        raise

    result = FeasibilityResult(**{
        k: v for k, v in data.items()
        if k in FeasibilityResult.__dataclass_fields__
    })

    logger.log_result("Agent 1", asdict(result))

    score = result.suitability_score
    label = "EXCELLENT" if score >= 8 else "GOOD" if score >= 6 else "MARGINAL" if score >= 4 else "POOR"
    print(f"  [Agent 1] Score: {score}/10 ({label})")
    print(f"    Task ID: {result.task_id}")
    print(f"    Category: {result.category}")
    print(f"    Difficulty: {result.difficulty}")
    print(f"    Parallelism: {result.parallelism_analysis[:120]}...")
    if result.concerns:
        print(f"    Concerns:")
        for c in result.concerns:
            print(f"      - {c}")
    print()

    return result


def run_agent2(
    feasibility: FeasibilityResult,
    files: dict[str, str],
    client,
    model_id: str,
    logger: AgentLogger,
) -> TaskFiles:
    """Run Agent 2: Task Assembler."""
    logger.log_agent_start("Agent 2: Task Assembler", model_id)

    print(f"  [Agent 2] Generating task: {feasibility.task_id}")
    print()

    files_block = format_files_for_prompt(files)
    prompt = build_agent2_prompt(feasibility, files_block)

    print(f"  [Agent 2] Calling {model_id} for task generation...")
    resp = call_llm(client, AGENT2_SYSTEM, prompt, logger)

    print(f"  [Agent 2] Response received ({resp.output_tokens} tokens, {resp.latency_ms:.0f}ms)")

    try:
        data = parse_json_response(resp.content)
    except json.JSONDecodeError as e:
        logger.log_error(f"Failed to parse Agent 2 response as JSON: {e}")
        print(f"  [Agent 2] ERROR: Failed to parse JSON response")
        raise

    logger.log_result("Agent 2", {k: f"({len(v)} chars)" for k, v in data.items()})

    return TaskFiles(**data)


def write_task_folder(
    task_id: str,
    task_files: TaskFiles,
    orbench_root: str,
    logger: AgentLogger,
) -> Path:
    """Write all task files to tasks/{task_id}/."""
    task_dir = Path(orbench_root) / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "task.json": task_files.task_json,
        "prompt_template.yaml": task_files.prompt_template_yaml,
        "cpu_reference.c": task_files.cpu_reference_c,
        "gen_data.py": task_files.gen_data_py,
        "task_io.cu": task_files.task_io_cu,
        "task_io_cpu.c": task_files.task_io_cpu_c,
    }

    for filename, content in file_map.items():
        filepath = task_dir / filename
        filepath.write_text(content, encoding="utf-8")
        logger.log_info(f"Wrote: {filepath}")
        print(f"    {filepath}")

    gen_data_path = task_dir / "gen_data.py"
    gen_data_path.chmod(gen_data_path.stat().st_mode | 0o755)

    print(f"\n  Task folder created: {task_dir}")
    return task_dir


def process_single_paper(
    input_path: Path,
    model_id: str,
    orbench_root: str,
    force: bool = False,
    dry_run: bool = False,
    skip_feasibility: bool = False,
    min_score: int = 4,
):
    """Process a single paper/folder through the full pipeline."""
    paper_name = input_path.stem
    log_dir = _SCRIPT_DIR / "logs"
    logger = AgentLogger(log_dir, paper_name)

    print(f"\n{'='*70}")
    print(f"  Processing: {input_path}")
    print(f"  Model: {model_id}")
    print(f"  Log: {logger.log_path}")
    print(f"{'='*70}\n")

    # Init LLM client
    registry = LLMRegistry()
    client = registry.get_client(model_id)
    logger.log_info(f"Model config: {json.dumps(client.model, indent=2, default=str)}")

    # 1. Ingest files (PDF + supplementary code folder if present)
    print("  Reading input files...")
    files = read_folder(str(input_path))

    # Auto-discover supplementary code folder: <stem>_sup/ next to the PDF
    if input_path.is_file():
        sup_patterns = [
            input_path.parent / f"{input_path.stem}_sup",
            input_path.parent / f"{input_path.stem}_code",
            input_path.parent / f"{input_path.stem}_repo",
            input_path.parent / input_path.stem,  # e.g. ORpaper1/ next to ORpaper1.pdf
        ]
        for sup_dir in sup_patterns:
            if sup_dir.is_dir():
                print(f"  Found supplementary code: {sup_dir}")
                sup_files = read_folder(str(sup_dir))
                for rel, content in sup_files.items():
                    prefixed = f"{sup_dir.name}/{rel}"
                    files[prefixed] = content
                break

    if not files:
        msg = f"No supported files found in {input_path}"
        logger.log_error(msg)
        print(f"  ERROR: {msg}")
        logger.close()
        return

    for f_name, f_content in files.items():
        logger.log_info(f"Ingested: {f_name} ({len(f_content):,} chars)")

    # 2. Agent 1: Feasibility check
    if not skip_feasibility:
        try:
            feasibility = run_agent1(files, client, model_id, logger)
        except Exception as e:
            logger.log_error(f"Agent 1 crashed: {e}")
            print(f"  Agent 1 CRASHED: {e}")
            logger.close()
            return

        if feasibility.suitability_score < min_score and not force:
            print(f"  Score {feasibility.suitability_score}/10 below threshold {min_score}. Use --force or --min-score to override.")
            report_path = log_dir / f"{paper_name}_feasibility.json"
            report_path.write_text(
                json.dumps(asdict(feasibility), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  Feasibility report saved to: {report_path}")
            logger.close()
            return
    else:
        feasibility = FeasibilityResult(suitability_score=10, task_id="unknown_task")

    # 3. Agent 2: Task assembly
    try:
        task_files = run_agent2(feasibility, files, client, model_id, logger)
    except Exception as e:
        logger.log_error(f"Agent 2 crashed: {e}")
        print(f"  Agent 2 CRASHED: {e}")
        logger.close()
        return

    if dry_run:
        print(f"\n  [DRY RUN] Would write these files:")
        for name in ["task_json", "prompt_template_yaml", "cpu_reference_c",
                      "gen_data_py", "task_io_cu", "task_io_cpu_c"]:
            content = getattr(task_files, name)
            print(f"    {name}: {len(content)} chars")
        logger.close()
        return

    # 4. Write task folder
    print(f"\n  Writing task files:")
    task_dir = write_task_folder(
        feasibility.task_id, task_files, orbench_root, logger
    )

    # 5. Save feasibility report alongside
    report_path = task_dir / "feasibility_report.json"
    report_path.write_text(
        json.dumps(asdict(feasibility), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n  Done! Task '{feasibility.task_id}' created at {task_dir}")
    print(f"  Next steps:")
    print(f"    1. Review generated files")
    print(f"    2. cd {task_dir} && python gen_data.py small data/small --with-expected")
    print(f"    3. Compile and validate CPU baseline")

    logger.log_info(f"Pipeline complete. Task dir: {task_dir}")
    logger.close()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ORBench Auto-Add-Task Agent Pipeline"
    )
    parser.add_argument("input_path", help="Paper PDF, code folder, or parent folder (with --batch)")
    parser.add_argument(
        "--orbench-root", default=str(_ORBENCH_ROOT),
        help=f"Path to ORBench root directory (default: {_ORBENCH_ROOT})"
    )
    parser.add_argument(
        "--model", default="gemini-3.1-pro-preview",
        help="Model ID from models.yaml (default: gemini-3.1-pro-preview)"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Process all PDFs in the input folder individually"
    )
    parser.add_argument(
        "--skip-feasibility", action="store_true",
        help="Skip Agent 1 and go directly to task generation"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force task generation regardless of suitability score"
    )
    parser.add_argument(
        "--min-score", type=int, default=4,
        help="Minimum suitability score (1-10) to proceed to Agent 2 (default: 4)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run agents but don't write files"
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of papers to process in parallel (default: 1 = serial)"
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)

    if args.batch:
        # Process each PDF in the folder separately
        if not input_path.is_dir():
            print(f"ERROR: --batch requires a directory, got: {input_path}")
            sys.exit(1)
        pdfs = sorted(input_path.glob("*.pdf"))
        if not pdfs:
            print(f"ERROR: No PDF files found in {input_path}")
            sys.exit(1)

        num_parallel = min(args.parallel, len(pdfs))

        if num_parallel <= 1:
            # Serial mode
            print(f"Batch mode: processing {len(pdfs)} papers (serial)")
            for pdf in pdfs:
                try:
                    process_single_paper(
                        pdf, args.model, args.orbench_root,
                        force=args.force, dry_run=args.dry_run,
                        skip_feasibility=args.skip_feasibility,
                        min_score=args.min_score,
                    )
                except Exception as e:
                    print(f"  FAILED on {pdf.name}: {e}")
        else:
            # Parallel mode
            from concurrent.futures import ThreadPoolExecutor, as_completed
            print(f"Batch mode: processing {len(pdfs)} papers ({num_parallel} parallel)")

            def _run_one(pdf: Path):
                try:
                    process_single_paper(
                        pdf, args.model, args.orbench_root,
                        force=args.force, dry_run=args.dry_run,
                        skip_feasibility=args.skip_feasibility,
                        min_score=args.min_score,
                    )
                    return pdf.name, True, ""
                except Exception as e:
                    return pdf.name, False, str(e)

            with ThreadPoolExecutor(max_workers=num_parallel) as pool:
                futures = {pool.submit(_run_one, pdf): pdf for pdf in pdfs}
                for future in as_completed(futures):
                    name, ok, err = future.result()
                    if not ok:
                        print(f"  FAILED on {name}: {err}")
    else:
        process_single_paper(
            input_path, args.model, args.orbench_root,
            force=args.force, dry_run=args.dry_run,
            skip_feasibility=args.skip_feasibility,
            min_score=args.min_score,
        )


if __name__ == "__main__":
    main()
