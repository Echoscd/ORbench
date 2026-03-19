"""
prompts.py - Prompt helpers for agent-style multi-turn optimization loops.
"""

BASE_AGENT_INSTRUCTIONS = """\
You are optimizing a CUDA implementation produced in a previous attempt.
You will be given:
- The previous CUDA source code
- The results from compile / correctness validation / profiling summary

Your job:
1) If there is any compile error or runtime error, fix it first.
2) If correctness failed, fix correctness next (do not sacrifice correctness for speed).
3) If correctness passed, optimize performance. Focus on the biggest bottlenecks reported.

Rules:
- Output a complete, compilable single-file .cu implementation (solution_init + solution_compute).
- Do NOT include explanations outside the code block.
- Do NOT use file I/O or printf.
- Do NOT cudaMalloc in solution_compute.
"""


def build_feedback_prompt(
    task_prompt: str,
    prev_code: str,
    eval_summary: str,
) -> str:
    """
    Compose a multi-turn prompt that includes:
    - The original task prompt (interface + constraints + CPU ref)
    - Agent instructions
    - Previous code
    - Evaluation / profiling feedback
    """
    return (
        task_prompt.rstrip()
        + "\n\n"
        + "## Agent Mode: Iterative Optimization\n\n"
        + BASE_AGENT_INSTRUCTIONS
        + "\n\n"
        + "## Previous Attempt (CUDA Source)\n\n"
        + "```c\n"
        + prev_code.rstrip()
        + "\n```\n\n"
        + "## Evaluation & Profiling Feedback\n\n"
        + eval_summary.rstrip()
        + "\n"
    )


