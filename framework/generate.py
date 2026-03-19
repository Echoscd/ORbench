"""
generate.py - Call LLM APIs to generate CUDA solutions.

Two modes:
  1. Legacy (single model):  generate_solutions()  — used by `run.py generate`
  2. Registry-based:         generate_with_registry()  — used by `run.py generate-batch`

The extract_cuda_code() function is shared by both modes and the scheduler.
"""

import os
import re
import json
import argparse
from datetime import datetime
from typing import Optional

from .task import load_task, load_prompt, ORBENCH_ROOT
from .config import get_config


# ═══════════════════════════════════════════════════════════════
#  Code extraction (shared utility)
# ═══════════════════════════════════════════════════════════════

def extract_cuda_code(response_text: str) -> str:
    """
    Extract CUDA code from LLM response.
    Handles markdown code blocks (```cuda, ```cpp, ```c, or bare ```)
    """
    if not response_text or not isinstance(response_text, str):
        raise ValueError(f"Cannot extract code: response is {type(response_text).__name__}")

    patterns = [
        r'```cuda\s*\n(.*?)```',
        r'```cpp\s*\n(.*?)```',
        r'```c\s*\n(.*?)```',
        r'```\s*\n(.*?)```',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            # Return the longest match (most likely the full solution)
            return max(matches, key=len).strip()

    # If no code block found, return the entire response
    return response_text.strip()


# ═══════════════════════════════════════════════════════════════
#  Legacy API callers (kept for backward compatibility)
# ═══════════════════════════════════════════════════════════════

def call_anthropic(model: str, prompt: str, api_key: str, max_tokens: int = 8192) -> str:
    """Call Anthropic API (legacy)."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def call_openai(model: str, prompt: str, api_key: str, api_base: str = None, max_tokens: int = 8192) -> str:
    """Call OpenAI-compatible API (legacy)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def call_llm(model: str, prompt: str, api_key: str, api_base: str = None, max_tokens: int = 8192) -> str:
    """Dispatch to the appropriate LLM API based on model name (legacy)."""
    if "claude" in model.lower():
        return call_anthropic(model, prompt, api_key, max_tokens)
    else:
        return call_openai(model, prompt, api_key, api_base, max_tokens)


# ═══════════════════════════════════════════════════════════════
#  Legacy single-model generation (run.py generate)
# ═══════════════════════════════════════════════════════════════

def generate_solutions(
    task_id: str,
    model: str,
    level: int,
    num_samples: int = 3,
    api_key: str = None,
    api_base: str = None,
    run_name: str = None,
    split_kernels: bool = False,
) -> list[str]:
    """
    Generate CUDA solutions for a task using the legacy single-model path.

    Returns list of file paths to saved solutions.
    """
    task = load_task(task_id)
    prompt = load_prompt(task_id, level, split_kernels=split_kernels)

    if run_name is None:
        date_tag = datetime.now().strftime("%Y%m%d")
        run_name = f"{model.replace('/', '_')}_l{level}_{date_tag}"

    run_dir = os.path.join(ORBENCH_ROOT, "runs", run_name, task_id)
    os.makedirs(run_dir, exist_ok=True)

    saved_paths = []
    for i in range(num_samples):
        output_path = os.path.join(run_dir, f"sample_{i}.cu")

        # Skip if already generated
        if os.path.exists(output_path):
            print(f"  [SKIP] {task_id} sample_{i} already exists")
            saved_paths.append(output_path)
            continue

        print(f"  [GEN] {task_id} sample_{i} with {model}...")
        try:
            config = get_config()
            response = call_llm(model, prompt, api_key, api_base, max_tokens=config.llm.max_tokens)
            code = extract_cuda_code(response)

            with open(output_path, "w") as f:
                f.write(code)

            # Also save raw response for debugging
            raw_path = os.path.join(run_dir, f"sample_{i}_raw.txt")
            with open(raw_path, "w") as f:
                f.write(response)

            saved_paths.append(output_path)
            print(f"  [OK] Saved to {output_path}")

        except Exception as e:
            print(f"  [ERROR] {task_id} sample_{i}: {e}")

    return saved_paths


# ═══════════════════════════════════════════════════════════════
#  Registry-based generation (run.py generate-batch)
# ═══════════════════════════════════════════════════════════════

def generate_with_registry(
    model_id: str,
    task_id: str,
    level: int,
    sample_id: int,
    registry=None,
    run_name: str = None,
    temperature: float = 0.7,
    split_kernels: bool = False,
) -> dict:
    """
    Generate a single sample using the LLM registry.

    Returns a dict with keys: success, output_path, input_tokens,
    output_tokens, cost_usd, latency_ms, error.
    """
    from .llm.registry import LLMRegistry
    from .llm.resilient import RateLimiter, ResilientLLMClient

    if registry is None:
        registry = LLMRegistry()

    prompt = load_prompt(task_id, level, split_kernels=split_kernels)

    if run_name is None:
        run_name = f"{model_id}_l{level}"

    run_dir = os.path.join(ORBENCH_ROOT, "runs", run_name, task_id)
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, f"sample_{sample_id}.cu")

    # Skip if exists
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return {"success": True, "output_path": output_path, "skipped": True}

    try:
        client = registry.get_client(model_id)
        rate_cfg = registry.get_rate_limit(model_id)
        retry_cfg = registry.get_retry_config(model_id)

        limiter = RateLimiter(rate_cfg.get("requests_per_minute", 60))
        resilient = ResilientLLMClient(
            client, limiter,
            max_retries=retry_cfg.get("max_retries", 3),
            backoff_base=retry_cfg.get("backoff_base", 2.0),
        )

        model_cfg = registry.get_model_config(model_id)
        max_tokens = model_cfg.get("max_tokens", 8192)

        response = resilient.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        code = extract_cuda_code(response.content)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)

        raw_path = os.path.join(run_dir, f"sample_{sample_id}_raw.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(response.content)

        return {
            "success": True,
            "output_path": output_path,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
            "latency_ms": response.latency_ms,
        }

    except Exception as e:
        return {"success": False, "error": str(e)[:500]}


# ═══════════════════════════════════════════════════════════════
#  CLI entry point (standalone)
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate CUDA solutions using LLMs")
    parser.add_argument("--task", required=True, help="Task ID (e.g., bellman_ford)")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--level", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY"))
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--split", action="store_true",
                        help="Encourage the LLM to split the solution into multiple kernels (profiling-friendly)")
    args = parser.parse_args()

    generate_solutions(
        task_id=args.task,
        model=args.model,
        level=args.level,
        num_samples=args.samples,
        api_key=args.api_key,
        api_base=args.api_base,
        run_name=args.run_name,
        split_kernels=args.split,
    )


if __name__ == "__main__":
    main()
