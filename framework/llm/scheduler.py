"""
scheduler.py - Batch generation scheduler.

Orchestrates parallel LLM calls across multiple models, tasks, levels,
and samples with per-provider concurrency control, progress tracking,
and resume support.
"""

import os
import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from .base import LLMResponse
from .registry import LLMRegistry
from .resilient import RateLimiter, ResilientLLMClient
from . import logger as llm_logger


@dataclass
class GenerationJob:
    """A single generation work item."""
    model_id: str
    task_id: str
    level: int
    sample_id: int


@dataclass
class GenerationResult:
    """Result of a single generation attempt."""
    job: GenerationJob
    success: bool
    output_path: str = ""
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    timestamp: str = ""


def estimate_cost(
    registry: LLMRegistry,
    jobs: list[GenerationJob],
    avg_prompt_tokens: int = 2000,
    avg_output_tokens: int = 3000,
) -> float:
    """Rough cost estimate (USD) before running a batch."""
    total = 0.0
    for job in jobs:
        total += registry.estimate_single_cost(
            job.model_id, avg_prompt_tokens, avg_output_tokens
        )
    return total


class GenerationScheduler:
    """
    Coordinates batch LLM generation with:
    - Per-provider rate limiting (shared across models of the same provider)
    - Thread-pool concurrency
    - Resume / skip already-generated samples
    - Incremental progress file
    - Per-sample metadata JSON
    """

    def __init__(self, registry: LLMRegistry, runs_dir: str):
        self.registry = registry
        self.runs_dir = runs_dir
        self._split_kernels = False

        # One rate limiter per provider (shared across models of same provider)
        self._limiters: dict[str, RateLimiter] = {}
        for pname, pcfg in registry.config.get("providers", {}).items():
            rpm = pcfg.get("rate_limit", {}).get("requests_per_minute", 60)
            self._limiters[pname] = RateLimiter(rpm)

    # ── Job building ─────────────────────────────────────────

    def build_jobs(
        self,
        model_ids: list[str],
        task_ids: list[str],
        levels: list[int],
        num_samples: int,
    ) -> list[GenerationJob]:
        """Create all generation jobs from the cartesian product."""
        jobs = []
        for model_id in model_ids:
            for task_id in task_ids:
                for level in levels:
                    for s in range(num_samples):
                        jobs.append(GenerationJob(model_id, task_id, level, s))
        return jobs

    # ── Path helpers ─────────────────────────────────────────

    def _run_name(self, job: GenerationJob) -> str:
        date_tag = datetime.now().strftime("%Y%m%d")
        return f"{job.model_id}_l{job.level}_{date_tag}"

    def _output_path(self, job: GenerationJob) -> str:
        return os.path.join(
            self.runs_dir,
            self._run_name(job),
            job.task_id,
            f"sample_{job.sample_id}.cu",
        )

    def _already_done(self, job: GenerationJob) -> bool:
        p = self._output_path(job)
        return os.path.exists(p) and os.path.getsize(p) > 0

    # ── Single job execution ─────────────────────────────────

    def _execute_one(self, job: GenerationJob, temperature: float) -> GenerationResult:
        """Execute a single generation job (called in worker thread)."""
        # Resume support: skip if already generated
        if self._already_done(job):
            return GenerationResult(
                job=job, success=True,
                output_path=self._output_path(job),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

        try:
            # Load prompt
            from ..task import load_prompt
            prompt = load_prompt(job.task_id, job.level, split_kernels=self._split_kernels)

            # Build resilient client
            model_cfg = self.registry.get_model_config(job.model_id)
            provider = model_cfg["provider"]
            client = self.registry.get_client(job.model_id)
            retry_cfg = self.registry.get_retry_config(job.model_id)
            max_tok = model_cfg.get("max_tokens", 8192)

            resilient = ResilientLLMClient(
                client,
                self._limiters[provider],
                max_retries=retry_cfg.get("max_retries", 3),
                backoff_base=retry_cfg.get("backoff_base", 2.0),
            )

            # Log request
            llm_logger.log_request(
                model_id=job.model_id,
                model_string=model_cfg.get("model_string", ""),
                provider=provider,
                prompt=prompt,
                task_id=job.task_id,
                level=job.level,
                sample_id=job.sample_id,
                max_tokens=max_tok,
                temperature=temperature,
            )

            # Call LLM
            response = resilient.generate(prompt, temperature=temperature)

            # Log response
            llm_logger.log_response(
                model_id=job.model_id,
                task_id=job.task_id,
                level=job.level,
                sample_id=job.sample_id,
                content=response.content,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
            )

            # Extract CUDA code
            from ..generate import extract_cuda_code
            code = extract_cuda_code(response.content)

            # Save solution
            out_path = self._output_path(job)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Save raw response
            raw_path = out_path.replace(".cu", "_raw.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(response.content)

            # Save per-sample metadata
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            meta = {
                "model_id": job.model_id,
                "model_string": model_cfg.get("model_string", ""),
                "provider": provider,
                "task_id": job.task_id,
                "level": job.level,
                "sample_id": job.sample_id,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "latency_ms": response.latency_ms,
                "timestamp": ts,
            }
            meta_path = out_path.replace(".cu", "_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            return GenerationResult(
                job=job,
                success=True,
                output_path=out_path,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
                timestamp=ts,
            )

        except Exception as e:
            # Log error
            llm_logger.log_error(
                model_id=job.model_id,
                task_id=job.task_id,
                level=job.level,
                sample_id=job.sample_id,
                error=str(e)[:500],
                error_type=type(e).__name__,
            )
            return GenerationResult(
                job=job,
                success=False,
                error=str(e)[:500],
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

    # ── Batch execution ──────────────────────────────────────

    def run(
        self,
        jobs: list[GenerationJob],
        max_workers_per_provider: int = 3,
        progress_file: Optional[str] = None,
        temperature: float = 0.7,
        split_kernels: bool = False,
    ) -> list[GenerationResult]:
        """
        Execute all jobs with per-provider concurrency control.

        Different providers can run in parallel, but requests to the same
        provider are serialized by the rate limiter.
        """
        # Initialize log file for this batch
        llm_logger.init_logger(tag="batch")
        self._split_kernels = bool(split_kernels)

        results: list[GenerationResult] = []
        total = len(jobs)

        pending = [j for j in jobs if not self._already_done(j)]
        skipped = total - len(pending)

        if skipped > 0:
            print(f"  Skipping {skipped} already-generated samples (resume)")

        if not pending:
            print("  All samples already generated. Nothing to do.")
            return results

        # Count distinct providers in this batch
        providers_in_batch = set()
        for job in pending:
            providers_in_batch.add(
                self.registry.get_model_config(job.model_id)["provider"]
            )

        max_workers = len(providers_in_batch) * max_workers_per_provider
        print(f"  Generating {len(pending)} samples "
              f"({len(providers_in_batch)} providers, "
              f"max {max_workers} concurrent workers)...\n")

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_job = {}
            for job in pending:
                future = pool.submit(self._execute_one, job, temperature)
                future_to_job[future] = job

            done_count = 0
            total_cost = 0.0

            for future in as_completed(future_to_job):
                result = future.result()
                results.append(result)
                done_count += 1
                total_cost += result.cost_usd

                j = result.job
                status = "OK" if result.success else f"FAIL: {result.error[:80]}"
                cost_str = f"${result.cost_usd:.4f}" if result.cost_usd > 0 else "—"
                lat_str = f"{result.latency_ms:.0f}ms" if result.latency_ms > 0 else "—"

                print(
                    f"  [{done_count}/{len(pending)}] "
                    f"{j.model_id} / {j.task_id} / l{j.level} / s{j.sample_id}: "
                    f"{status}  ({cost_str}, {lat_str})"
                )

                # Incremental progress save
                if progress_file:
                    self._save_progress(results, progress_file)

        # Summary
        success_count = sum(1 for r in results if r.success)
        print(f"\n  Generation complete: {done_count} samples")
        print(f"  Success: {success_count}/{done_count}")
        print(f"  Total cost: ${total_cost:.4f}")

        log_path = llm_logger.get_log_file()
        if log_path:
            print(f"  Log file: {log_path}")

        return results

    # ── Progress persistence ─────────────────────────────────

    @staticmethod
    def _save_progress(results: list[GenerationResult], path: str):
        """Incrementally save progress to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = []
        for r in results:
            entry = {
                "model_id": r.job.model_id,
                "task_id": r.job.task_id,
                "level": r.job.level,
                "sample_id": r.job.sample_id,
                "success": r.success,
                "output_path": r.output_path,
                "error": r.error,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": r.cost_usd,
                "latency_ms": r.latency_ms,
                "timestamp": r.timestamp,
            }
            data.append(entry)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

