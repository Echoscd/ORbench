"""
resilient.py - Retry and rate-limiting wrappers for LLM clients.

RateLimiter:  Token-bucket style per-provider rate limiting.
ResilientLLMClient:  Wraps any BaseLLMClient with automatic retry + backoff.
"""

import time
import threading
from typing import Optional

from .base import BaseLLMClient, LLMResponse


# ── Error classification ─────────────────────────────────────


def classify_error(e: Exception) -> str:
    """Classify an API error into a category."""
    msg = str(e).lower()
    if "rate" in msg or "429" in msg:
        return "rate_limit"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "auth" in msg or "401" in msg or "403" in msg:
        return "auth"
    if "404" in msg or "not_found" in msg or "not found" in msg:
        return "not_found"
    if "500" in msg or "502" in msg or "503" in msg:
        return "server"
    return "unknown"


def is_retryable(error_type: str) -> bool:
    """Decide whether an error category is worth retrying."""
    return error_type in ("rate_limit", "timeout", "server")


# ── Rate limiter ─────────────────────────────────────────────


class RateLimiter:
    """
    Simple token-bucket rate limiter (per-provider).

    Enforces a minimum interval between requests so that the aggregate
    rate does not exceed *rpm* requests per minute.
    """

    def __init__(self, rpm: int):
        self.interval = 60.0 / max(rpm, 1)
        self._lock = threading.Lock()
        self._last_call = 0.0

    def acquire(self):
        """Block until the next request is allowed."""
        with self._lock:
            now = time.monotonic()
            wait = self._last_call + self.interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


# ── Resilient wrapper ────────────────────────────────────────


class ResilientLLMClient:
    """
    Wraps a BaseLLMClient with:

    1. Per-provider rate limiting (via a shared RateLimiter).
    2. Exponential-backoff retry on transient errors.
    """

    def __init__(
        self,
        client: BaseLLMClient,
        rate_limiter: RateLimiter,
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ):
        self.client = client
        self.limiter = rate_limiter
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: float = 0.7) -> LLMResponse:
        """
        Call the underlying client with retry + rate limiting.

        Raises the last exception if all retries are exhausted.
        """
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            try:
                return self.client.generate(prompt, max_tokens, temperature)
            except Exception as e:
                last_error = e
                error_type = classify_error(e)

                if not is_retryable(error_type):
                    raise

                wait = self.backoff_base ** attempt
                print(f"    [retry] attempt {attempt + 1}/{self.max_retries}, "
                      f"waiting {wait:.1f}s: {error_type} — {str(e)[:120]}")
                time.sleep(wait)

        # Should not be reached, but just in case
        raise last_error  # type: ignore[misc]

