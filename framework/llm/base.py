"""
base.py - Base LLM client and response dataclasses.

All provider clients inherit from BaseLLMClient.
"""

import os
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float
    cost_usd: float


@dataclass
class LLMError:
    """Structured error from LLM call."""
    error_type: str   # "rate_limit", "timeout", "auth", "server", "unknown"
    message: str
    retryable: bool


class BaseLLMClient(ABC):
    """
    Abstract base for all provider clients.

    Subclasses implement generate() for a single API call (no retry logic here;
    retry is handled by ResilientLLMClient).
    """

    def __init__(self, provider_config: dict, model_config: dict):
        self.provider = provider_config
        self.model = model_config
        self.api_key = os.environ.get(provider_config.get("api_key_env", ""), "")

    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: float = 0.7) -> LLMResponse:
        """
        Single API call. No retry.

        Args:
            prompt: The user prompt text.
            max_tokens: Override model default max_tokens.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with content, token usage, cost, latency.
        """
        ...

    def compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Compute USD cost from token counts and model pricing."""
        return (
            input_tokens / 1000.0 * self.model.get("cost_per_1k_input", 0)
            + output_tokens / 1000.0 * self.model.get("cost_per_1k_output", 0)
        )

