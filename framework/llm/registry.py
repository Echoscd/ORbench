"""
registry.py - LLM model registry.

Loads models.yaml and provides get_client(model_id) to obtain a properly
configured provider client for any registered model.
"""

import os
import yaml
from typing import Optional

from .base import BaseLLMClient
from .anthropic_client import AnthropicClient
from .openai_client import OpenAIClient
from .google_client import GoogleClient


# Provider name → client class mapping.
# DeepSeek uses the OpenAI-compatible format.
CLIENT_MAP: dict[str, type] = {
    "anthropic": AnthropicClient,
    "openai":    OpenAIClient,
    "deepseek":  OpenAIClient,
    "google":    GoogleClient,
}


class LLMRegistry:
    """
    Central registry for all LLM models and providers.

    Usage::

        registry = LLMRegistry("models.yaml")
        client = registry.get_client("claude-sonnet-4")
        response = client.generate(prompt)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Load models.yaml.

        Args:
            config_path: Path to models.yaml. Defaults to ORBENCH_ROOT/models.yaml.
        """
        if config_path is None:
            from ..task import ORBENCH_ROOT
            config_path = os.path.join(ORBENCH_ROOT, "models.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model registry not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self._providers = self.config.get("providers", {})
        self._models = self.config.get("models", {})

    # ── Query ────────────────────────────────────────────────

    def list_models(self) -> list[str]:
        """Return all registered model IDs."""
        return list(self._models.keys())

    def list_providers(self) -> list[str]:
        """Return all registered provider names."""
        return list(self._providers.keys())

    def get_model_config(self, model_id: str) -> dict:
        """Return the raw model config dict for *model_id*."""
        if model_id not in self._models:
            raise KeyError(f"Unknown model: {model_id}. "
                           f"Available: {', '.join(self.list_models())}")
        return self._models[model_id]

    def get_provider_config(self, model_id: str) -> dict:
        """Return the provider config dict for *model_id*."""
        model_cfg = self.get_model_config(model_id)
        provider_name = model_cfg["provider"]
        if provider_name not in self._providers:
            raise KeyError(f"Unknown provider: {provider_name}")
        return self._providers[provider_name]

    def get_provider_name(self, model_id: str) -> str:
        """Return the provider name (e.g. 'anthropic') for *model_id*."""
        return self.get_model_config(model_id)["provider"]

    # ── Client construction ──────────────────────────────────

    def get_client(self, model_id: str) -> BaseLLMClient:
        """
        Create and return a provider client for *model_id*.

        The returned client is stateless; callers may cache it if desired.
        """
        model_cfg = self.get_model_config(model_id)
        provider_name = model_cfg["provider"]
        provider_cfg = self.get_provider_config(model_id)

        client_cls = CLIENT_MAP.get(provider_name)
        if client_cls is None:
            raise ValueError(
                f"No client implementation for provider '{provider_name}'. "
                f"Supported: {', '.join(CLIENT_MAP.keys())}"
            )
        return client_cls(provider_cfg, model_cfg)

    # ── Rate limit / retry config ────────────────────────────

    def get_rate_limit(self, model_id: str) -> dict:
        """Return rate limit config for the provider of *model_id*."""
        provider_cfg = self.get_provider_config(model_id)
        return provider_cfg.get("rate_limit", {
            "requests_per_minute": 60,
            "tokens_per_minute": 150000,
        })

    def get_retry_config(self, model_id: str) -> dict:
        """Return retry config for the provider of *model_id*."""
        provider_cfg = self.get_provider_config(model_id)
        return provider_cfg.get("retry", {
            "max_retries": 3,
            "backoff_base": 2.0,
        })

    # ── Cost helpers ─────────────────────────────────────────

    def estimate_single_cost(self, model_id: str,
                             input_tokens: int = 2000,
                             output_tokens: int = 3000) -> float:
        """Estimate USD cost for a single call with given token counts."""
        model_cfg = self.get_model_config(model_id)
        return (
            input_tokens / 1000.0 * model_cfg.get("cost_per_1k_input", 0)
            + output_tokens / 1000.0 * model_cfg.get("cost_per_1k_output", 0)
        )

