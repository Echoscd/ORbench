"""
openai_client.py - OpenAI-compatible API client.

Works for OpenAI, DeepSeek, and any provider that follows the OpenAI
chat completions format.
"""

import time
from typing import Optional

from .base import BaseLLMClient, LLMResponse


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI and OpenAI-compatible APIs (DeepSeek, etc.)."""

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: float = 0.7) -> LLMResponse:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.provider.get("api_base"),
        )
        max_tok = max_tokens or self.model.get("max_tokens", 8192)
        supports_system = self.model.get("supports_system_prompt", True)

        # Reasoning models (o3-mini, deepseek-r1) don't support system prompt
        # and may not support temperature setting
        messages = [{"role": "user", "content": prompt}]

        # Build kwargs — some reasoning models don't accept temperature
        kwargs = dict(
            model=self.model["model_string"],
            messages=messages,
            max_tokens=max_tok,
        )
        # Only set temperature for non-reasoning models
        if supports_system:
            kwargs["temperature"] = temperature

        t0 = time.monotonic()
        response = client.chat.completions.create(**kwargs)
        latency = (time.monotonic() - t0) * 1000.0

        usage = response.usage
        inp_tok = usage.prompt_tokens
        out_tok = usage.completion_tokens

        return LLMResponse(
            content=response.choices[0].message.content,
            input_tokens=inp_tok,
            output_tokens=out_tok,
            model=self.model["model_string"],
            latency_ms=latency,
            cost_usd=self.compute_cost(inp_tok, out_tok),
        )

