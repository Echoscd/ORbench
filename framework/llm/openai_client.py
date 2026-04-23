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

        api_base = self.provider.get("api_base", "")
        is_openrouter = "openrouter.ai" in api_base

        client = OpenAI(
            api_key=self.api_key,
            base_url=api_base,
            timeout=600.0,
        )
        max_tok = max_tokens or self.model.get("max_tokens", 8192)
        supports_system = self.model.get("supports_system_prompt", True)

        messages = [{"role": "user", "content": prompt}]

        kwargs = dict(
            model=self.model["model_string"],
            messages=messages,
            max_tokens=max_tok,
        )
        if supports_system:
            kwargs["temperature"] = temperature

        if is_openrouter:
            kwargs["extra_body"] = {
                "provider": {
                    "sort": "throughput",
                    "allow_fallbacks": True,
                }
            }

        t0 = time.monotonic()
        response = client.chat.completions.create(**kwargs)
        latency = (time.monotonic() - t0) * 1000.0

        if not response.choices:
            raise RuntimeError(f"empty_response: no choices returned (model={self.model['model_string']})")
        content = response.choices[0].message.content
        if content is None or content == "":
            raise RuntimeError(f"empty_response: content is None/empty (model={self.model['model_string']})")

        usage = response.usage
        inp_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0

        return LLMResponse(
            content=content,
            input_tokens=inp_tok,
            output_tokens=out_tok,
            model=self.model["model_string"],
            latency_ms=latency,
            cost_usd=self.compute_cost(inp_tok, out_tok),
        )

