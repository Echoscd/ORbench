"""
anthropic_client.py - Anthropic (Claude) API client.
"""

import time
from typing import Optional

from .base import BaseLLMClient, LLMResponse


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: float = 0.7) -> LLMResponse:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        max_tok = max_tokens or self.model.get("max_tokens", 8192)

        t0 = time.monotonic()
        message = client.messages.create(
            model=self.model["model_string"],
            max_tokens=max_tok,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = (time.monotonic() - t0) * 1000.0

        inp_tok = message.usage.input_tokens
        out_tok = message.usage.output_tokens

        return LLMResponse(
            content=message.content[0].text,
            input_tokens=inp_tok,
            output_tokens=out_tok,
            model=self.model["model_string"],
            latency_ms=latency,
            cost_usd=self.compute_cost(inp_tok, out_tok),
        )

