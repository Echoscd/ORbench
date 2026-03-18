"""
google_client.py - Google Gemini API client.
"""

import time
from typing import Optional

from .base import BaseLLMClient, LLMResponse


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models via google-genai SDK."""

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: float = 0.7) -> LLMResponse:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        max_tok = max_tokens or self.model.get("max_tokens", 8192)

        config = types.GenerateContentConfig(
            max_output_tokens=max_tok,
            temperature=temperature,
        )

        t0 = time.monotonic()
        response = client.models.generate_content(
            model=self.model["model_string"],
            contents=prompt,
            config=config,
        )
        latency = (time.monotonic() - t0) * 1000.0

        # Extract text — Gemini thinking models may split response into
        # multiple parts (thinking + output). Concatenate all text parts.
        text = ""

        # Method 1: iterate over parts (handles thinking models)
        candidates = response.candidates or []
        if candidates:
            candidate = candidates[0]
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                # Skip thinking parts, only take text output
                if getattr(part, "thought", False):
                    continue
                part_text = getattr(part, "text", None)
                if part_text:
                    text += part_text

        # Method 2: fallback to response.text
        if not text:
            try:
                text = response.text or ""
            except Exception:
                text = ""

        if not text:
            finish_reason = "unknown"
            if candidates:
                finish_reason = getattr(candidates[0], "finish_reason", "unknown")
            raise RuntimeError(
                f"Gemini returned empty response. "
                f"Finish reason: {finish_reason}"
            )

        # Extract token counts from usage_metadata
        usage = response.usage_metadata
        inp_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0

        return LLMResponse(
            content=text,
            input_tokens=inp_tok,
            output_tokens=out_tok,
            model=self.model["model_string"],
            latency_ms=latency,
            cost_usd=self.compute_cost(inp_tok, out_tok),
        )

