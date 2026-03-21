"""Async LLM utilities for chat module — uses sonality.provider.LLMProvider."""

from __future__ import annotations

import asyncio

from sonality.provider import LLMProvider

from . import config

# Chat module's own provider instance (separate semaphore from sonality)
provider = LLMProvider(
    config.TTS_OPTIMIZE_BASE_URL, config.TTS_OPTIMIZE_API_KEY, int(config.TTS_OPTIMIZE_TIMEOUT)
)


async def llm_call(prompt: str, max_tokens: int, temperature: float) -> str:
    """Async LLM completion using chat's provider."""
    return await asyncio.to_thread(
        lambda: (
            provider.chat_completion(
                model=config.TTS_OPTIMIZE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                enable_thinking=False,
            ).text
        )
    )
