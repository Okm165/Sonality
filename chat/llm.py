"""Async LLM utilities for chat module — wraps sonality.provider."""

from __future__ import annotations

import asyncio

from sonality.provider import chat_completion

from . import config


async def llm_call(prompt: str, max_tokens: int, temperature: float) -> str:
    """Async LLM completion using sonality's provider."""

    def _sync_call() -> str:
        result = chat_completion(
            model=config.TTS_OPTIMIZE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=False,
            base_url=config.TTS_OPTIMIZE_BASE_URL,
            api_key=config.TTS_OPTIMIZE_API_KEY,
        )
        return result.text

    return await asyncio.to_thread(_sync_call)
