"""LLM integration layer: HTTP transport, structured calls, JSON parsing.

Submodules:
  provider — LLMProvider HTTP client with retries and concurrency control.
  caller   — llm_call with retry, JSON repair, and Pydantic validation.
  parse    — pure functions for JSON extraction, thinking trace removal, tool parsing.
"""

from .caller import JSON_SYSTEM_PROMPT, LLMCallResult, llm_call, raw_call
from .provider import ChatResult, LLMProvider, StreamChunk

__all__ = [
    "JSON_SYSTEM_PROMPT",
    "ChatResult",
    "LLMCallResult",
    "LLMProvider",
    "StreamChunk",
    "llm_call",
    "raw_call",
]
