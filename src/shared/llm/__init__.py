"""LLM integration layer: HTTP transport, structured calls, output cleanup.

Submodules:
  parse    — all output cleanup: thinking removal, JSON extraction, tool parsing.
  provider — LLMProvider HTTP client with retries and threading concurrency.
  caller   — llm_call (sync), async_llm_call (async bridge), format_prompt,
             compose_guarded.  Per-module semaphore gating is the caller's job.
"""
