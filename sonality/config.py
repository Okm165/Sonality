"""Centralized configuration — all environment variables in one place.

Values are read once at import time from .env (via python-dotenv) and frozen
as module-level Final constants. See .env.example for full documentation of
each variable's purpose, valid values, and tuning guidance.

Sections:
    - Core LLM endpoint and model tiers
    - Database connections (Neo4j, Qdrant)
    - Embedding and vector search tuning
    - Token budgets (per-call max_tokens for every LLM invocation)
    - Limits, timeouts, and retry behavior
    - Web access (Firecrawl search + scrape)
    - HTTP API authentication
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def env_str(name: str, default: str) -> str:
    """Read an environment variable as string with a default."""
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    """Read an environment variable as integer with a default."""
    return int(env_str(name, str(default)))


def env_float(name: str, default: float) -> float:
    """Read an environment variable as float with a default."""
    return float(env_str(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    """Read an environment variable as boolean with a default."""
    return env_str(name, str(default).lower()).lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Core LLM endpoint
# ---------------------------------------------------------------------------

DATA_DIR: Final = PROJECT_ROOT / "data"
API_KEY: Final = env_str("SONALITY_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
BASE_URL: Final = env_str("SONALITY_BASE_URL", "https://api.openai.com/v1")
MODEL: Final = env_str("SONALITY_MODEL", "gpt-4.1-mini")
LOG_LEVEL: Final = env_str("SONALITY_LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Model tiers — each defaults to MODEL; override per-tier for speed/quality.
# FAST:       routing, boundary detection, reranking, summarization
# STRUCTURED: ESS, knowledge extraction (needs tool_choice)
# AGENT:      main agentic loop with tool calling
# REASONING:  reflection, synthesis, forgetting, quorum challenger (can be a thinking model)
# ---------------------------------------------------------------------------

AGENT_MODEL: Final = env_str("SONALITY_AGENT_MODEL", MODEL)
REASONING_MODEL: Final = env_str("SONALITY_REASONING_MODEL", MODEL)
STRUCTURED_MODEL: Final = env_str("SONALITY_STRUCTURED_MODEL", MODEL)
FAST_MODEL: Final = env_str("SONALITY_FAST_MODEL", MODEL)

# ---------------------------------------------------------------------------
# Database connections
# ---------------------------------------------------------------------------

NEO4J_URL: Final = env_str("SONALITY_NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER: Final = env_str("SONALITY_NEO4J_USER", "neo4j")
NEO4J_PASSWORD: Final = env_str("SONALITY_NEO4J_PASSWORD", "sonality_password")
NEO4J_DATABASE: Final = env_str("SONALITY_NEO4J_DATABASE", "neo4j")
NEO4J_MAX_POOL_SIZE: Final = env_int("SONALITY_NEO4J_MAX_POOL_SIZE", 50)
NEO4J_CONNECTION_TIMEOUT: Final = env_float("SONALITY_NEO4J_CONNECTION_TIMEOUT", 30.0)

QDRANT_URL: Final = env_str("SONALITY_QDRANT_URL", "http://localhost:6333")

# ---------------------------------------------------------------------------
# Embedding — local FastEmbed bge-large-en-v1.5 (1024 dims, no API needed)
# ---------------------------------------------------------------------------

EMBEDDING_DIMENSIONS: Final = 1024
EMBEDDING_MAX_CHARS: Final = env_int("SONALITY_EMBEDDING_MAX_CHARS", 4096)
EMBEDDING_CACHE_SIZE: Final = env_int("SONALITY_EMBEDDING_CACHE_SIZE", 10000)

# ---------------------------------------------------------------------------
# Qdrant search tuning
# ---------------------------------------------------------------------------

QDRANT_SEARCH_EF: Final = env_int("SONALITY_QDRANT_SEARCH_EF", 128)
QDRANT_RESCORE_QUANTIZED: Final = env_bool("SONALITY_QDRANT_RESCORE", True)

# ---------------------------------------------------------------------------
# LLM generation parameters
# ---------------------------------------------------------------------------

AGENT_TEMPERATURE: Final = env_float("SONALITY_AGENT_TEMPERATURE", 0.6)

# ---------------------------------------------------------------------------
# Token budget — single max_tokens for all LLM calls (set high to never constrain)
# ---------------------------------------------------------------------------

LLM_MAX_TOKENS: Final = env_int("SONALITY_LLM_MAX_TOKENS", 8192)

# ---------------------------------------------------------------------------
# Agent loop parameters
# ---------------------------------------------------------------------------

AGENT_LOOP_HARD_CEILING: Final = env_int("SONALITY_AGENT_LOOP_HARD_CEILING", 12)
AGENT_COMPRESS_THRESHOLD: Final = env_int("SONALITY_AGENT_COMPRESS_THRESHOLD", 12)
AGENT_COMPRESS_KEEP_TAIL: Final = env_int("SONALITY_AGENT_COMPRESS_KEEP_TAIL", 4)

# ---------------------------------------------------------------------------
# Limits, timeouts, and retry behavior
# ---------------------------------------------------------------------------

EPISODE_CONTENT_LIMIT: Final = env_int("SONALITY_EPISODE_CONTENT_LIMIT", 300)
BELIEF_PROMPT_WINDOW: Final = env_int("SONALITY_BELIEF_PROMPT_WINDOW", 15)
ESS_TIMEOUT: Final = env_int("SONALITY_ESS_TIMEOUT", 300)
FORGETTING_CANDIDATE_LIMIT: Final = env_int("SONALITY_FORGETTING_CANDIDATE_LIMIT", 10)

HTTP_API_KEY: Final[str | None] = os.environ.get("SONALITY_HTTP_API_KEY")

LLM_MAX_RETRIES: Final = env_int("SONALITY_LLM_MAX_RETRIES", 3)
LLM_BACKOFF_BASE: Final = env_float("SONALITY_LLM_BACKOFF_BASE", 2.0)

LLM_CONCURRENCY: Final = max(1, env_int("SONALITY_LLM_CONCURRENCY", 1))

# Total input token budget (system + messages); oldest turns trimmed when exceeded.
CHAT_INPUT_TOKEN_BUDGET: Final = env_int("SONALITY_CHAT_INPUT_TOKEN_BUDGET", 128000)

# ---------------------------------------------------------------------------
# Retrieval pipeline
# ---------------------------------------------------------------------------

RETRIEVAL_MAX_ITERATIONS: Final = env_int("SONALITY_RETRIEVAL_MAX_ITERATIONS", 3)
RETRIEVAL_OVER_FETCH_FACTOR: Final = env_int("SONALITY_RETRIEVAL_OVER_FETCH_FACTOR", 3)
MAX_RERANK_CANDIDATES: Final = env_int("SONALITY_MAX_RERANK_CANDIDATES", 50)

# ---------------------------------------------------------------------------
# HTTP timeouts
# ---------------------------------------------------------------------------

# 600s (10 min) covers slow local models (32B+ on consumer hardware).
LLM_REQUEST_TIMEOUT: Final = env_int("SONALITY_LLM_TIMEOUT", 600)

# Must exceed: semaphore wait + max_retries x LLM_REQUEST_TIMEOUT.
ASYNC_TIMEOUT: Final = env_int("SONALITY_ASYNC_TIMEOUT", LLM_REQUEST_TIMEOUT * 5)

# ---------------------------------------------------------------------------
# Web access — Firecrawl self-hosted search + scrape API
# ---------------------------------------------------------------------------

WEB_SEARCH_URL: Final = env_str("SONALITY_WEB_SEARCH_URL", "http://localhost:3002")
WEB_SEARCH_ENABLED: Final = env_bool("SONALITY_WEB_SEARCH_ENABLED", True)
WEB_CACHE_TTL: Final = env_int("SONALITY_WEB_CACHE_TTL", 14400)
REFLECTION_WEB_QUERIES: Final = env_int("SONALITY_REFLECTION_WEB_QUERIES", 3)


def missing_live_api_config() -> tuple[str, ...]:
    """Return required configuration keys that are unset.

    API key is optional for local OpenAI-compatible servers (Ollama, llama.cpp).
    """
    return ("SONALITY_BASE_URL",) if not BASE_URL.strip() else ()
