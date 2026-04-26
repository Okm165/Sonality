from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    """Read an environment variable as string with a default."""
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    """Read an environment variable as integer with a default."""
    return int(_env_str(name, str(default)))


def _env_float(name: str, default: float) -> float:
    """Read an environment variable as float with a default."""
    return float(_env_str(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    """Read an environment variable as boolean with a default."""
    return _env_str(name, str(default).lower()).lower() in ("true", "1", "yes")


DATA_DIR: Final = PROJECT_ROOT / "data"
API_KEY: Final = _env_str("SONALITY_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
BASE_URL: Final = _env_str("SONALITY_BASE_URL", "https://api.openai.com/v1")
MODEL: Final = _env_str("SONALITY_MODEL", "gpt-4.1-mini")
ESS_MODEL: Final = _env_str("SONALITY_ESS_MODEL", MODEL)
LOG_LEVEL: Final = _env_str("SONALITY_LOG_LEVEL", "INFO")

# --- Database (Neo4j + Qdrant) ---
NEO4J_URL: Final = _env_str("SONALITY_NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER: Final = _env_str("SONALITY_NEO4J_USER", "neo4j")
NEO4J_PASSWORD: Final = _env_str("SONALITY_NEO4J_PASSWORD", "sonality_password")
NEO4J_DATABASE: Final = _env_str("SONALITY_NEO4J_DATABASE", "neo4j")

QDRANT_URL: Final = _env_str("SONALITY_QDRANT_URL", "http://localhost:6333")

# --- Embedding (local FastEmbed bge-large-en-v1.5, 1024 dims) ---
EMBEDDING_DIMENSIONS: Final = 1024
EMBEDDING_MAX_CHARS: Final = _env_int("SONALITY_EMBEDDING_MAX_CHARS", 4096)
# --- Qdrant search tuning ---
QDRANT_SEARCH_EF: Final = _env_int("SONALITY_QDRANT_SEARCH_EF", 128)
QDRANT_RESCORE_QUANTIZED: Final = _env_bool("SONALITY_QDRANT_RESCORE", True)

# --- LLM for scoring/assessment tasks (fast, cheap model) ---
FAST_LLM_MODEL: Final = _env_str("SONALITY_FAST_LLM_MODEL", ESS_MODEL)
AGENT_TEMPERATURE: Final = _env_float("SONALITY_AGENT_TEMPERATURE", 0.6)

# --- LLM max_tokens (unified for all task types) ---
LLM_MAX_TOKENS: Final = _env_int("SONALITY_LLM_MAX_TOKENS", 8192)

# --- Retrieval ---
RETRIEVAL_MAX_ITERATIONS: Final = _env_int("SONALITY_RETRIEVAL_MAX_ITERATIONS", 3)
RETRIEVAL_CONFIDENCE_THRESHOLD: Final = _env_float("SONALITY_RETRIEVAL_CONFIDENCE_THRESHOLD", 0.8)
RETRIEVAL_OVER_FETCH_FACTOR: Final = _env_int("SONALITY_RETRIEVAL_OVER_FETCH_FACTOR", 3)
MAX_RERANK_CANDIDATES: Final = _env_int("SONALITY_MAX_RERANK_CANDIDATES", 50)

# Per-HTTP-request timeout for LLM calls. 300s (5 min) covers 4096-token outputs
# on slower endpoints. Increase further for 70B+ models or throttled endpoints.
LLM_REQUEST_TIMEOUT: Final = _env_int("SONALITY_LLM_TIMEOUT", 300)

# Timeout for async ops dispatched from the sync context via run_coroutine_threadsafe.
# Must exceed the maximum realistic duration of any single coroutine, which in the
# worst case is: semaphore_queue_wait + max_retries x LLM_REQUEST_TIMEOUT.
# Default: 5 x LLM_REQUEST_TIMEOUT.
ASYNC_TIMEOUT: Final = _env_int("SONALITY_ASYNC_TIMEOUT", LLM_REQUEST_TIMEOUT * 5)


def missing_live_api_config() -> tuple[str, ...]:
    """Return required live configuration keys that are currently unset.

    API key is optional for local OpenAI-compatible servers (for example, Ollama).
    """
    return ("SONALITY_BASE_URL",) if not BASE_URL.strip() else ()
