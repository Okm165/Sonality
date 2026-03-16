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


DATA_DIR: Final = PROJECT_ROOT / "data"
SPONGE_FILE: Final = DATA_DIR / "sponge.json"
SPONGE_HISTORY_DIR: Final = DATA_DIR / "sponge_history"
ESS_AUDIT_LOG_FILE: Final = DATA_DIR / "ess_log.jsonl"
API_KEY: Final = _env_str("SONALITY_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
BASE_URL: Final = _env_str("SONALITY_BASE_URL", "https://api.openai.com/v1")
MODEL: Final = _env_str("SONALITY_MODEL", "gpt-4.1-mini")
ESS_MODEL: Final = _env_str("SONALITY_ESS_MODEL", MODEL)
LOG_LEVEL: Final = _env_str("SONALITY_LOG_LEVEL", "INFO")

SPONGE_MAX_TOKENS: Final = 500
EPISODIC_RETRIEVAL_COUNT: Final = _env_int("SONALITY_EPISODIC_RETRIEVAL_COUNT", 3)
SEMANTIC_RETRIEVAL_COUNT: Final = _env_int("SONALITY_SEMANTIC_RETRIEVAL_COUNT", 2)

OPINION_COOLING_PERIOD: Final = _env_int("SONALITY_OPINION_COOLING_PERIOD", 3)

MAX_CONVERSATION_CHARS: Final = 100_000
REFLECTION_EVERY: Final = _env_int("SONALITY_REFLECTION_EVERY", 20)

# --- Database (Neo4j + Qdrant) ---
NEO4J_URL: Final = _env_str("SONALITY_NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER: Final = _env_str("SONALITY_NEO4J_USER", "neo4j")
NEO4J_PASSWORD: Final = _env_str("SONALITY_NEO4J_PASSWORD", "sonality_password")
NEO4J_DATABASE: Final = _env_str("SONALITY_NEO4J_DATABASE", "neo4j")

QDRANT_URL: Final = _env_str("SONALITY_QDRANT_URL", "http://localhost:6333")

# --- Embedding (local FastEmbed bge-large-en-v1.5, 1024 dims) ---
EMBEDDING_DIMENSIONS: Final = 1024
EMBEDDING_MAX_CHARS: Final = _env_int("SONALITY_EMBEDDING_MAX_CHARS", 2000)

# --- Qdrant search tuning ---
QDRANT_SEARCH_EF: Final = _env_int("SONALITY_QDRANT_SEARCH_EF", 128)
QDRANT_RESCORE_QUANTIZED: Final = _env_str("SONALITY_QDRANT_RESCORE", "true").lower() == "true"

# --- LLM for scoring/assessment tasks (fast, cheap model) ---
FAST_LLM_MODEL: Final = _env_str("SONALITY_FAST_LLM_MODEL", ESS_MODEL)
FAST_LLM_MAX_TOKENS: Final = _env_int("SONALITY_FAST_LLM_MAX_TOKENS", 1024)

# --- STM ---
STM_BUFFER_CAPACITY: Final = _env_int("SONALITY_STM_BUFFER_CAPACITY", 64000)
STM_BATCH_THRESHOLD: Final = _env_int("SONALITY_STM_BATCH_THRESHOLD", 3)
STM_MAX_BATCH_SIZE: Final = _env_int("SONALITY_STM_MAX_BATCH_SIZE", 10)
STM_POLL_INTERVAL: Final = _env_float("SONALITY_STM_POLL_INTERVAL", 30.0)

# --- Retrieval ---
RETRIEVAL_MAX_ITERATIONS: Final = _env_int("SONALITY_RETRIEVAL_MAX_ITERATIONS", 3)
RETRIEVAL_CONFIDENCE_THRESHOLD: Final = _env_float("SONALITY_RETRIEVAL_CONFIDENCE_THRESHOLD", 0.8)
RETRIEVAL_OVER_FETCH_FACTOR: Final = _env_int("SONALITY_RETRIEVAL_OVER_FETCH_FACTOR", 3)
MAX_RERANK_CANDIDATES: Final = _env_int("SONALITY_MAX_RERANK_CANDIDATES", 25)

# Per-HTTP-request timeout for LLM calls. 90s is sufficient for the chat response,
# but knowledge extraction (1500-token prompt + 512-token output) needs ~130-150s on
# a 35B model at ~4 tok/s. 180s gives a safe buffer for all task types.
# Increase further for 70B+ models or throttled endpoints.
LLM_REQUEST_TIMEOUT: Final = _env_int("SONALITY_LLM_TIMEOUT", 180)

# Timeout for async ops dispatched from the sync context via run_coroutine_threadsafe.
# Must exceed the maximum realistic duration of any single coroutine, which in the
# worst case is: semaphore_queue_wait + max_retries x LLM_REQUEST_TIMEOUT.
# Default: 5 x LLM_REQUEST_TIMEOUT covers one queued call (3 retries) + current call
# (2 retries) with fail-fast TimeoutErrors: 5 x 90 = 450 s.
ASYNC_TIMEOUT: Final = _env_int("SONALITY_ASYNC_TIMEOUT", LLM_REQUEST_TIMEOUT * 5)


def missing_live_api_config() -> tuple[str, ...]:
    """Return required live configuration keys that are currently unset.

    API key is optional for local OpenAI-compatible servers (for example, Ollama).
    """
    return ("SONALITY_BASE_URL",) if not BASE_URL.strip() else ()
