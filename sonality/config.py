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

BOOTSTRAP_DAMPENING_UNTIL: Final = _env_int("SONALITY_BOOTSTRAP_DAMPENING_UNTIL", 10)
OPINION_COOLING_PERIOD: Final = _env_int("SONALITY_OPINION_COOLING_PERIOD", 3)

MAX_CONVERSATION_CHARS: Final = 100_000
REFLECTION_EVERY: Final = _env_int("SONALITY_REFLECTION_EVERY", 20)

# --- Database (Neo4j + PostgreSQL) ---
NEO4J_URL: Final = _env_str("SONALITY_NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER: Final = _env_str("SONALITY_NEO4J_USER", "neo4j")
NEO4J_PASSWORD: Final = _env_str("SONALITY_NEO4J_PASSWORD", "sonality_password")
NEO4J_DATABASE: Final = _env_str("SONALITY_NEO4J_DATABASE", "neo4j")

POSTGRES_URL: Final = _env_str(
    "SONALITY_POSTGRES_URL",
    "postgresql://sonality:sonality_password@localhost:5432/sonality",
)
PG_POOL_MIN_SIZE: Final = _env_int("SONALITY_PG_POOL_MIN_SIZE", 2)
PG_POOL_MAX_SIZE: Final = _env_int("SONALITY_PG_POOL_MAX_SIZE", 10)

# --- Embedding (can use separate endpoint and API key from chat models) ---
EMBEDDING_BASE_URL: Final = _env_str("SONALITY_EMBEDDING_BASE_URL", "")
EMBEDDING_API_KEY: Final = _env_str("SONALITY_EMBEDDING_API_KEY", "")
EMBEDDING_MODEL: Final = _env_str("SONALITY_EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIMENSIONS: Final = _env_int("SONALITY_EMBEDDING_DIMENSIONS", 768)
EMBEDDING_BATCH_SIZE: Final = _env_int("SONALITY_EMBEDDING_BATCH_SIZE", 32)
# Set to false for Ollama models that don't accept a dimensions truncation parameter
EMBEDDING_SEND_DIMENSIONS: Final = _env_str("SONALITY_EMBEDDING_SEND_DIMENSIONS", "true").lower() not in ("false", "0", "no")
EMBEDDING_QUERY_INSTRUCTION: Final = _env_str(
    "SONALITY_EMBEDDING_QUERY_INSTRUCTION",
    "Represent this memory retrieval query for finding relevant past conversations:",
)
EMBEDDING_DOC_INSTRUCTION: Final = _env_str(
    "SONALITY_EMBEDDING_DOC_INSTRUCTION",
    "Represent this conversation memory for semantic retrieval:",
)

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


def missing_live_api_config() -> tuple[str, ...]:
    """Return required live configuration keys that are currently unset.

    API key is optional for local OpenAI-compatible servers (for example, Ollama).
    """
    missing: list[str] = []
    if not BASE_URL.strip():
        missing.append("SONALITY_BASE_URL")
    return tuple(missing)
