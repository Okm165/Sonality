"""Sonality settings — loaded from SONALITY_* environment variables.

Each module owns its config. Missing required values fail fast at startup.
Access via ``from sonality.config import settings`` — symmetric with fathom.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings

from shared.config import PROJECT_ROOT
from shared.errors import ConfigError

load_dotenv(PROJECT_ROOT / ".env", override=False)


class Settings(BaseSettings):
    model_config = {"env_prefix": "SONALITY_", "env_file": ".env", "extra": "ignore"}

    # --- Core LLM endpoint ---
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    log_level: str = "INFO"

    # --- Model tiers ---
    agent_model: str = ""
    reasoning_model: str = ""
    structured_model: str = ""
    fast_model: str = ""

    # --- Database connections ---
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "sonality_password"
    neo4j_database: str = "neo4j"
    neo4j_max_pool_size: int = 50
    neo4j_connection_timeout: float = 30.0
    qdrant_url: str = "http://localhost:6333"

    # --- Embedding ---
    embedding_dimensions: int = 1024
    embedding_max_chars: int = 4096
    embedding_cache_size: int = 10000

    # --- Qdrant search tuning ---
    qdrant_search_ef: int = 128
    qdrant_rescore: bool = True

    # --- LLM generation ---
    agent_temperature: float = 0.6
    llm_max_tokens: int = 8192

    # --- Agent loop ---
    agent_loop_hard_ceiling: int = 12
    agent_compress_threshold: int = 12
    agent_compress_keep_tail: int = 4

    # --- Limits and retry ---
    episode_content_limit: int = 300
    belief_prompt_window: int = 15
    ess_timeout: int = 300
    forgetting_candidate_limit: int = 10
    http_api_key: str = ""
    llm_max_retries: int = 3
    llm_backoff_base: float = 2.0
    llm_concurrency: int = 1
    chat_input_token_budget: int = 8192
    llm_timeout: int = 600

    # --- Retrieval pipeline ---
    retrieval_max_iterations: int = 3
    retrieval_over_fetch_factor: int = 3
    max_rerank_candidates: int = 50

    # --- Web access ---
    fathom_url: str = "http://localhost:8010"
    web_search_enabled: bool = True
    reflection_web_queries: int = 3

    @model_validator(mode="after")
    def _resolve_and_validate(self) -> Settings:
        """Validate required fields, fill model tiers from base model."""
        if not self.base_url:
            raise ConfigError("SONALITY_BASE_URL must be set")
        if not self.model:
            raise ConfigError("SONALITY_MODEL must be set")
        for attr in ("agent_model", "reasoning_model", "structured_model", "fast_model"):
            if not getattr(self, attr):
                setattr(self, attr, self.model)
        self.llm_concurrency = max(1, self.llm_concurrency)
        if not self.http_api_key:
            self.http_api_key = os.environ.get("SONALITY_HTTP_API_KEY", "")
        return self

    @property
    def async_timeout(self) -> int:
        return self.llm_timeout * 5


settings = Settings()

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"


def missing_live_api_config() -> tuple[str, ...]:
    """Return required configuration keys that are unset."""
    return ("SONALITY_BASE_URL",) if not settings.base_url.strip() else ()
