"""Sonality settings — loaded from SONALITY_* environment variables.

Inherits infrastructure fields from ``shared.config.InfraSettings`` (Neo4j,
Qdrant, embedding, LLM).  Adds sonality-specific model tiers, agent loop,
retrieval, and web access settings.  Symmetric with ``fathom.config``.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import SettingsConfigDict

from shared.config import PROJECT_ROOT, InfraSettings

load_dotenv(PROJECT_ROOT / ".env", override=False)


class Settings(InfraSettings):
    model_config = SettingsConfigDict(env_prefix="SONALITY_", env_file=".env", extra="ignore")

    # --- Model tiers ---
    agent_model: str = ""
    reasoning_model: str = ""
    structured_model: str = ""
    fast_model: str = ""

    # --- Qdrant search tuning ---
    qdrant_search_ef: int = 128

    # --- LLM generation ---
    agent_temperature: float = 0.6
    ingest_temperature: float = 0.3

    # --- Agent loop ---
    agent_loop_hard_ceiling: int = 12
    max_stalls: int = 3
    max_extends: int = 4

    # --- Limits ---
    episode_content_limit: int = 300
    belief_prompt_window: int = 15
    forgetting_candidate_limit: int = 10
    http_api_key: str = ""
    chat_input_token_budget: int = 150_000
    llm_timeout: int = 3600
    context_char_limit: int = 80_000

    # --- Retrieval pipeline ---
    retrieval_over_fetch_factor: int = 3
    max_rerank_candidates: int = 50

    # --- Web access ---
    fathom_url: str = "http://localhost:8010"
    reflection_web_queries: int = 3

    @model_validator(mode="after")
    def _resolve_sonality(self) -> Settings:
        """Fill model tiers from base model, resolve http_api_key, clamp bounds."""
        for attr in ("agent_model", "reasoning_model", "structured_model", "fast_model"):
            if not getattr(self, attr):
                setattr(self, attr, self.model)
        if not self.http_api_key:
            self.http_api_key = os.environ.get("SONALITY_HTTP_API_KEY", "")
        self.agent_loop_hard_ceiling = max(1, self.agent_loop_hard_ceiling)
        self.context_char_limit = max(4_000, self.context_char_limit)
        self.max_stalls = max(1, self.max_stalls)
        self.max_extends = max(1, self.max_extends)
        self.llm_timeout = max(10, self.llm_timeout)
        self.agent_temperature = max(0.0, self.agent_temperature)
        self.qdrant_search_ef = max(1, self.qdrant_search_ef)
        self.episode_content_limit = max(1, self.episode_content_limit)
        self.belief_prompt_window = max(1, self.belief_prompt_window)
        self.forgetting_candidate_limit = max(0, self.forgetting_candidate_limit)
        self.retrieval_over_fetch_factor = max(1, self.retrieval_over_fetch_factor)
        self.max_rerank_candidates = max(1, self.max_rerank_candidates)
        self.chat_input_token_budget = max(1000, self.chat_input_token_budget)
        self.reflection_web_queries = max(0, self.reflection_web_queries)
        return self

    @property
    def async_timeout(self) -> int:
        return self.llm_timeout * 5


settings = Settings()

# Backward-compat aliases (used by Makefile)
BASE_URL = settings.base_url
MODEL = settings.model
STRUCTURED_MODEL = settings.structured_model


def missing_live_api_config() -> tuple[str, ...]:
    """Return required configuration keys that are unset."""
    return ("SONALITY_BASE_URL",) if not settings.base_url.strip() else ()
