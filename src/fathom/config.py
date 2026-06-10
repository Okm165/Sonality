"""Fathom settings — loaded from FATHOM_* environment variables.

Inherits infrastructure fields from ``shared.config.InfraSettings`` (Neo4j,
Qdrant, embedding, LLM).  Adds fathom-specific browser, search, and
exploration settings.  Symmetric with ``sonality.config``.
"""

from __future__ import annotations

from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import SettingsConfigDict

from shared.config import PROJECT_ROOT, InfraSettings

load_dotenv(PROJECT_ROOT / ".env", override=False)


class Settings(InfraSettings):
    model_config = SettingsConfigDict(env_prefix="FATHOM_", env_file=".env", extra="ignore")

    # --- Browser ---
    browser_concurrency: int = 4
    sample_temperature: float = 2.0
    browser_ws_url: str = ""

    # --- LLM ---
    llm_max_tokens: int = -1
    """Output token limit for Fathom structured calls.  Default -1 omits from
    payload, letting llama.cpp apply its default.  Structured JSON calls
    naturally terminate when the schema is complete."""

    # Must accommodate worst-case queueing behind --parallel 2 llama.cpp.
    # 10 pages x ~60s each = 600s worst case when concurrent sessions compete.
    llm_request_timeout: int = 3600

    # --- Qdrant ---
    qdrant_search_ef: int = 128

    # --- Search ---
    exploration_divergence: float = 0.2
    search_concurrency: int = 16

    # --- Session ---
    session_timeout: int = 7200
    """Max session duration in seconds before marking as timed out."""

    @model_validator(mode="after")
    def _clamp_fathom_bounds(self) -> Settings:
        self.browser_concurrency = max(1, self.browser_concurrency)
        self.sample_temperature = max(0.01, self.sample_temperature)
        self.exploration_divergence = max(0.0, min(1.0, self.exploration_divergence))
        self.search_concurrency = max(1, self.search_concurrency)
        self.session_timeout = max(10, self.session_timeout)
        self.llm_request_timeout = max(10, self.llm_request_timeout)
        return self


settings = Settings()
