"""Fathom settings — loaded from FATHOM_* environment variables.

Each module owns its config. No fallbacks to SONALITY_* or OPENAI_*.
Missing required values fail fast at startup.
"""

from __future__ import annotations

from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings

from shared.config import PROJECT_ROOT
from shared.errors import ConfigError

load_dotenv(PROJECT_ROOT / ".env", override=False)


class Settings(BaseSettings):
    model_config = {"env_prefix": "FATHOM_", "env_file": ".env", "extra": "ignore"}

    n: int = 4
    max_pages: int = 80
    max_urls_for_scoring: int = 15
    stall_limit: int = 3
    browser_ws_url: str = ""

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    llm_max_tokens: int = 4096
    llm_max_retries: int = 3
    llm_backoff_base: float = 2.0
    llm_concurrency: int = 1
    llm_request_timeout: int = 600

    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "sonality_password"
    neo4j_database: str = "neo4j"
    neo4j_max_pool_size: int = 50
    neo4j_connection_timeout: float = 30.0

    search_concurrency: int = 8
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _validate_required(self) -> Settings:
        """Fail fast if required LLM config is missing."""
        if not self.base_url:
            raise ConfigError("FATHOM_BASE_URL must be set")
        if not self.model:
            raise ConfigError("FATHOM_MODEL must be set")
        return self


settings = Settings()
