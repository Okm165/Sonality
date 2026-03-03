from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Final, Literal

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
    """Read an environment variable as boolean with conservative parsing."""
    raw = _env_str(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


ApiVariant = Literal["anthropic", "openrouter"]
OpenRouterDataCollection = Literal["allow", "deny"]

_API_VARIANTS: Final[dict[str, ApiVariant]] = {
    "anthropic": "anthropic",
    "openrouter": "openrouter",
}
API_BASE_URL_BY_VARIANT: Final[dict[ApiVariant, str]] = {
    "anthropic": "https://api.anthropic.com",
    "openrouter": "https://openrouter.ai/api",
}
_OPENROUTER_DATA_COLLECTIONS: Final[dict[str, OpenRouterDataCollection]] = {
    "allow": "allow",
    "deny": "deny",
}
OPENROUTER_ANTHROPIC_MODEL_ALIASES: Final[dict[str, str]] = {
    "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4",
    "claude-3-5-haiku-20241022": "anthropic/claude-3.5-haiku",
    "claude-3-7-sonnet-20250219": "anthropic/claude-3.7-sonnet",
}
_OPENROUTER_ANTHROPIC_DATE_SUFFIX_RE: Final = re.compile(r"-\d{8}$")
DEFAULT_MODEL_BY_VARIANT: Final[dict[ApiVariant, str]] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openrouter": "anthropic/claude-sonnet-4",
}
DEFAULT_ESS_MODEL_BY_VARIANT: Final[dict[ApiVariant, str]] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openrouter": "anthropic/claude-3.7-sonnet",
}


def _normalize_openrouter_model_id(model_id: str) -> str:
    """Normalize Anthropic-style model IDs into OpenRouter-prefixed IDs."""
    if "/" in model_id:
        return model_id
    if (aliased := OPENROUTER_ANTHROPIC_MODEL_ALIASES.get(model_id)) is not None:
        return aliased
    if not model_id.startswith("claude-"):
        return model_id
    normalized = _OPENROUTER_ANTHROPIC_DATE_SUFFIX_RE.sub("", model_id)
    normalized = normalized.replace("claude-3-5-", "claude-3.5-")
    normalized = normalized.replace("claude-3-7-", "claude-3.7-")
    return f"anthropic/{normalized}"


DATA_DIR: Final = PROJECT_ROOT / "data"
SPONGE_FILE: Final = DATA_DIR / "sponge.json"
SPONGE_HISTORY_DIR: Final = DATA_DIR / "sponge_history"
CHROMADB_DIR: Final = DATA_DIR / "chromadb"
ESS_AUDIT_LOG_FILE: Final = DATA_DIR / "ess_log.jsonl"
API_KEY: Final = os.environ.get("SONALITY_API_KEY")
_api_variant_raw = _env_str("SONALITY_API_VARIANT", "").strip().lower()
if not _api_variant_raw:
    _api_variant: ApiVariant = "anthropic"
elif (resolved_api_variant := _API_VARIANTS.get(_api_variant_raw)) is not None:
    _api_variant = resolved_api_variant
else:
    expected = ", ".join(sorted(_API_VARIANTS))
    raise ValueError(f"SONALITY_API_VARIANT must be one of: {expected}")
API_VARIANT: Final[ApiVariant] = _api_variant
BASE_URL: Final = API_BASE_URL_BY_VARIANT[API_VARIANT]
_raw_model = _env_str("SONALITY_MODEL", DEFAULT_MODEL_BY_VARIANT[API_VARIANT])
_raw_ess_model = _env_str("SONALITY_ESS_MODEL", DEFAULT_ESS_MODEL_BY_VARIANT[API_VARIANT])
MODEL: Final = (
    _raw_model if API_VARIANT == "anthropic" else _normalize_openrouter_model_id(_raw_model)
)
ESS_MODEL: Final = (
    _raw_ess_model if API_VARIANT == "anthropic" else _normalize_openrouter_model_id(_raw_ess_model)
)
OPENROUTER_PROVIDER_ORDER: Final[tuple[str, ...]] = tuple(
    slug.strip()
    for slug in _env_str(
        "SONALITY_OPENROUTER_PROVIDER_ORDER", "google-vertex,amazon-bedrock"
    ).split(",")
    if slug.strip()
)
OPENROUTER_ALLOW_FALLBACKS: Final = _env_bool("SONALITY_OPENROUTER_ALLOW_FALLBACKS", True)
OPENROUTER_FORCE_ZDR: Final = _env_bool("SONALITY_OPENROUTER_FORCE_ZDR", True)
_openrouter_data_collection_raw = (
    _env_str("SONALITY_OPENROUTER_DATA_COLLECTION", "").strip().lower()
)
if (
    _openrouter_data_collection_raw
    and _openrouter_data_collection_raw not in _OPENROUTER_DATA_COLLECTIONS
):
    expected = ", ".join(sorted(_OPENROUTER_DATA_COLLECTIONS))
    raise ValueError(f"SONALITY_OPENROUTER_DATA_COLLECTION must be one of: {expected}")
OPENROUTER_DATA_COLLECTION: Final[OpenRouterDataCollection | None] = (
    _OPENROUTER_DATA_COLLECTIONS.get(_openrouter_data_collection_raw)
)
LOG_LEVEL: Final = _env_str("SONALITY_LOG_LEVEL", "INFO")

ESS_THRESHOLD: Final = _env_float("SONALITY_ESS_THRESHOLD", 0.3)
SPONGE_MAX_TOKENS: Final = 500
EPISODIC_RETRIEVAL_COUNT: Final = _env_int("SONALITY_EPISODIC_RETRIEVAL_COUNT", 3)
SEMANTIC_RETRIEVAL_COUNT: Final = _env_int("SONALITY_SEMANTIC_RETRIEVAL_COUNT", 2)

OPINION_BASE_RATE: Final = 0.1
BELIEF_DECAY_RATE: Final = 0.15  # power-law exponent β (Ebbinghaus/FadeMem 2025)
BOOTSTRAP_DAMPENING_UNTIL: Final = _env_int("SONALITY_BOOTSTRAP_DAMPENING_UNTIL", 10)
OPINION_COOLING_PERIOD: Final = _env_int("SONALITY_OPINION_COOLING_PERIOD", 3)

MAX_CONVERSATION_CHARS: Final = 100_000
REFLECTION_EVERY: Final = _env_int("SONALITY_REFLECTION_EVERY", 20)
REFLECTION_SHIFT_THRESHOLD: Final = 0.1  # cumulative magnitude to trigger early reflection


def anthropic_client_kwargs() -> dict[str, str]:
    """Build Anthropic SDK kwargs from explicit live API config."""
    if not API_KEY:
        raise ValueError("Missing required API config: SONALITY_API_KEY")
    return {"api_key": API_KEY, "base_url": BASE_URL}


def missing_live_api_config() -> tuple[str, ...]:
    """Return required live configuration keys that are currently unset."""
    missing: list[str] = []
    if not API_KEY:
        missing.append("SONALITY_API_KEY")
    if not _api_variant_raw:
        missing.append("SONALITY_API_VARIANT")
    return tuple(missing)
