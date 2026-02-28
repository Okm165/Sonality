from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(_env_str(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env_str(name, str(default)))


def _env_path(name: str, default: Path) -> Path:
    return Path(_env_str(name, str(default)))


DATA_DIR: Final = PROJECT_ROOT / "data"
SPONGE_FILE: Final = DATA_DIR / "sponge.json"
SPONGE_HISTORY_DIR: Final = DATA_DIR / "sponge_history"
CHROMADB_DIR: Final = DATA_DIR / "chromadb"
ESS_AUDIT_LOG_FILE: Final = _env_path("SONALITY_ESS_AUDIT_LOG_FILE", DATA_DIR / "ess_log.jsonl")

API_KEY: Final = os.environ.get("SONALITY_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
# Backward-compatible alias kept for tests/integrations that still reference it.
ANTHROPIC_API_KEY: Final = API_KEY
MODEL: Final = _env_str("SONALITY_MODEL", "claude-sonnet-4-20250514")
ESS_MODEL: Final = _env_str("SONALITY_ESS_MODEL", MODEL)
LOG_LEVEL: Final = _env_str("SONALITY_LOG_LEVEL", "INFO")

ESS_THRESHOLD: Final = _env_float("SONALITY_ESS_THRESHOLD", 0.3)
SPONGE_MAX_TOKENS: Final = 500
EPISODIC_RETRIEVAL_COUNT: Final = _env_int("SONALITY_EPISODIC_RETRIEVAL_COUNT", 3)
SEMANTIC_RETRIEVAL_COUNT: Final = _env_int("SONALITY_SEMANTIC_RETRIEVAL_COUNT", 2)
# Backward-compatible aggregate count used by docs/tests.
EPISODE_RETRIEVAL_COUNT: Final = EPISODIC_RETRIEVAL_COUNT + SEMANTIC_RETRIEVAL_COUNT

OPINION_BASE_RATE: Final = 0.1
BELIEF_DECAY_RATE: Final = 0.15  # power-law exponent β (Ebbinghaus/FadeMem 2025)
BOOTSTRAP_DAMPENING_UNTIL: Final = _env_int("SONALITY_BOOTSTRAP_DAMPENING_UNTIL", 10)
OPINION_COOLING_PERIOD: Final = _env_int("SONALITY_OPINION_COOLING_PERIOD", 3)

MAX_CONVERSATION_CHARS: Final = 100_000
REFLECTION_EVERY: Final = _env_int("SONALITY_REFLECTION_EVERY", 20)
REFLECTION_SHIFT_THRESHOLD: Final = 0.1  # cumulative magnitude to trigger early reflection
