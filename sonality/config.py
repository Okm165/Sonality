from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR: Final = PROJECT_ROOT / "data"
SPONGE_FILE: Final = DATA_DIR / "sponge.json"
SPONGE_HISTORY_DIR: Final = DATA_DIR / "sponge_history"
CHROMADB_DIR: Final = DATA_DIR / "chromadb"

ANTHROPIC_API_KEY: Final = os.environ.get("ANTHROPIC_API_KEY")
MODEL: Final = os.environ.get("SONALITY_MODEL", "claude-sonnet-4-20250514")
ESS_MODEL: Final = os.environ.get("SONALITY_ESS_MODEL", MODEL)
LOG_LEVEL: Final = os.environ.get("SONALITY_LOG_LEVEL", "INFO")

ESS_THRESHOLD: Final = float(os.environ.get("SONALITY_ESS_THRESHOLD", "0.3"))
SPONGE_MAX_TOKENS: Final = 500
EPISODE_RETRIEVAL_COUNT: Final = 5

OPINION_BASE_RATE: Final = 0.1
BELIEF_DECAY_RATE: Final = 0.15  # power-law exponent β (Ebbinghaus/FadeMem 2025)
BOOTSTRAP_DAMPENING_UNTIL: Final = int(os.environ.get("SONALITY_BOOTSTRAP_DAMPENING_UNTIL", "10"))

MAX_CONVERSATION_CHARS: Final = 100_000
REFLECTION_EVERY: Final = int(os.environ.get("SONALITY_REFLECTION_EVERY", "20"))
REFLECTION_SHIFT_THRESHOLD: Final = 0.1  # cumulative magnitude to trigger early reflection
