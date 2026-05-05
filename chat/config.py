"""Chat module configuration — env vars for Telegram bot, STT/TTS, and HTTP client.

All variables use the ``CHAT_`` prefix to avoid collisions with the core
``SONALITY_`` namespace. TTS optimization variables default to the core LLM
endpoint so voice responses use the same provider unless overridden.

See .env.example for full documentation of each variable.
"""

from __future__ import annotations

from typing import Final

import sonality.config as sonality_config
from sonality.config import env_bool, env_float, env_int, env_str

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN: Final = env_str("CHAT_TELEGRAM_TOKEN", "")

# ---------------------------------------------------------------------------
# Service URLs — where the chat client connects
# ---------------------------------------------------------------------------
SONALITY_URL: Final = env_str("CHAT_SONALITY_URL", "http://localhost:8000")
SPEACHES_URL: Final = env_str("CHAT_SPEACHES_URL", "http://localhost:8001")

# ---------------------------------------------------------------------------
# STT (Speech-to-Text via Speaches Whisper API)
# ---------------------------------------------------------------------------
STT_MODEL: Final = env_str("CHAT_STT_MODEL", "Systran/faster-distil-whisper-large-v3")
STT_LANGUAGE: Final = env_str("CHAT_STT_LANGUAGE", "en")
STT_TIMEOUT: Final = env_float("CHAT_STT_TIMEOUT", 60.0)

# ---------------------------------------------------------------------------
# TTS (Text-to-Speech via Speaches Kokoro API)
# ---------------------------------------------------------------------------
TTS_MODEL: Final = env_str("CHAT_TTS_MODEL", "speaches-ai/Kokoro-82M-v1.0-ONNX")
TTS_VOICE: Final = env_str("CHAT_TTS_VOICE", "af_heart")
TTS_SPEED: Final = env_float("CHAT_TTS_SPEED", 1.0)
TTS_TIMEOUT: Final = env_float("CHAT_TTS_TIMEOUT", 180.0)
TTS_MAX_LENGTH: Final = env_int("CHAT_TTS_MAX_LENGTH", 4096)
TTS_FORMAT: Final = env_str("CHAT_TTS_FORMAT", "mp3")

# ---------------------------------------------------------------------------
# TTS optimization — LLM rewrites text for natural speech before synthesis.
# Defaults to the core Sonality LLM endpoint unless overridden.
# ---------------------------------------------------------------------------
TTS_OPTIMIZE_ENABLED: Final = env_bool("CHAT_TTS_OPTIMIZE", True)
TTS_OPTIMIZE_BASE_URL: Final = env_str("CHAT_TTS_OPTIMIZE_BASE_URL", sonality_config.BASE_URL)
TTS_OPTIMIZE_API_KEY: Final = env_str("CHAT_TTS_OPTIMIZE_API_KEY", sonality_config.API_KEY)
TTS_OPTIMIZE_MODEL: Final = env_str("CHAT_TTS_OPTIMIZE_MODEL", sonality_config.MODEL)
TTS_OPTIMIZE_MAX_TOKENS: Final = env_int("CHAT_TTS_OPTIMIZE_MAX_TOKENS", 8192)
TTS_OPTIMIZE_TEMPERATURE: Final = env_float("CHAT_TTS_OPTIMIZE_TEMPERATURE", 0.3)
TTS_OPTIMIZE_TIMEOUT: Final = env_float("CHAT_TTS_OPTIMIZE_TIMEOUT", 60.0)

# ---------------------------------------------------------------------------
# General HTTP client settings
# ---------------------------------------------------------------------------
MODEL_ID: Final = env_str("CHAT_MODEL_ID", "sonality")
HTTP_TIMEOUT: Final = env_float("CHAT_HTTP_TIMEOUT", 512.0)
LOG_LEVEL: Final = env_str("CHAT_LOG_LEVEL", "INFO")
