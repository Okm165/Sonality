"""Chat module configuration — imports helpers from sonality.config."""

from __future__ import annotations

from typing import Final

import sonality.config as sonality_config
from sonality.config import _env_bool, _env_float, _env_int, _env_str

# --- Telegram ---
TELEGRAM_TOKEN: Final = _env_str("CHAT_TELEGRAM_TOKEN", "")

# --- Service URLs ---
SONALITY_URL: Final = _env_str("CHAT_SONALITY_URL", "http://localhost:8000")
SPEACHES_URL: Final = _env_str("CHAT_SPEACHES_URL", "http://localhost:8001")

# --- STT ---
STT_MODEL: Final = _env_str("CHAT_STT_MODEL", "Systran/faster-distil-whisper-large-v3")
STT_LANGUAGE: Final = _env_str("CHAT_STT_LANGUAGE", "en")
STT_TIMEOUT: Final = _env_float("CHAT_STT_TIMEOUT", 60.0)

# --- TTS ---
TTS_MODEL: Final = _env_str("CHAT_TTS_MODEL", "speaches-ai/Kokoro-82M-v1.0-ONNX")
TTS_VOICE: Final = _env_str("CHAT_TTS_VOICE", "af_heart")
TTS_SPEED: Final = _env_float("CHAT_TTS_SPEED", 1.0)
TTS_TIMEOUT: Final = _env_float("CHAT_TTS_TIMEOUT", 180.0)
TTS_MAX_LENGTH: Final = _env_int("CHAT_TTS_MAX_LENGTH", 4096)
TTS_FORMAT: Final = _env_str("CHAT_TTS_FORMAT", "mp3")

# --- TTS Optimization (LLM rewrite for natural speech) ---
TTS_OPTIMIZE_ENABLED: Final = _env_bool("CHAT_TTS_OPTIMIZE", True)
TTS_OPTIMIZE_BASE_URL: Final = _env_str("CHAT_TTS_OPTIMIZE_BASE_URL", sonality_config.BASE_URL)
TTS_OPTIMIZE_API_KEY: Final = _env_str("CHAT_TTS_OPTIMIZE_API_KEY", sonality_config.API_KEY)
TTS_OPTIMIZE_MODEL: Final = _env_str("CHAT_TTS_OPTIMIZE_MODEL", sonality_config.MODEL)
TTS_OPTIMIZE_MAX_TOKENS: Final = _env_int("CHAT_TTS_OPTIMIZE_MAX_TOKENS", 8192)
TTS_OPTIMIZE_TEMPERATURE: Final = _env_float("CHAT_TTS_OPTIMIZE_TEMPERATURE", 0.3)
TTS_OPTIMIZE_TIMEOUT: Final = _env_float("CHAT_TTS_OPTIMIZE_TIMEOUT", 60.0)

# --- General ---
HTTP_TIMEOUT: Final = _env_float("CHAT_HTTP_TIMEOUT", 512.0)
LOG_LEVEL: Final = _env_str("CHAT_LOG_LEVEL", "DEBUG")
