"""Chat module configuration — loaded from CHAT_* environment variables.

Symmetric with sonality/config.py and fathom/config.py — all use pydantic-settings.
TTS optimization defaults to the core Sonality LLM endpoint unless overridden.
"""

from __future__ import annotations

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

import sonality.config as sonality_config
from shared.config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env", override=False)


class Settings(BaseSettings):
    model_config = {"env_prefix": "CHAT_", "env_file": ".env", "extra": "ignore"}

    # Telegram
    telegram_token: str = ""

    # Service URLs
    sonality_url: str = "http://localhost:8000"
    speaches_url: str = "http://localhost:8020"

    # STT
    stt_model: str = "Systran/faster-distil-whisper-large-v3"
    stt_language: str = "en"
    stt_timeout: float = 60.0

    # TTS
    tts_model: str = "speaches-ai/Kokoro-82M-v1.0-ONNX"
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    tts_timeout: float = 180.0
    tts_max_length: int = 4096
    tts_format: str = "mp3"

    # TTS optimization — LLM rewrites text for natural speech
    tts_optimize: bool = True
    tts_optimize_base_url: str = ""
    tts_optimize_api_key: str = ""
    tts_optimize_model: str = ""
    tts_optimize_max_tokens: int = 8192
    tts_optimize_temperature: float = 0.3
    tts_optimize_timeout: float = 60.0

    # HTTP client
    model_id: str = "sonality"
    http_timeout: float = 512.0
    log_level: str = "INFO"


settings = Settings()

# Fill TTS optimization defaults from sonality if not overridden
if not settings.tts_optimize_base_url:
    settings.tts_optimize_base_url = sonality_config.settings.base_url
if not settings.tts_optimize_api_key:
    settings.tts_optimize_api_key = sonality_config.settings.api_key
if not settings.tts_optimize_model:
    settings.tts_optimize_model = sonality_config.settings.model
