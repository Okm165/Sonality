"""Audio processing: STT/TTS via Speaches OpenAI-compatible API.

AudioProcessor wraps Speaches endpoints for speech-to-text (Whisper) and
text-to-speech (Kokoro). TTS optimization rewrites agent responses into
natural spoken prose via an LLM before synthesis. MP3→OGG Opus conversion
uses ffmpeg for Telegram voice message compatibility.
"""

from __future__ import annotations

import asyncio
import re

import httpx
import structlog

from shared.llm.caller import raw_call as _shared_raw_call
from shared.llm.provider import LLMProvider

from . import config

log = structlog.get_logger()

_provider = LLMProvider(
    config.settings.tts_optimize_base_url, config.settings.tts_optimize_api_key, int(config.settings.tts_optimize_timeout)
)


async def llm_call(prompt: str, max_tokens: int, temperature: float) -> str:
    """Async LLM completion using chat's provider."""
    return await asyncio.to_thread(
        _shared_raw_call,
        _provider,
        prompt=prompt,
        model=config.settings.tts_optimize_model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


TTS_OPTIMIZATION_PROMPT = """\
Rewrite this text for natural speech synthesis. The result should sound good \
when read aloud — flowing prose with natural pauses, no formatting artifacts \
(markdown, bullet points, headers, confidence scores, citations). Preserve \
the exact information and meaning. Vary sentence openings.

Output only the rewritten text.

Original text:
{text}"""


async def optimize_for_speech(text: str) -> str:
    """Rewrite text for natural speech synthesis using LLM."""
    if not config.settings.tts_optimize or not text.strip():
        return text

    try:
        result = await llm_call(
            prompt=TTS_OPTIMIZATION_PROMPT.format(text=text),
            max_tokens=config.settings.tts_optimize_max_tokens,
            temperature=config.settings.tts_optimize_temperature,
        )
        log.debug("tts_optimized", input_chars=len(text), output_chars=len(result))
        return result
    except Exception as e:
        log.warning("tts_optimization_failed", error=str(e))
        return text


async def mp3_to_ogg_opus(mp3_data: bytes) -> bytes:
    """Convert MP3 audio to OGG Opus format for Telegram voice messages.

    Telegram requires OGG Opus for proper voice message display with waveform.
    Uses ffmpeg subprocess for conversion.
    """
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-c:a",
        "libopus",
        "-b:a",
        "48k",
        "-vbr",
        "on",
        "-f",
        "ogg",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(mp3_data), timeout=30.0)
    except TimeoutError:
        proc.kill()
        from shared.errors import ServiceUnavailableError

        raise ServiceUnavailableError("ffmpeg conversion timed out after 30s") from None
    if proc.returncode != 0:
        from shared.errors import ServiceUnavailableError

        log.error("ffmpeg_conversion_failed", stderr=stderr.decode()[-500:])
        raise ServiceUnavailableError(f"ffmpeg failed with code {proc.returncode}")
    log.debug("mp3_to_ogg", input_bytes=len(mp3_data), output_bytes=len(stdout))
    return stdout


def chunk_text(text: str) -> list[str]:
    """Split text into chunks at sentence boundaries, respecting TTS_MAX_LENGTH."""
    limit = config.settings.tts_max_length
    if len(text) <= limit:
        log.debug("chunk_text_single", chars=len(text), limit=limit)
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > limit:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sentence), limit):
                chunks.append(sentence[i : i + limit])
        elif len(current) + len(sentence) + 1 <= limit:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    log.debug("chunk_text", chars=len(text), chunks=len(chunks), sizes=[len(c) for c in chunks])
    return chunks


class AudioProcessor:
    """STT/TTS via Speaches OpenAI-compatible API."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(base_url=config.settings.speaches_url)
        log.debug("audio_processor_init", url=config.settings.speaches_url)

    async def stt(self, audio: bytes, mime_type: str = "audio/ogg") -> str:
        """Transcribe audio to text."""
        ext = mime_type.split("/")[-1]
        log.debug("stt_request", model=config.settings.stt_model, lang=config.settings.stt_language, bytes=len(audio), mime=mime_type)
        r = await self._client.post(
            "/v1/audio/transcriptions",
            files={"file": (f"audio.{ext}", audio, mime_type)},
            data={"model": config.settings.stt_model, "language": config.settings.stt_language},
            timeout=config.settings.stt_timeout,
        )
        if r.status_code != 200:
            log.error("stt_error", status=r.status_code, body=r.text[:500])
        r.raise_for_status()
        text = str(r.json().get("text", "")).strip()
        log.info("stt_complete", input_bytes=len(audio), output_chars=len(text))
        return text

    async def tts(self, text: str) -> bytes:
        """Synthesize text to audio (format from config, default mp3)."""
        payload = {
            "model": config.settings.tts_model,
            "voice": config.settings.tts_voice,
            "input": text,
            "response_format": config.settings.tts_format,
            "speed": config.settings.tts_speed,
        }
        log.debug("tts_request", model=config.settings.tts_model, voice=config.settings.tts_voice, format=config.settings.tts_format, chars=len(text))
        r = await self._client.post("/v1/audio/speech", json=payload, timeout=config.settings.tts_timeout)
        if r.status_code != 200:
            log.error("tts_error", status=r.status_code, body=r.text[:500])
        r.raise_for_status()
        log.info("tts_complete", input_chars=len(text), output_bytes=len(r.content), format=config.settings.tts_format)
        return r.content

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AudioProcessor:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
