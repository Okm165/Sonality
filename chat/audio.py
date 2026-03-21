"""Audio processing: STT/TTS via Speaches OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import logging
import re

import httpx

from . import config
from .llm import llm_call

log = logging.getLogger(__name__)

TTS_OPTIMIZATION_PROMPT = """\
Rewrite the following text for natural speech synthesis.

Requirements:
- Remove all formatting artifacts (markdown, bullet points, headers)
- Remove confidence scores and citation markers
- Convert lists into flowing prose with natural transitions
- Keep the exact same information and meaning
- Use conversational phrasing that sounds good when read aloud
- Add natural pauses with commas where appropriate
- Avoid starting sentences with "I" repeatedly

Output ONLY the rewritten text. No preamble, no explanation, no quotes.

Original text:
{text}"""


async def optimize_for_speech(text: str) -> str:
    """Rewrite text for natural speech synthesis using LLM."""
    if not config.TTS_OPTIMIZE_ENABLED or not text.strip():
        return text

    try:
        result = await llm_call(
            prompt=TTS_OPTIMIZATION_PROMPT.format(text=text),
            max_tokens=config.TTS_OPTIMIZE_MAX_TOKENS,
            temperature=config.TTS_OPTIMIZE_TEMPERATURE,
        )
        log.debug("TTS optimized: %d -> %d chars", len(text), len(result))
        return result
    except Exception as e:
        log.warning("TTS optimization failed, using original: %s", e)
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
    stdout, stderr = await proc.communicate(mp3_data)
    if proc.returncode != 0:
        log.error("ffmpeg conversion failed: %s", stderr.decode()[-500:])
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")
    log.debug("Converted MP3 %d bytes -> OGG Opus %d bytes", len(mp3_data), len(stdout))
    return stdout


def chunk_text(text: str) -> list[str]:
    """Split text into chunks at sentence boundaries, respecting TTS_MAX_LENGTH."""
    limit = config.TTS_MAX_LENGTH
    if len(text) <= limit:
        log.debug("Text fits in single chunk: %d chars (limit=%d)", len(text), limit)
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
    log.debug("Chunked %d chars -> %d chunks: %s", len(text), len(chunks), [len(c) for c in chunks])
    return chunks


class AudioProcessor:
    """STT/TTS via Speaches OpenAI-compatible API."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(base_url=config.SPEACHES_URL)
        log.debug("AudioProcessor: url=%s", config.SPEACHES_URL)

    async def stt(self, audio: bytes, mime_type: str = "audio/ogg") -> str:
        """Transcribe audio to text."""
        ext = mime_type.split("/")[-1]
        log.debug(
            "STT request: model=%s lang=%s bytes=%d mime=%s",
            config.STT_MODEL,
            config.STT_LANGUAGE,
            len(audio),
            mime_type,
        )
        r = await self._client.post(
            "/v1/audio/transcriptions",
            files={"file": (f"audio.{ext}", audio, mime_type)},
            data={"model": config.STT_MODEL, "language": config.STT_LANGUAGE},
            timeout=config.STT_TIMEOUT,
        )
        if r.status_code != 200:
            log.error("STT error: status=%d body=%s", r.status_code, r.text[:500])
        r.raise_for_status()
        text = str(r.json().get("text", "")).strip()
        log.info("STT: %d bytes -> %d chars", len(audio), len(text))
        return text

    async def tts(self, text: str) -> bytes:
        """Synthesize text to audio (format from config, default mp3)."""
        payload = {
            "model": config.TTS_MODEL,
            "voice": config.TTS_VOICE,
            "input": text,
            "response_format": config.TTS_FORMAT,
            "speed": config.TTS_SPEED,
        }
        log.debug(
            "TTS request: model=%s voice=%s format=%s len=%d",
            config.TTS_MODEL,
            config.TTS_VOICE,
            config.TTS_FORMAT,
            len(text),
        )
        r = await self._client.post("/v1/audio/speech", json=payload, timeout=config.TTS_TIMEOUT)
        if r.status_code != 200:
            log.error("TTS error: status=%d body=%s", r.status_code, r.text[:500])
        r.raise_for_status()
        log.info("TTS: %d chars -> %d bytes (%s)", len(text), len(r.content), config.TTS_FORMAT)
        return r.content

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AudioProcessor:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
