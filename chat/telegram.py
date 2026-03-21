"""Telegram bot interface for Sonality with STT/TTS support."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from io import BytesIO

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, Message

from . import config
from .audio import AudioProcessor, chunk_text, mp3_to_ogg_opus, optimize_for_speech
from .client import SonalityClient

log = logging.getLogger(__name__)
router = Router()


def _uid(msg: Message) -> int:
    return msg.from_user.id if msg.from_user else 0


async def _generate_voices(audio: AudioProcessor, chunks: list[str]) -> list[bytes | None]:
    """Generate OGG voice data for all chunks in parallel."""

    async def gen(chunk: str, idx: int) -> tuple[int, bytes | None]:
        try:
            mp3 = await audio.tts(chunk)
            ogg = await mp3_to_ogg_opus(mp3)
            return idx, ogg
        except Exception as e:
            log.warning("Voice generation failed chunk=%d: %s", idx, e)
            return idx, None

    results = await asyncio.gather(*[gen(c, i) for i, c in enumerate(chunks)])
    return [data for _, data in sorted(results)]


async def _send_text_reply(msg: Message, text: str, audio: AudioProcessor) -> None:
    """Reply to text message: alternate text chunk + voice chunk."""
    chunks = chunk_text(text)
    if not chunks:
        return

    voices = await _generate_voices(audio, chunks)
    for i, (chunk, ogg) in enumerate(zip(chunks, voices, strict=True)):
        await msg.answer(chunk)
        if ogg:
            try:
                await msg.answer_voice(BufferedInputFile(ogg, f"voice_{i}.ogg"))
            except Exception as e:
                log.warning("Voice send failed chunk=%d: %s", i, e)


async def _send_voice_reply(msg: Message, text: str, audio: AudioProcessor) -> None:
    """Reply to voice message: display original text, then optimized voice."""
    display_chunks = chunk_text(text)
    if not display_chunks:
        return

    # Optimize for speech and generate voices
    tts_text = await optimize_for_speech(text)
    tts_chunks = chunk_text(tts_text)
    voices = await _generate_voices(audio, tts_chunks)

    # Send all text first
    for chunk in display_chunks:
        await msg.answer(chunk)

    # Then all voice
    for i, ogg in enumerate(voices):
        if ogg:
            try:
                await msg.answer_voice(BufferedInputFile(ogg, f"voice_{i}.ogg"))
            except Exception as e:
                log.warning("Voice send failed chunk=%d: %s", i, e)


@router.message(CommandStart())
async def cmd_start(msg: Message) -> None:
    await msg.answer(
        "<b>Sonality</b> — Send text or voice to chat. /help for commands.", parse_mode="HTML"
    )


@router.message(Command("help"))
async def cmd_help(msg: Message) -> None:
    await msg.answer(
        "<b>Commands:</b> /start /help /beliefs /health\n<b>Chat:</b> Send text or voice message",
        parse_mode="HTML",
    )


@router.message(Command("beliefs"))
async def cmd_beliefs(msg: Message, client: SonalityClient) -> None:
    try:
        beliefs = await client.beliefs()
        if not beliefs:
            await msg.answer("No beliefs yet.")
            return
        lines = [
            f"• <b>{b.topic}</b>: {b.position:+.2f} ({b.confidence:.0%})" for b in beliefs[:15]
        ]
        await msg.answer("<b>Beliefs:</b>\n" + "\n".join(lines), parse_mode="HTML")
    except Exception as e:
        log.error("beliefs error uid=%d: %s", _uid(msg), e)
        await msg.answer(f"Error: {e}")


@router.message(Command("health"))
async def cmd_health(msg: Message, client: SonalityClient) -> None:
    try:
        h = await client.health()
        await msg.answer(
            f"<b>Health:</b> v{h.version} | {h.interaction_count} interactions | "
            f"{h.belief_count} beliefs | {h.staged_updates} staged",
            parse_mode="HTML",
        )
    except Exception as e:
        log.error("health error uid=%d: %s", _uid(msg), e)
        await msg.answer(f"Error: {e}")


@router.message(F.text)
async def handle_text(msg: Message, client: SonalityClient, audio: AudioProcessor) -> None:
    uid = _uid(msg)
    text = msg.text or ""
    log.debug("Text message uid=%d: %d chars", uid, len(text))
    try:
        resp = await client.chat(text)
        log.debug("Chat response uid=%d: %d chars ess=%.2f", uid, len(resp.text), resp.ess_score)
        await _send_text_reply(msg, resp.text, audio)
    except Exception as e:
        log.error("Chat error uid=%d: %s", uid, e, exc_info=True)
        await msg.answer(f"Error: {e}")


@router.message(F.voice)
async def handle_voice(
    msg: Message, bot: Bot, client: SonalityClient, audio: AudioProcessor
) -> None:
    voice = msg.voice
    if not voice:
        return
    uid = _uid(msg)
    log.debug(
        "Voice message uid=%d: duration=%ds mime=%s", uid, voice.duration or 0, voice.mime_type
    )

    try:
        dl = await bot.download(voice.file_id)
        if not isinstance(dl, BytesIO):
            raise TypeError("Expected BytesIO")
        voice_bytes = dl.read()
        log.debug("Downloaded voice uid=%d: %d bytes", uid, len(voice_bytes))
    except Exception as e:
        log.error("Download failed uid=%d: %s", uid, e, exc_info=True)
        await msg.answer(f"Download error: {e}")
        return

    try:
        text = await audio.stt(voice_bytes, voice.mime_type or "audio/ogg")
        log.debug("STT result uid=%d: %d chars", uid, len(text))
        await msg.answer(f"<i>You:</i> {text}", parse_mode="HTML")
    except Exception as e:
        log.error("STT failed uid=%d: %s", uid, e, exc_info=True)
        await msg.answer(f"Transcription error: {e}")
        return

    if not text.strip():
        log.warning("Empty transcription uid=%d", uid)
        await msg.answer("Could not transcribe. Try again.")
        return

    try:
        resp = await client.chat(text)
        log.debug("Chat response uid=%d: %d chars ess=%.2f", uid, len(resp.text), resp.ess_score)
        await _send_voice_reply(msg, resp.text, audio)
    except Exception as e:
        log.error("Chat error uid=%d: %s", uid, e, exc_info=True)
        await msg.answer(f"Error: {e}")


async def _main() -> None:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not config.TELEGRAM_TOKEN:
        sys.exit("Error: CHAT_TELEGRAM_TOKEN not set")

    log.info("=== Sonality Telegram Bot ===")
    log.info("Sonality API: %s", config.SONALITY_URL)
    log.info("Speaches API: %s", config.SPEACHES_URL)
    log.debug(
        "STT: model=%s lang=%s timeout=%.0fs",
        config.STT_MODEL,
        config.STT_LANGUAGE,
        config.STT_TIMEOUT,
    )
    log.debug(
        "TTS: model=%s voice=%s format=%s speed=%.1f timeout=%.0fs max_len=%d",
        config.TTS_MODEL,
        config.TTS_VOICE,
        config.TTS_FORMAT,
        config.TTS_SPEED,
        config.TTS_TIMEOUT,
        config.TTS_MAX_LENGTH,
    )
    log.debug(
        "TTS optimize: enabled=%s model=%s", config.TTS_OPTIMIZE_ENABLED, config.TTS_OPTIMIZE_MODEL
    )

    bot = Bot(token=config.TELEGRAM_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    async with SonalityClient() as client, AudioProcessor() as audio:
        try:
            h = await client.health()
            log.info("Connected: v%d, %d beliefs", h.version, h.belief_count)
        except Exception as e:
            sys.exit(f"Cannot connect to Sonality: {e}")

        dp["client"], dp["audio"] = client, audio
        await dp.start_polling(bot)


def main() -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_main())


if __name__ == "__main__":
    main()
