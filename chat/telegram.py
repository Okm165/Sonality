"""Telegram bot interface for Sonality with STT/TTS and streaming progress."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
import sys
import time
from io import BytesIO

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, Message

from . import config
from .audio import AudioProcessor, chunk_text, mp3_to_ogg_opus, optimize_for_speech
from .client import TOOL_LABELS, ProgressEvent, SonalityClient, pipeline_summary

log = logging.getLogger(__name__)
router = Router()


class ClientPool:
    """Per-user SonalityClient pool with automatic cleanup of idle clients."""

    def __init__(self, idle_timeout: float = 3600.0) -> None:
        self._clients: dict[int, SonalityClient] = {}
        self._last_access: dict[int, float] = {}
        self._idle_timeout = idle_timeout
        self._lock = asyncio.Lock()

    async def get(self, user_id: int) -> SonalityClient:
        async with self._lock:
            now = time.monotonic()
            if user_id not in self._clients:
                self._clients[user_id] = SonalityClient()
                log.debug("Created new client for user_id=%d", user_id)
            self._last_access[user_id] = now
            return self._clients[user_id]

    async def cleanup(self) -> int:
        async with self._lock:
            now = time.monotonic()
            to_remove = [
                uid for uid, last in self._last_access.items() if now - last > self._idle_timeout
            ]
            for uid in to_remove:
                await self._clients[uid].close()
                del self._clients[uid]
                del self._last_access[uid]
                log.debug("Removed idle client for user_id=%d", uid)
            return len(to_remove)

    async def close_all(self) -> None:
        async with self._lock:
            for client in self._clients.values():
                await client.close()
            self._clients.clear()
            self._last_access.clear()


def _uid(msg: Message) -> int:
    return msg.from_user.id if msg.from_user else 0


async def _generate_voices(audio: AudioProcessor, chunks: list[str]) -> list[bytes | None]:
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
    display_chunks = chunk_text(text)
    if not display_chunks:
        return
    tts_text = await optimize_for_speech(text)
    tts_chunks = chunk_text(tts_text)
    voices = await _generate_voices(audio, tts_chunks)
    for chunk in display_chunks:
        await msg.answer(chunk)
    for i, ogg in enumerate(voices):
        if ogg:
            try:
                await msg.answer_voice(BufferedInputFile(ogg, f"voice_{i}.ogg"))
            except Exception as e:
                log.warning("Voice send failed chunk=%d: %s", i, e)


@router.message(CommandStart())
async def cmd_start(msg: Message, pool: ClientPool) -> None:
    uid = _uid(msg)
    try:
        client = await pool.get(uid)
        h = await client.health()
        beliefs = await client.beliefs()
        status = f"v{h.snapshot_version} | {h.belief_count} beliefs"
        if beliefs:
            top = ", ".join(
                b.topic for b in sorted(beliefs, key=lambda b: b.confidence, reverse=True)[:3]
            )
            status += f"\nTop: {top}"
    except Exception:
        status = "connecting..."
    await msg.answer(
        f"<b>Sonality</b> - a mind that grows\n\n{status}\n\nSend a message to chat | /help for commands",
        parse_mode="HTML",
    )


@router.message(Command("help"))
async def cmd_help(msg: Message) -> None:
    await msg.answer(
        "<b>Commands</b>\n\n"
        "Send any text or voice message to chat\n\n"
        "/beliefs - Current beliefs with confidence\n"
        "/snapshot - Personality overview\n"
        "/health - Service health\n"
        "/clear - Reset conversation\n"
        "/help - This message",
        parse_mode="HTML",
    )


@router.message(Command("beliefs"))
async def cmd_beliefs(msg: Message, pool: ClientPool) -> None:
    uid = _uid(msg)
    try:
        client = await pool.get(uid)
        beliefs = await client.beliefs()
        if not beliefs:
            await msg.answer("No beliefs formed yet. Start chatting!")
            return
        sorted_beliefs = sorted(beliefs, key=lambda b: b.confidence, reverse=True)
        lines: list[str] = []
        for b in sorted_beliefs[:15]:
            sign = "[+]" if b.valence > 0.1 else "[-]" if b.valence < -0.1 else "[o]"
            text_preview = f"\n   <i>{b.belief_text[:90]}</i>" if b.belief_text else ""
            lines.append(f"{sign} <b>{b.topic}</b> ({b.confidence:.0%}){text_preview}")
        await msg.answer(
            f"<b>Beliefs</b> ({len(beliefs)})\n\n" + "\n\n".join(lines),
            parse_mode="HTML",
        )
    except Exception as e:
        log.error("beliefs error uid=%d: %s", uid, e)
        await msg.answer(f"Error: {e}")


@router.message(Command("snapshot"))
async def cmd_snapshot(msg: Message, pool: ClientPool) -> None:
    uid = _uid(msg)
    try:
        client = await pool.get(uid)
        h = await client.health()
        beliefs = await client.beliefs()
        uptime_min = int(h.uptime_seconds // 60)
        top_topics = ""
        if beliefs:
            top = ", ".join(
                b.topic for b in sorted(beliefs, key=lambda b: b.confidence, reverse=True)[:5]
            )
            top_topics = f"\nTop beliefs: {top}"
        await msg.answer(
            f"<b>Personality Snapshot</b>\n\n"
            f"Version: {h.snapshot_version}\n"
            f"Beliefs: {h.belief_count}\n"
            f"Uptime: {uptime_min}m" + (f"\nBuild: {h.version}" if h.version else "") + top_topics,
            parse_mode="HTML",
        )
    except Exception as e:
        log.error("snapshot error uid=%d: %s", uid, e)
        await msg.answer(f"Error: {e}")


@router.message(Command("health"))
async def cmd_health(msg: Message, pool: ClientPool) -> None:
    uid = _uid(msg)
    try:
        client = await pool.get(uid)
        h = await client.health()
        await msg.answer(
            f"<b>Healthy</b>\nSnapshot: v{h.snapshot_version}\nBeliefs: {h.belief_count}",
            parse_mode="HTML",
        )
    except Exception as e:
        log.error("health error uid=%d: %s", uid, e)
        await msg.answer(f"Error: {e}")


@router.message(Command("clear"))
async def cmd_clear(msg: Message, pool: ClientPool) -> None:
    uid = _uid(msg)
    try:
        client = await pool.get(uid)
        client.clear_history()
        await msg.answer("Conversation cleared.")
    except Exception as e:
        log.error("clear error uid=%d: %s", uid, e)
        await msg.answer(f"Error: {e}")


def _format_step(tool_name: str, args_summary: str) -> str:
    label = TOOL_LABELS.get(tool_name, tool_name.replace("_", " "))
    if args_summary:
        return f"{label}: {args_summary}"
    return label


def _format_status(steps: list[str], current: str, elapsed: float) -> str:
    parts: list[str] = []
    for s in steps[-5:]:
        parts.append(f"[done] {s}")
    if current:
        parts.append(f">> {current}")
    status = "\n".join(parts) or "Thinking..."
    if elapsed > 1.0:
        status += f"\n({elapsed:.0f}s)"
    return status


def _format_done_footer(detail: str, tool_names: list[str]) -> str | None:
    try:
        d = json.loads(detail)
    except (json.JSONDecodeError, TypeError):
        return None
    parts: list[str] = []
    pipeline = pipeline_summary(tool_names)
    if pipeline:
        parts.append(pipeline)
    if el := d.get("elapsed"):
        parts.append(f"{el}s")
    if ess := d.get("ess_score"):
        parts.append(f"ESS {ess}")
    if topics := d.get("topics"):
        parts.append(" ".join(f"#{t}" for t in topics[:3]))
    return " | ".join(parts) if parts else None


@router.message(F.text)
async def handle_text(msg: Message, bot: Bot, pool: ClientPool, audio: AudioProcessor) -> None:
    uid = _uid(msg)
    text = msg.text or ""
    log.info("Text message uid=%d: %d chars", uid, len(text))
    t0 = time.perf_counter()

    with contextlib.suppress(Exception):
        await bot.send_chat_action(msg.chat.id, ChatAction.TYPING)

    try:
        client = await pool.get(uid)
        status_msg: Message | None = None
        accumulated_text = ""
        completed_steps: list[str] = []
        tool_names_used: list[str] = []
        current_step = ""
        last_edit = 0.0
        draft_id = int(time.monotonic() * 1000) % 2147483647 or 1
        use_draft = True
        draft_started = False
        done_footer: str | None = None

        async for item in client.chat_stream(text):
            now = time.monotonic()
            elapsed = time.perf_counter() - t0

            if isinstance(item, ProgressEvent):
                if item.type == "done":
                    done_footer = _format_done_footer(item.detail, tool_names_used)
                    continue

                if item.type == "tool_call":
                    args_summary = ""
                    if item.tool_args:
                        try:
                            parsed = json.loads(item.tool_args)
                            args_summary = parsed.get("query", parsed.get("url", ""))[:50]
                        except json.JSONDecodeError:
                            args_summary = item.tool_args[:50]
                    if item.tool_name:
                        tool_names_used.append(item.tool_name)
                    current_step = _format_step(item.tool_name, args_summary)
                elif item.type == "tool_result":
                    if current_step:
                        short = current_step[:45]
                        completed_steps.append(short)
                    current_step = ""
                elif item.type == "thinking":
                    current_step = "Thinking..."
                elif item.type == "context_build":
                    current_step = item.detail if item.detail else "Loading context..."
                elif item.type == "summarizing":
                    current_step = "Compressing context..."

                status_text = _format_status(completed_steps, current_step, elapsed)
                if status_msg is None:
                    status_msg = await msg.answer(status_text)
                    last_edit = now
                elif now - last_edit > 0.8:
                    with contextlib.suppress(Exception):
                        await status_msg.edit_text(status_text)
                    last_edit = now

            elif isinstance(item, str):
                accumulated_text += item
                if now - last_edit < 0.3:
                    continue

                if use_draft:
                    try:
                        if not draft_started and status_msg and completed_steps:
                            summary = " | ".join(f"[done] {s}" for s in completed_steps[-3:])
                            with contextlib.suppress(Exception):
                                await status_msg.edit_text(summary)
                            draft_started = True
                        await bot.send_message_draft(
                            chat_id=msg.chat.id,
                            draft_id=draft_id,
                            text=accumulated_text + " |",
                        )
                    except Exception:
                        log.info("sendMessageDraft unavailable, falling back to editMessageText")
                        use_draft = False

                if not use_draft and accumulated_text:
                    display = accumulated_text + " |"
                    if not draft_started:
                        if status_msg and completed_steps:
                            summary = " | ".join(f"[done] {s}" for s in completed_steps[-3:])
                            with contextlib.suppress(Exception):
                                await status_msg.edit_text(summary)
                        draft_started = True
                        status_msg = await msg.answer(display)
                    else:
                        with contextlib.suppress(Exception):
                            if status_msg:
                                await status_msg.edit_text(display)
                last_edit = now

        if status_msg and accumulated_text and tool_names_used:
            pipeline = pipeline_summary(tool_names_used)
            if pipeline:
                with contextlib.suppress(Exception):
                    await status_msg.edit_text(f"<i>{pipeline}</i>", parse_mode="HTML")
            else:
                with contextlib.suppress(Exception):
                    await status_msg.delete()
        elif status_msg and accumulated_text:
            with contextlib.suppress(Exception):
                await status_msg.delete()

        if accumulated_text:
            final = accumulated_text
            if done_footer:
                final += f"\n\n<i>{done_footer}</i>"
            await msg.answer(final, parse_mode="HTML")
            log.info(
                "Response uid=%d: %d chars, %d tools, elapsed=%.1fs",
                uid,
                len(accumulated_text),
                len(tool_names_used),
                time.perf_counter() - t0,
            )
        elif not accumulated_text:
            log.warning("Empty response uid=%d", uid)

    except Exception as e:
        log.error("Chat error uid=%d: %s", uid, e, exc_info=True)
        await msg.answer(f"Error: {e}")


@router.message(F.voice)
async def handle_voice(msg: Message, bot: Bot, pool: ClientPool, audio: AudioProcessor) -> None:
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
        await msg.answer(f"<i>{text}</i>", parse_mode="HTML")
    except Exception as e:
        log.error("STT failed uid=%d: %s", uid, e, exc_info=True)
        await msg.answer(f"Transcription error: {e}")
        return

    if not text.strip():
        log.warning("Empty transcription uid=%d", uid)
        await msg.answer("Could not transcribe. Try again.")
        return

    try:
        client = await pool.get(uid)
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

    bot = Bot(token=config.TELEGRAM_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    pool = ClientPool()
    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        log.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    async with AudioProcessor() as audio:
        async with SonalityClient() as test_client:
            try:
                h = await test_client.health()
                log.info("Connected: snapshot_v%d, %d beliefs", h.snapshot_version, h.belief_count)
            except Exception as e:
                sys.exit(f"Cannot connect to Sonality: {e}")

        dp["pool"], dp["audio"] = pool, audio
        try:
            polling_task = asyncio.create_task(dp.start_polling(bot))
            shutdown_task = asyncio.create_task(shutdown_event.wait())
            _done, pending = await asyncio.wait(
                [polling_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        finally:
            log.info("Closing client pool...")
            await pool.close_all()
            log.info("Telegram bot shutdown complete")


def main() -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_main())


if __name__ == "__main__":
    main()
