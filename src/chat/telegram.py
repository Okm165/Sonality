"""Telegram bot interface for Sonality with STT/TTS and streaming progress."""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import sys
import time
from dataclasses import dataclass, field
from io import BytesIO

import structlog
from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, Message

from . import config
from .audio import AudioProcessor, chunk_text, mp3_to_ogg_opus, optimize_for_speech
from .client import (
    TOOL_LABELS,
    ProgressEvent,
    SonalityClient,
    extract_tool_arg_summary,
    pipeline_summary,
)

log = structlog.get_logger()
router = Router()


class ClientPool:
    """Per-user SonalityClient pool with automatic cleanup of idle clients."""

    def __init__(self, idle_timeout: float = 3600.0) -> None:
        self._clients: dict[int, SonalityClient] = {}
        self._last_access: dict[int, float] = {}
        self._idle_timeout = idle_timeout
        self._lock = asyncio.Lock()

    async def get(self, user_id: int) -> SonalityClient:
        """Return a client for *user_id*, creating one if needed."""
        async with self._lock:
            now = time.monotonic()
            if user_id not in self._clients:
                self._clients[user_id] = SonalityClient()
                log.debug("client_created", user_id=user_id)
            self._last_access[user_id] = now
            return self._clients[user_id]

    def touch(self, user_id: int) -> None:
        """Bump idle timer for *user_id* — call periodically during long streams."""
        self._last_access[user_id] = time.monotonic()

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
                log.debug("client_removed_idle", user_id=uid)
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
            log.warning("voice_gen_failed", chunk=idx, error=str(e))
            return idx, None

    results = await asyncio.gather(*[gen(c, i) for i, c in enumerate(chunks)])
    return [data for _, data in sorted(results)]


async def _send_voice_reply(msg: Message, text: str, audio: AudioProcessor) -> None:
    """Send text chunks as messages, then all TTS voice clips."""
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
                log.warning("voice_send_failed", chunk=i, error=str(e))


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
        log.warning("cmd_start_health_failed", exc_info=True)
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
        log.error("beliefs_error", user_id=uid, error=str(e))
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
        log.error("snapshot_error", user_id=uid, error=str(e))
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
        log.error("health_error", user_id=uid, error=str(e))
        await msg.answer(f"Error: {e}")


@router.message(Command("clear"))
async def cmd_clear(msg: Message, pool: ClientPool) -> None:
    uid = _uid(msg)
    try:
        client = await pool.get(uid)
        client.clear_history()
        await msg.answer("Conversation cleared.")
    except Exception as e:
        log.error("clear_error", user_id=uid, error=str(e))
        await msg.answer(f"Error: {e}")


def _format_step(tool_name: str, args_summary: str) -> str:
    label = TOOL_LABELS.get(tool_name, tool_name.replace("_", " "))
    return f"{label}: {args_summary}" if args_summary else label


def _format_status(steps: list[str], current: str, elapsed: float) -> str:
    parts = [f"[done] {s}" for s in steps[-5:]]
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
    if rtype := d.get("reasoning_type"):
        parts.append(f"({rtype})")
    if topics := d.get("topics"):
        parts.append(" ".join(f"#{t}" for t in topics[:3]))
    return " | ".join(parts) if parts else None


@dataclass
class _StreamResult:
    """Accumulated state from consuming a chat_stream."""

    text: str = ""
    tool_names: list[str] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)
    done_footer: str | None = None


async def _consume_stream(
    msg: Message,
    client: SonalityClient,
    user_text: str,
    t0: float,
    *,
    edit_throttle: float = 0.8,
    stream_text: bool = False,
    bot: Bot | None = None,
    pool: ClientPool | None = None,
    user_id: int = 0,
) -> _StreamResult:
    """Shared streaming consumer for both text and voice handlers.

    Manages the status message lifecycle and accumulates results. When
    stream_text is True and bot is provided, progressive draft streaming
    is attempted for the text path.

    If *pool* and *user_id* are provided, the client's idle timer is
    bumped periodically to prevent cleanup during long streams.
    """
    status_msg: Message | None = None
    current_step = ""
    last_progress_edit = 0.0
    last_text_edit = 0.0
    result = _StreamResult()

    draft_id = int(time.monotonic() * 1000) % 2147483647 or 1
    use_draft = stream_text
    draft_started = False

    async for item in client.chat_stream(user_text):
        now = time.monotonic()
        elapsed = time.perf_counter() - t0

        if pool and user_id:
            pool.touch(user_id)

        if isinstance(item, ProgressEvent):
            if item.type == "done":
                result.done_footer = _format_done_footer(item.detail, result.tool_names)
                continue

            if item.type == "tool_call":
                args_summary = extract_tool_arg_summary(item.tool_args)[:50]
                if item.tool_name:
                    result.tool_names.append(item.tool_name)
                current_step = _format_step(item.tool_name, args_summary)
            elif item.type == "tool_result":
                if current_step:
                    detail = ""
                    if item.sources_count:
                        detail = f" ({item.sources_count} sources)"
                    elif item.tool_result_summary:
                        detail = f" - {item.tool_result_summary[:40]}"
                    result.completed_steps.append(f"{current_step[:40]}{detail}")
                current_step = ""
            elif item.type == "thinking":
                current_step = "Thinking..."
            elif item.type == "reviewing":
                current_step = item.detail or "Cross-checking evidence..."
            elif item.type == "context_build":
                current_step = item.detail or "Loading context..."
            elif item.type == "summarizing":
                current_step = item.detail or "Compressing context..."

            status_text = _format_status(result.completed_steps, current_step, elapsed)
            if status_msg is None:
                status_msg = await msg.answer(status_text)
                last_progress_edit = now
            elif now - last_progress_edit > edit_throttle:
                with contextlib.suppress(Exception):
                    await status_msg.edit_text(status_text)
                last_progress_edit = now

        elif isinstance(item, str):
            result.text += item

            if stream_text and bot and now - last_text_edit >= 0.3:
                if use_draft:
                    try:
                        if not draft_started and status_msg and result.completed_steps:
                            summary = " | ".join(f"[done] {s}" for s in result.completed_steps[-3:])
                            with contextlib.suppress(Exception):
                                await status_msg.edit_text(summary)
                            draft_started = True
                        await bot.send_message_draft(
                            chat_id=msg.chat.id,
                            draft_id=draft_id,
                            text=result.text + " |",
                        )
                    except Exception:
                        log.info("draft_unavailable_fallback")
                        use_draft = False

                if not use_draft and result.text:
                    display = result.text + " |"
                    if not draft_started:
                        if status_msg and result.completed_steps:
                            summary = " | ".join(f"[done] {s}" for s in result.completed_steps[-3:])
                            with contextlib.suppress(Exception):
                                await status_msg.edit_text(summary)
                        draft_started = True
                        status_msg = await msg.answer(display)
                    else:
                        with contextlib.suppress(Exception):
                            if status_msg:
                                await status_msg.edit_text(display)
                last_text_edit = now

    if status_msg and result.tool_names:
        pipeline = pipeline_summary(result.tool_names)
        if pipeline:
            with contextlib.suppress(Exception):
                await status_msg.edit_text(f"<i>{pipeline}</i>", parse_mode="HTML")
        else:
            with contextlib.suppress(Exception):
                await status_msg.delete()
    elif status_msg:
        with contextlib.suppress(Exception):
            await status_msg.delete()

    return result


@router.message(F.text)
async def handle_text(msg: Message, bot: Bot, pool: ClientPool, **_: object) -> None:
    uid = _uid(msg)
    text = msg.text or ""
    log.info("text_message", user_id=uid, chars=len(text))
    t0 = time.perf_counter()

    with contextlib.suppress(Exception):
        await bot.send_chat_action(msg.chat.id, ChatAction.TYPING)

    try:
        client = await pool.get(uid)
        result = await _consume_stream(
            msg,
            client,
            text,
            t0,
            edit_throttle=0.8,
            stream_text=True,
            bot=bot,
            pool=pool,
            user_id=uid,
        )

        if result.text:
            final = result.text
            if result.done_footer:
                final += f"\n\n<i>{result.done_footer}</i>"
            await msg.answer(final, parse_mode="HTML")
            log.info("response_complete", user_id=uid, chars=len(result.text), tools=len(result.tool_names), elapsed=f"{time.perf_counter() - t0:.1f}s")
        else:
            log.warning("empty_response", user_id=uid)
            await msg.answer("<i>No response generated. Try rephrasing.</i>", parse_mode="HTML")

    except Exception as e:
        log.error("chat_error", user_id=uid, error=str(e), exc_info=True)
        await msg.answer("Something went wrong. Please try again.")


@router.message(F.voice)
async def handle_voice(msg: Message, bot: Bot, pool: ClientPool, audio: AudioProcessor) -> None:
    voice = msg.voice
    if not voice:
        return
    uid = _uid(msg)
    log.debug("voice_message", user_id=uid, duration=voice.duration or 0, mime=voice.mime_type)

    try:
        dl = await bot.download(voice.file_id)
        if not isinstance(dl, BytesIO):
            raise TypeError("Expected BytesIO")
        voice_bytes = dl.read()
        log.debug("voice_downloaded", user_id=uid, bytes=len(voice_bytes))
    except Exception as e:
        log.error("voice_download_failed", user_id=uid, error=str(e))
        await msg.answer(f"Download error: {e}")
        return

    with contextlib.suppress(Exception):
        await bot.send_chat_action(msg.chat.id, ChatAction.RECORD_VOICE)
    try:
        text = await audio.stt(voice_bytes, voice.mime_type or "audio/ogg")
        log.debug("stt_result", user_id=uid, chars=len(text))
        await msg.answer(f"<i>{text}</i>", parse_mode="HTML")
    except Exception as e:
        log.error("stt_failed", user_id=uid, error=str(e))
        await msg.answer(f"Transcription error: {e}")
        return

    if not text.strip():
        log.warning("empty_transcription", user_id=uid)
        await msg.answer("Could not transcribe. Try again.")
        return

    t0 = time.perf_counter()
    try:
        client = await pool.get(uid)
        result = await _consume_stream(
            msg,
            client,
            text,
            t0,
            edit_throttle=1.2,
            stream_text=False,
            pool=pool,
            user_id=uid,
        )

        if result.text:
            log.debug("voice_response", user_id=uid, chars=len(result.text))
            with contextlib.suppress(Exception):
                await bot.send_chat_action(msg.chat.id, ChatAction.RECORD_VOICE)
            await _send_voice_reply(msg, result.text, audio)
            if result.done_footer:
                await msg.answer(f"<i>{result.done_footer}</i>", parse_mode="HTML")
        else:
            log.warning("empty_voice_response", user_id=uid)
            await msg.answer("<i>No response generated. Try again.</i>", parse_mode="HTML")

    except Exception as e:
        log.error("chat_error", user_id=uid, error=str(e), exc_info=True)
        await msg.answer("Something went wrong. Please try again.")


async def _main() -> None:
    if not config.settings.telegram_token:
        sys.exit("Error: CHAT_TELEGRAM_TOKEN not set")

    log.info("telegram_bot_starting", sonality_url=config.settings.sonality_url, speaches_url=config.settings.speaches_url)

    bot = Bot(token=config.settings.telegram_token)
    dp = Dispatcher()
    dp.include_router(router)

    pool = ClientPool()
    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        log.info("shutdown_signal")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    async with AudioProcessor() as audio:
        async with SonalityClient() as test_client:
            try:
                h = await test_client.health()
                log.info("connected", snapshot_version=h.snapshot_version, beliefs=h.belief_count)
            except Exception as e:
                sys.exit(f"Cannot connect to Sonality: {e}")

        async def _periodic_cleanup() -> None:
            while not shutdown_event.is_set():
                await asyncio.sleep(600)
                removed = await pool.cleanup()
                if removed:
                    log.info("clients_cleaned", removed=removed)

        dp["pool"], dp["audio"] = pool, audio
        try:
            polling_task = asyncio.create_task(dp.start_polling(bot))
            shutdown_task = asyncio.create_task(shutdown_event.wait())
            cleanup_task = asyncio.create_task(_periodic_cleanup())
            _done, pending = await asyncio.wait(
                [polling_task, shutdown_task, cleanup_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        finally:
            log.info("closing_pool")
            await pool.close_all()
            log.info("telegram_bot_shutdown")


def main() -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_main())


if __name__ == "__main__":
    main()
