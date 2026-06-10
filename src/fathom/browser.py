"""Page fetching — Remote CDP browser via browserless.

Requires FATHOM_BROWSER_WS_URL to be set (ws://browserless:3000 in Docker).
No fallbacks — if the browser service is down, fetches fail with clear errors.
Reconnects per-batch to handle browserless recycling browser instances.
"""

from __future__ import annotations

import asyncio
import contextlib

import structlog
from playwright.async_api import Browser, Playwright, async_playwright

from shared.errors import ServiceUnavailableError

from .config import settings

log = structlog.get_logger(__name__)

_pw_context_manager = None
_pw: Playwright | None = None
_launch_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    global _launch_lock
    if _launch_lock is None:
        _launch_lock = asyncio.Lock()
    return _launch_lock


async def _ensure_pw() -> Playwright:
    """Start the Playwright event loop once."""
    global _pw_context_manager, _pw
    if _pw is not None:
        return _pw
    _pw_context_manager = async_playwright()
    _pw = await _pw_context_manager.__aenter__()
    return _pw


async def launch() -> None:
    """Verify CDP connection is available."""
    async with _get_lock():
        if _pw is not None:
            return
        if not settings.browser_ws_url:
            raise ServiceUnavailableError(
                "FATHOM_BROWSER_WS_URL is not set. "
                "The browserless service must be running for page fetching."
            )
        pw = await _ensure_pw()
        browser = await pw.chromium.connect_over_cdp(settings.browser_ws_url)
        await browser.close()
        log.info("browser_ready", url=settings.browser_ws_url)


async def check() -> None:
    """Verify browserless is currently accepting CDP connections."""
    if _pw is None:
        await launch()
        return
    browser = await _pw.chromium.connect_over_cdp(settings.browser_ws_url)
    await browser.close()


async def close() -> None:
    """Shut down Playwright."""
    global _pw_context_manager, _pw
    if _pw_context_manager is not None:
        with contextlib.suppress(Exception):
            await _pw_context_manager.__aexit__(None, None, None)
        _pw_context_manager = None
        _pw = None
    log.info("browser_closed")


async def _fetch_pw(browser: Browser, url: str, timeout_ms: int) -> str:
    """Fetch a single page using a Playwright browser."""
    ctx = await browser.new_context()
    page = await ctx.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        await page.wait_for_timeout(1000)
        return await page.content()
    finally:
        await page.close()
        await ctx.close()


async def fetch_preview_batch(urls: list[str]) -> list[str | Exception]:
    """Lightweight Playwright fetch for page previews — no extra wait, short timeout.

    Returns raw HTML (or Exception) per URL. Caller runs extract_preview on successes.
    """
    if _pw is None:
        await launch()
    assert _pw is not None

    browser = await _pw.chromium.connect_over_cdp(settings.browser_ws_url)
    sem = asyncio.Semaphore(settings.browser_concurrency)

    async def _one(url: str) -> str | Exception:
        async with sem:
            ctx = await browser.new_context()
            page = await ctx.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=5000)
                return await page.content()
            except Exception as exc:
                return exc
            finally:
                await page.close()
                await ctx.close()

    try:
        return await asyncio.gather(*[_one(u) for u in urls])
    finally:
        with contextlib.suppress(Exception):
            await browser.close()


async def fetch_batch(urls: list[str]) -> list[str | Exception]:
    """Fetch up to N URLs in parallel via a single CDP connection per batch."""
    if _pw is None:
        await launch()
    assert _pw is not None

    browser = await _pw.chromium.connect_over_cdp(settings.browser_ws_url)
    sem = asyncio.Semaphore(settings.browser_concurrency)

    async def _one(url: str) -> str | Exception:
        async with sem:
            try:
                return await _fetch_pw(browser, url, 15_000)
            except Exception as exc:
                log.warning("fetch_failed", url=url, error=str(exc))
                return exc

    try:
        return await asyncio.gather(*[_one(u) for u in urls])
    finally:
        with contextlib.suppress(Exception):
            await browser.close()
