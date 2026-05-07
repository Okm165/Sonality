"""FastAPI routes + SSE streaming.

Three service tiers:
  /search, /extract  — lightweight, synchronous-style endpoints for sonality's web tools
  /research          — full autonomous research session with SSE progress stream
  /health            — liveness probe
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import browser, db, search, session
from .extract import extract_content
from .models import (
    ResearchRequest,
    ResearchResponse,
    SessionStatus,
    WebExtractRequest,
    WebExtractResponse,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResult,
)

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup: init Neo4j driver + schema + browser. Shutdown: close both."""
    await db.get_driver()
    await browser.launch()
    yield
    await browser.close()
    await db.close_driver()


app = FastAPI(title="Fathom", description="Autonomous web research agent", lifespan=lifespan)

_tasks: dict[str, asyncio.Task[object]] = {}


# ---------------------------------------------------------------------------
# Lightweight search / extract — used by sonality's web tools
# ---------------------------------------------------------------------------


@app.post("/search", response_model=WebSearchResponse)
async def web_search(req: WebSearchRequest) -> WebSearchResponse:
    """Quick DuckDuckGo search returning URLs, titles, and snippets."""
    links = await search.query(req.query, max_results=req.max_results)
    results: list[WebSearchResult] = []
    for link in links:
        results.append(
            WebSearchResult(
                url=link.url,
                title=link.anchor_text,
                snippet=link.context,
            )
        )
    log.info("web_search", query=req.query, results=len(results))
    return WebSearchResponse(results=results, query=req.query)


@app.post("/extract", response_model=WebExtractResponse)
async def web_extract(req: WebExtractRequest) -> WebExtractResponse:
    """Fetch a single URL via Playwright, extract clean text via trafilatura."""
    try:
        html = await browser.fetch(req.url)
        page = extract_content(html, req.url)
    except Exception as exc:
        log.error("extract_failed", url=req.url, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Failed to extract: {exc}") from exc
    content = page.markdown[:8000] if page.has_content else ""
    log.info("web_extract", url=req.url, chars=len(content))
    return WebExtractResponse(url=req.url, title=page.title, content=content)


# ---------------------------------------------------------------------------
# Full research sessions
# ---------------------------------------------------------------------------


@app.post("/research", response_model=ResearchResponse)
async def start_research(req: ResearchRequest) -> ResearchResponse:
    """Start a new research session. Returns immediately; research runs in background."""
    driver = await db.get_driver()
    session_id = await db.create_session(driver, req.goal)

    async def _guarded_run() -> None:
        try:
            await session.run(
                driver, session_id, req.goal, req.seeds,
                n=req.n, max_pages=req.max_pages, depth=req.depth,
            )
        except Exception:
            log.error("session_failed", session_id=session_id, exc_info=True)
            for attempt in range(3):
                try:
                    await db.update_session(driver, session_id, status="failed")
                    break
                except Exception:
                    log.error("session_status_update_failed", session_id=session_id, attempt=attempt + 1)
                    await asyncio.sleep(1.0)

    log.info("research_accepted", session_id=session_id, depth=req.depth, max_pages=req.max_pages, seeds=len(req.seeds), goal_len=len(req.goal))
    task = asyncio.create_task(_guarded_run())
    task.add_done_callback(lambda _t: _tasks.pop(session_id, None))
    _tasks[session_id] = task

    return ResearchResponse(id=session_id, status="active")


@app.get("/research/{session_id}", response_model=SessionStatus)
async def get_research(session_id: str) -> SessionStatus:
    """Get current state of a research session."""
    driver = await db.get_driver()
    row = await db.get_session(driver, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    checklist = await db.get_checklist(driver, session_id)
    fact_rows = await db.get_facts(driver, session_id)

    return SessionStatus(
        id=row["id"],
        status=row["status"],
        goal=row["goal"],
        document=row["document"],
        pages_scraped=row["pages_scraped"],
        facts_gathered=len(fact_rows),
        checklist=checklist,
    )


@app.get("/research/{session_id}/stream")
async def stream_research(session_id: str) -> StreamingResponse:
    """SSE stream of research progress events.

    Events emitted:
      progress — pages scraped, facts gathered, checklist state
      complete — final document length; stream ends
    """
    driver = await db.get_driver()
    row = await db.get_session(driver, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_stream():
        last_pages = 0
        while True:
            current = await db.get_session(driver, session_id)
            if current is None:
                break

            pages = current["pages_scraped"]
            status = current["status"]
            if pages != last_pages:
                last_pages = pages
                facts = await db.get_facts(driver, session_id)
                checklist = await db.get_checklist(driver, session_id)
                answered = sum(1 for i in checklist if i.answered)
                data = json.dumps(
                    {
                        "pages": pages,
                        "facts": len(facts),
                        "checklist_answered": answered,
                        "checklist_total": len(checklist),
                        "status": status,
                    }
                )
                yield f"event: progress\ndata: {data}\n\n"

            if status in ("completed", "failed"):
                event_name = "complete" if status == "completed" else "error"
                data = json.dumps(
                    {
                        "status": status,
                        "document_length": len(current["document"]),
                        "pages_scraped": pages,
                    }
                )
                yield f"event: {event_name}\ndata: {data}\n\n"
                break

            await asyncio.sleep(2)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class _HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""


@app.get("/health")
async def health() -> _HealthResponse:
    from . import __version__

    return _HealthResponse(version=__version__)


def serve() -> None:
    """CLI entrypoint for fathom-server."""
    from shared.server import make_server_parser, run_uvicorn

    parser = make_server_parser("Fathom research server", default_port=8010)
    args = parser.parse_args()
    run_uvicorn("fathom.api:app", args, env_log_key="FATHOM_LOG_LEVEL")
