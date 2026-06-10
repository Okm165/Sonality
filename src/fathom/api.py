"""FastAPI routes + SSE streaming.

Endpoints:
  /research — full autonomous research session with SSE progress stream
  /health   — liveness probe
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from qdrant_client import AsyncQdrantClient

from shared.embedder import probe_embedding_dims
from shared.server import (
    HealthResponse,
    RequestIDMiddleware,
    check_health_dependencies,
    http_dependency_status,
)

from . import browser, config, db, session
from .models import (
    FactWithSource,
    ResearchRequest,
    ResearchResponse,
    SessionStatus,
)
from .source_memory import init_source_collection

log = structlog.get_logger(__name__)

_qdrant_client: AsyncQdrantClient | None = None


async def get_qdrant() -> AsyncQdrantClient:
    """Get or create Qdrant client with probed embedding dimensions."""
    global _qdrant_client
    if _qdrant_client is None:
        dims = min(
            config.settings.embedding_dimensions,
            await asyncio.to_thread(probe_embedding_dims, config.settings.embedding_url),
        )
        _qdrant_client = AsyncQdrantClient(url=config.settings.qdrant_url)
        await init_source_collection(_qdrant_client, dims=dims)
        log.info("qdrant_ready", url=config.settings.qdrant_url, dims=dims)
    return _qdrant_client


async def close_qdrant() -> None:
    """Close Qdrant client."""
    global _qdrant_client
    if _qdrant_client is not None:
        await _qdrant_client.close()
        _qdrant_client = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup: init Neo4j + Qdrant + browser + cleanup stale sessions. Shutdown: close all."""
    driver = await db.get_driver()
    await get_qdrant()
    await browser.launch()
    cleaned = await db.cleanup_stale_sessions(driver, max_age_s=config.settings.session_timeout)
    if cleaned:
        log.info("stale_sessions_cleaned", count=cleaned)
    yield
    await browser.close()
    await close_qdrant()
    await db.close_driver()


app = FastAPI(title="Fathom", description="Autonomous web research agent", lifespan=lifespan)
app.add_middleware(RequestIDMiddleware)

_session_queues: dict[str, asyncio.Queue[dict[str, object] | None]] = {}
_MAX_CONCURRENT_SESSIONS = 2


@app.post("/research", response_model=ResearchResponse)
async def start_research(req: ResearchRequest) -> ResearchResponse:
    """Start a new research session. Returns immediately; research runs in background."""
    if len(_session_queues) >= _MAX_CONCURRENT_SESSIONS:
        raise HTTPException(
            status_code=429,
            detail=f"Max {_MAX_CONCURRENT_SESSIONS} concurrent sessions; try again later",
        )
    driver = await db.get_driver()
    qdrant = await get_qdrant()
    session_id = await db.create_session(driver, req.goal)
    event_queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
    _session_queues[session_id] = event_queue

    async def _guarded_run() -> None:
        try:
            await session.run(
                driver,
                qdrant,
                session_id,
                req.goal,
                req.seeds,
                n=req.n,
                max_pages=req.max_pages,
                depth=req.depth,
                event_queue=event_queue,
            )
        except Exception:
            log.error("session_failed", session_id=session_id[:12], exc_info=True)
            event_queue.put_nowait({"event": "error", "status": "failed"})
            for attempt in range(3):
                try:
                    await db.update_session(driver, session_id, status="failed")
                    break
                except Exception:
                    log.error(
                        "session_status_update_failed",
                        session_id=session_id[:12],
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(1.0)
        finally:
            event_queue.put_nowait(None)

    log.info(
        "research_accepted",
        session_id=session_id[:12],
        depth=req.depth,
        max_pages=req.max_pages,
        seeds=len(req.seeds),
        goal_len=len(req.goal),
    )
    task = asyncio.create_task(_guarded_run())

    def _cleanup(_t: object) -> None:
        _session_queues.pop(session_id, None)

    task.add_done_callback(_cleanup)

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

    facts = [
        FactWithSource(
            claim=r["claim"],
            confidence=r["confidence"],
            source_quality=r.get("source_quality") or 0.5,
            source_url=r["source_url"],
            topic=r.get("topic", ""),
        )
        for r in fact_rows
    ]
    return SessionStatus(
        id=row["id"],
        status=row["status"],
        goal=row["goal"],
        pages_scraped=row["pages_scraped"],
        facts=facts,
        checklist=checklist,
    )


@app.get("/research/{session_id}/stream")
async def stream_research(session_id: str) -> StreamingResponse:
    """SSE stream of rich research progress events.

    Events are pushed directly from the session loop via an asyncio.Queue,
    giving real-time visibility into decomposition, fetching, analysis, and facts.
    """
    queue = _session_queues.get(session_id)
    if queue is None:
        driver = await db.get_driver()
        row = await db.get_session(driver, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")
        raise HTTPException(
            status_code=410,
            detail="Stream no longer available (session already finished or queue expired)",
        )

    async def event_stream():
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=config.settings.session_timeout)
            except TimeoutError:
                yield f"event: error\ndata: {json.dumps({'status': 'stream_timeout'})}\n\n"
                break
            if item is None:
                break
            event_type = item.get("event", "progress")
            yield f"event: {event_type}\ndata: {json.dumps(item)}\n\n"
            if event_type in ("complete", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health() -> HealthResponse:
    from . import __version__

    dependencies: dict[str, str] = {}
    try:
        await db.check()
        dependencies["neo4j"] = "ok"
    except Exception as exc:
        dependencies["neo4j"] = f"error: {exc}"

    try:
        qdrant = await get_qdrant()
        await qdrant.get_collections()
        dependencies["qdrant"] = "ok"
    except Exception as exc:
        dependencies["qdrant"] = f"error: {exc}"

    try:
        await browser.check()
        dependencies["browserless"] = "ok"
    except Exception as exc:
        dependencies["browserless"] = f"error: {exc}"

    dependencies["llm"] = await asyncio.to_thread(
        http_dependency_status, f"{config.settings.base_url.rstrip('/')}/models"
    )
    check_health_dependencies(dependencies)
    return HealthResponse(version=__version__, dependencies=dependencies)


def serve() -> None:
    """CLI entrypoint for fathom-server."""
    from shared.server import make_server_parser, run_uvicorn

    parser = make_server_parser("Fathom research server", default_port=8010)
    args = parser.parse_args()
    run_uvicorn("fathom.api:app", args, env_log_key="FATHOM_LOG_LEVEL")
