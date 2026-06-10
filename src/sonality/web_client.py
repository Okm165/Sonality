"""Async HTTP + SSE client for the Fathom research service.

Sonality delegates all web access to fathom via configurable-depth research sessions.
All web interaction flows through the /research endpoint with appropriate depth presets.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx
import structlog

log = structlog.get_logger(__name__)


def _parse_confidence(v: object) -> float:
    try:
        return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _format_research_detail(event: str, payload: dict[str, object]) -> str:
    """Build a concise human-readable string from a Fathom SSE event."""
    if event == "decompose":
        questions = payload.get("questions", [])
        if isinstance(questions, list) and questions:
            return f"Breaking into {len(questions)} questions: {str(questions[0])[:100]}"
        return "Decomposing research goal..."
    if event == "searching":
        queries = payload.get("queries", [])
        if isinstance(queries, list) and queries:
            return f"Searching: {str(queries[0])[:60]}"
        return "Searching..."
    if event == "fetching":
        count = payload.get("count", 0)
        urls = payload.get("urls", [])
        if isinstance(urls, list) and urls:
            first = str(urls[0])
            domain = first.split("/")[2] if first.count("/") >= 2 else first[:40]
            return f"Fetching {count} pages ({domain}...)"
        return f"Fetching {count} pages..."
    if event == "analyzed":
        url = str(payload.get("url", ""))
        title = str(payload.get("title", ""))
        fc = payload.get("fact_count", 0)
        facts = payload.get("facts", [])
        domain = url.split("/")[2] if url.count("/") >= 2 else url[:40]
        label = title[:50] if title else domain
        preview = ""
        if isinstance(facts, list) and facts:
            preview = f" — {str(facts[0])[:80]}"
        return f"{label}: {fc} facts{preview}"
    if event == "facts":
        items = payload.get("items", [])
        source = str(payload.get("source_url", ""))
        domain = source.split("/")[2] if source.count("/") >= 2 else source[:30]
        n = len(items) if isinstance(items, list) else 0
        preview = ""
        if isinstance(items, list) and items and isinstance(items[0], dict):
            preview = f" — {str(items[0].get('claim', ''))[:80]}"
        return f"Found {n} facts from {domain}{preview}"
    if event == "round_end":
        return f"Round {payload.get('round', '?')}: {payload.get('round_facts', 0)} new facts ({payload.get('total_facts', 0)} total, {payload.get('total_pages', 0)} pages)"
    if event == "complete":
        return f"Research complete: {payload.get('facts', 0)} facts from {payload.get('pages', 0)} pages"
    if event == "error":
        return f"Research failed: {payload.get('status', 'unknown')}"
    return event.replace("_", " ").capitalize()


def _trace_headers() -> dict[str, str]:
    """Retrieve current trace_id from contextvars and return as header dict."""
    ctx = structlog.contextvars.get_contextvars()
    trace_id = ctx.get("trace_id")
    return {"X-Trace-ID": trace_id} if trace_id else {}


@dataclass(frozen=True, slots=True)
class ResearchFact:
    """A single extracted fact with its source and page context."""

    claim: str
    confidence: float
    source_url: str
    topic: str = ""
    source_title: str = ""
    summary: str = ""


@dataclass(frozen=True, slots=True)
class ResearchProgress:
    """SSE progress event from a running research session.

    Event types from Fathom: decompose, memory_sources, searching, ranked,
    fetching, analyzed, facts, round_end, complete, error.
    ``partial_facts`` carries structured facts streamed mid-research.
    """

    event: str
    detail: str = ""
    partial_facts: tuple[ResearchFact, ...] = ()


@dataclass(frozen=True, slots=True)
class ResearchResult:
    """Final result from a completed research session."""

    session_id: str
    status: str
    facts: tuple[ResearchFact, ...]
    pages_scraped: int = 0


@dataclass(frozen=True, slots=True)
class ResearchSession:
    """Handle for a started research session."""

    session_id: str


@dataclass
class ResearchClient:
    """HTTP client for fathom's research API.

    All web access flows through /research with configurable depth presets.
    Methods raise on HTTP errors (via raise_for_status); callers handle degradation.
    """

    base_url: str
    _http: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            timeout=60.0,
        )

    async def start_research(
        self,
        goal: str,
        *,
        max_pages: int | None = None,
        n: int | None = None,
        seeds: list[str] | None = None,
        depth: str = "standard",
    ) -> ResearchSession:
        """Start a research session. Returns immediately.

        Depth presets: glance, quick, focused, standard, thorough, deep, exhaustive.
        Only pass max_pages/n to override the preset.
        """
        payload: dict[str, object] = {"goal": goal, "seeds": seeds or [], "depth": depth}
        if max_pages is not None:
            payload["max_pages"] = max_pages
        if n is not None:
            payload["n"] = n
        resp = await self._http.post(
            "/research", headers=_trace_headers(), json=payload, timeout=30.0
        )
        resp.raise_for_status()
        data = resp.json()
        sid = data.get("id") if isinstance(data, dict) else None
        if not sid:
            raise httpx.HTTPStatusError(
                "Missing session id in response", request=resp.request, response=resp
            )
        return ResearchSession(session_id=str(sid))

    async def stream_research(self, session_id: str) -> AsyncIterator[ResearchProgress]:
        """Connect to SSE stream for a research session. Yields rich progress events."""
        async with self._http.stream(
            "GET",
            f"/research/{session_id}/stream",
            headers=_trace_headers(),
            timeout=httpx.Timeout(connect=30.0, read=18000.0, write=30.0, pool=30.0),
        ) as resp:
            resp.raise_for_status()
            event_type = ""
            data_buffer = ""
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    chunk = line[5:].strip()
                    data_buffer = f"{data_buffer}\n{chunk}" if data_buffer else chunk
                elif line == "" and data_buffer:
                    try:
                        payload = json.loads(data_buffer)
                    except json.JSONDecodeError:
                        log.warning("sse_json_decode_failed", data=data_buffer[:80])
                        payload = {}
                    et = event_type or "progress"
                    terminal = et in ("complete", "error")
                    pfacts: tuple[ResearchFact, ...] = ()
                    if et == "facts":
                        raw_items = payload.get("items", [])
                        src_url = str(payload.get("source_url", ""))
                        src_title = str(payload.get("source_title", ""))
                        src_summary = str(payload.get("summary", ""))
                        if isinstance(raw_items, list):
                            pfacts = tuple(
                                ResearchFact(
                                    claim=str(f.get("claim", "")),
                                    confidence=_parse_confidence(f.get("confidence", 0.5)),
                                    source_url=src_url,
                                    topic=str(f.get("topic", "")),
                                    source_title=src_title,
                                    summary=src_summary,
                                )
                                for f in raw_items
                                if isinstance(f, dict) and f.get("claim")
                            )
                    yield ResearchProgress(
                        event=et,
                        detail=_format_research_detail(et, payload),
                        partial_facts=pfacts,
                    )
                    data_buffer = ""
                    event_type = ""
                    if terminal:
                        break

    async def get_research_result(self, session_id: str) -> ResearchResult:
        """Fetch the final research result for a completed session."""
        resp = await self._http.get(f"/research/{session_id}", headers=_trace_headers())
        resp.raise_for_status()
        data = resp.json()
        raw_facts = data.get("facts") if isinstance(data, dict) else None
        if not isinstance(raw_facts, list):
            raw_facts = []

        facts = tuple(
            ResearchFact(
                claim=str(f.get("claim", "")),
                confidence=_parse_confidence(f.get("confidence", 0.0)),
                source_url=str(f.get("source_url", "")),
                topic=str(f.get("topic", "")),
            )
            for f in raw_facts
            if isinstance(f, dict) and f.get("claim")
        )
        pages = data.get("pages_scraped") if isinstance(data, dict) else 0
        return ResearchResult(
            session_id=session_id,
            status=str(data.get("status", "")) if isinstance(data, dict) else "",
            facts=facts,
            pages_scraped=int(pages) if isinstance(pages, (int, float)) else 0,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()
