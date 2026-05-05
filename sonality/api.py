"""FastAPI wrapper for programmatic Sonality access.

Provides HTTP endpoints for data ingestion, belief querying, and
OpenAI-compatible chat completions. Uses a singleton SonalityAgent with
an async lock to prevent concurrent mutation.

/ingest is fire-and-forget: returns 202 immediately with a job_id.
Poll /ingest/{job_id} for status and result.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from enum import StrEnum
from functools import partial
from typing import Final

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from . import __version__, config
from .agent import SonalityAgent
from .ess import ESSResult, ReasoningType, UrgencyLevel
from .memory.graph import BeliefNode
from .progress import AgentEvent
from .provider import StreamChunk
from .schema import ChatRole
from .token_budget import estimate_tokens_utf8

MODEL_ID: Final = "sonality"

log = logging.getLogger(__name__)

_agent_store: dict[str, SonalityAgent] = {}
_agent_lock: asyncio.Lock | None = None
_startup_time: float = 0.0

# --- Ingest job queue ---


class _JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class _IngestJob:
    """Tracks lifecycle of one queued ingest request."""

    __slots__ = ("error", "job_id", "result", "started_at", "status", "text", "topic_override")

    def __init__(self, job_id: str, text: str, topic_override: str) -> None:
        self.job_id = job_id
        self.status: _JobStatus = _JobStatus.PENDING
        self.text = text
        self.topic_override = topic_override
        self.result: ESSResult | None = None
        self.error: str = ""
        self.started_at: float = time.time()


_ingest_jobs: dict[str, _IngestJob] = {}
_ingest_queue: asyncio.Queue[_IngestJob] | None = None
_ingest_worker_task: asyncio.Task[None] | None = None

_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach X-Request-ID header to every response for tracing."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


async def verify_api_key(request: Request, api_key: str | None = Depends(_api_key_header)) -> None:
    """Verify API key if SONALITY_HTTP_API_KEY is set. Skip auth for health endpoints."""
    if config.HTTP_API_KEY is None:
        return
    if request.url.path in ("/health", "/v1/health"):
        return
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    expected = f"Bearer {config.HTTP_API_KEY}"
    if api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _get_lock() -> asyncio.Lock:
    """Return the singleton asyncio lock, creating on first access."""
    global _agent_lock
    if _agent_lock is None:
        _agent_lock = asyncio.Lock()
    return _agent_lock


def _get_agent() -> SonalityAgent:
    agent = _agent_store.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """App startup: create agent + ingest worker. Shutdown: cancel worker + close agent."""
    global _startup_time, _ingest_queue, _ingest_worker_task
    _startup_time = time.time()
    log.info("Initializing Sonality agent for API server")
    _agent_store["agent"] = SonalityAgent()
    _ingest_queue = asyncio.Queue(maxsize=256)
    _ingest_worker_task = asyncio.create_task(_ingest_worker())
    yield
    log.info("Shutting down Sonality agent")
    if _ingest_worker_task:
        _ingest_worker_task.cancel()
    agent = _agent_store.pop("agent", None)
    if agent is not None:
        agent.shutdown()


async def _ingest_worker() -> None:
    """Background worker: processes ingest jobs one at a time, serialized."""
    assert _ingest_queue is not None
    while True:
        job = await _ingest_queue.get()
        job.status = _JobStatus.RUNNING
        log.info("Ingest worker: starting job=%s chars=%d", job.job_id[:8], len(job.text))
        try:
            agent = _get_agent()
            ess = await asyncio.to_thread(
                partial(agent.ingest, job.text, topic_override=job.topic_override)
            )
            job.result = ess
            job.status = _JobStatus.DONE
            log.info("Ingest worker: job=%s done ess=%.2f", job.job_id[:8], ess.score)
        except Exception as exc:
            job.error = str(exc)
            job.status = _JobStatus.FAILED
            log.exception("Ingest worker: job=%s failed", job.job_id[:8])
        finally:
            _ingest_queue.task_done()
        # Evict jobs older than 2 hours to prevent unbounded memory growth
        cutoff = time.time() - 7200
        stale = [jid for jid, j in _ingest_jobs.items() if j.started_at < cutoff]
        for jid in stale:
            _ingest_jobs.pop(jid, None)


app = FastAPI(
    title="Sonality API",
    description="World-understanding intelligence layer with OpenAI-compatible chat interface",
    version=__version__,
    lifespan=lifespan,
    dependencies=[Depends(verify_api_key)],
)
app.add_middleware(RequestIDMiddleware)


# --- OpenAI-Compatible Chat Completions API ---


class ChatMessage(BaseModel):
    role: ChatRole = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=MODEL_ID)
    messages: list[ChatMessage] = Field(..., description="Full conversation history")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    stream: bool = Field(default=False)


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions. Passes full messages[] to the agent."""
    log.info(
        "HTTP /v1/chat/completions stream=%s messages=%d", request.stream, len(request.messages)
    )
    agent = _get_agent()
    user_messages = [m for m in request.messages if m.role is ChatRole.USER]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    eff_max = max(1, min(request.max_tokens, config.LLM_MAX_TOKENS))

    if request.stream:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        def sse_chunk(delta: dict[str, str], finish: str | None = None) -> str:
            return f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]})}\n\n"

        def sse_event(event_type: str, data: dict[str, object]) -> str:
            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

        async def generate() -> AsyncIterator[str]:
            async with _get_lock():
                # Build the stream iterator in a thread (agent setup is synchronous),
                # then consume items via to_thread so each blocking LLM/tool step
                # doesn't block the asyncio event loop.
                loop = asyncio.get_event_loop()
                stream_iter = await loop.run_in_executor(
                    None,
                    partial(
                        agent.respond_stream,
                        messages,
                        max_tokens=eff_max,
                        temperature=request.temperature,
                    ),
                )
                it = iter(stream_iter)
                _sentinel = object()
                while True:
                    item = await asyncio.to_thread(next, it, _sentinel)
                    if item is _sentinel:
                        break
                    if isinstance(item, AgentEvent):
                        yield sse_event(
                            item.type,
                            {
                                "detail": item.detail,
                                "tool_name": item.tool_name,
                                "tool_args": item.tool_args,
                                "tool_result_summary": item.tool_result_summary,
                                "iteration": item.iteration,
                                "sources_count": item.sources_count,
                            },
                        )
                    elif isinstance(item, StreamChunk) and item.content:
                        yield sse_chunk({"content": item.content})
            yield sse_chunk({}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    async with _get_lock():
        response_text = await asyncio.to_thread(
            partial(
                agent.respond,
                messages,
                max_tokens=eff_max,
                temperature=request.temperature,
            )
        )

    prompt_tokens = sum(estimate_tokens_utf8(m.content) for m in request.messages)
    completion_tokens = estimate_tokens_utf8(response_text)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=MODEL_ID,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role=ChatRole.ASSISTANT, content=response_text),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """List available models (OpenAI-compatible)."""
    return ModelsResponse(
        data=[ModelInfo(id=MODEL_ID, created=int(time.time()), owned_by=MODEL_ID)]
    )


@app.get("/v1/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str) -> ModelInfo:
    """Retrieve a single model by ID (OpenAI-compatible)."""
    if model_id != MODEL_ID:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return ModelInfo(id=MODEL_ID, created=int(time.time()), owned_by=MODEL_ID)


# --- Simple Chat Endpoint ---


class SimpleChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    context: list[ChatMessage] = Field(
        default_factory=list, description="Optional conversation history"
    )


class SimpleChatResponse(BaseModel):
    response: str
    ess_score: float
    reasoning_type: ReasoningType
    topics: list[str]


@app.post("/chat", response_model=SimpleChatResponse)
async def simple_chat(request: SimpleChatRequest) -> SimpleChatResponse:
    """Simple chat endpoint. Optionally accepts context for multi-turn."""
    log.info("HTTP /chat context=%d", len(request.context))
    agent = _get_agent()
    messages = [{"role": m.role, "content": m.content} for m in request.context]
    messages.append({"role": ChatRole.USER, "content": request.message})
    async with _get_lock():
        response_text = await asyncio.to_thread(agent.respond, messages)
        ess = agent.last_ess
    return SimpleChatResponse(
        response=response_text,
        ess_score=ess.score,
        reasoning_type=ess.reasoning_type,
        topics=list(ess.topics),
    )


# --- Sonality-Specific Endpoints ---


class IngestRequest(BaseModel):
    text: str = Field(..., description="Content to ingest")
    topic_override: str = Field(default="")


class IngestResponse(BaseModel):
    success: bool
    score: float
    reasoning_type: ReasoningType
    belief_update_recommended: bool
    urgency: UrgencyLevel
    topics: list[str]
    summary: str


class IngestAccepted(BaseModel):
    job_id: str
    status: str = "pending"
    queue_depth: int = 0


class IngestJobStatus(BaseModel):
    job_id: str
    status: str
    result: IngestResponse | None = None
    error: str = ""


class BeliefResponse(BaseModel):
    topic: str
    valence: float
    confidence: float
    evidence_count: int
    uncertainty: float
    belief_text: str

    @classmethod
    def from_node(cls, b: BeliefNode) -> BeliefResponse:
        return cls(
            topic=b.topic,
            valence=b.valence,
            confidence=b.confidence,
            evidence_count=b.evidence_count,
            uncertainty=b.uncertainty,
            belief_text=b.belief_text,
        )


class HealthResponse(BaseModel):
    belief_count: int
    snapshot_version: int
    uptime_seconds: float = 0.0
    version: str = ""


@app.post("/ingest", response_model=IngestAccepted, status_code=202)
async def ingest(request: IngestRequest) -> IngestAccepted:
    """Fire-and-forget ingest. Returns 202 immediately with job_id.

    Poll GET /ingest/{job_id} for status and result.
    The ingest queue serializes jobs so the agent processes one at a time.
    """
    assert _ingest_queue is not None
    log.info("HTTP /ingest chars=%d", len(request.text))
    job_id = uuid.uuid4().hex
    job = _IngestJob(job_id, request.text, request.topic_override)
    _ingest_jobs[job_id] = job
    try:
        _ingest_queue.put_nowait(job)
    except asyncio.QueueFull:
        _ingest_jobs.pop(job_id, None)
        raise HTTPException(status_code=429, detail="Ingest queue full — try again later") from None
    return IngestAccepted(job_id=job_id, status="pending", queue_depth=_ingest_queue.qsize())


@app.get("/ingest/{job_id}", response_model=IngestJobStatus)
async def ingest_status(job_id: str) -> IngestJobStatus:
    """Poll ingest job status. Returns result when done."""
    job = _ingest_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    result = None
    if job.status == _JobStatus.DONE and job.result is not None:
        ess = job.result
        result = IngestResponse(
            success=True,
            score=ess.score,
            reasoning_type=ess.reasoning_type,
            belief_update_recommended=ess.belief_update_recommended,
            urgency=ess.urgency,
            topics=list(ess.topics),
            summary=ess.summary,
        )
    return IngestJobStatus(job_id=job_id, status=job.status, result=result, error=job.error)


@app.get("/beliefs", response_model=list[BeliefResponse])
async def get_beliefs() -> list[BeliefResponse]:
    """Return all current beliefs from graph, sorted by absolute valence."""
    agent = _get_agent()
    beliefs = await asyncio.to_thread(agent.get_all_beliefs)
    return [BeliefResponse.from_node(b) for b in beliefs]


@app.get("/beliefs/{topic}", response_model=BeliefResponse)
async def get_belief_endpoint(topic: str) -> BeliefResponse:
    agent = _get_agent()
    b = await asyncio.to_thread(agent.get_belief, topic)
    if b is None:
        raise HTTPException(status_code=404, detail=f"No belief for topic: {topic}")
    return BeliefResponse.from_node(b)


@app.get("/health", response_model=HealthResponse)
@app.get("/v1/health", response_model=HealthResponse, include_in_schema=False)
async def health() -> HealthResponse:
    """Return agent health: belief count, snapshot version, uptime."""
    agent = _get_agent()
    belief_count, snapshot_version = await asyncio.to_thread(agent.get_health)
    uptime = time.time() - _startup_time if _startup_time > 0 else 0.0
    return HealthResponse(
        belief_count=belief_count,
        snapshot_version=snapshot_version,
        uptime_seconds=round(uptime, 2),
        version=__version__,
    )


def serve() -> None:
    """CLI entry point: ``sonality-server``."""
    import argparse
    import os

    import uvicorn

    parser = argparse.ArgumentParser(description="Sonality API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default=None, help="Log level")
    args = parser.parse_args()
    log_level = args.log_level or os.environ.get("SONALITY_LOG_LEVEL", "info")
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    uvicorn.run(
        "sonality.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=log_level.lower(),
    )
