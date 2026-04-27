"""FastAPI wrapper for programmatic Sonality access.

Provides HTTP endpoints for data ingestion, belief querying, and
OpenAI-compatible chat completions. Uses a singleton SonalityAgent with
an async lock to prevent concurrent mutation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from functools import partial
from typing import Final

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from . import __version__, config
from .agent import SonalityAgent
from .ess import ReasoningType, UrgencyLevel
from .memory.graph import BeliefNode
from .schema import ChatRole

MODEL_ID: Final = "sonality"

# Request ID for distributed tracing and debugging
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")

log = logging.getLogger(__name__)

_agent_store: dict[str, SonalityAgent] = {}
_agent_lock: asyncio.Lock | None = None
_startup_time: float = 0.0

# Optional API key authentication
_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add request ID to each request for tracing."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        _request_id_ctx.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def get_request_id() -> str:
    """Get current request ID for logging context."""
    return _request_id_ctx.get()


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
    global _startup_time
    _startup_time = time.time()
    log.info("Initializing Sonality agent for API server")
    _agent_store["agent"] = SonalityAgent()
    yield
    log.info("Shutting down Sonality agent")
    agent = _agent_store.pop("agent", None)
    if agent is not None:
        agent.shutdown()


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

        async def generate() -> AsyncIterator[str]:
            async with _get_lock():
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
                for content, reasoning in stream_iter:
                    delta = {
                        k: v
                        for k, v in [("content", content), ("reasoning_content", reasoning)]
                        if v
                    }
                    if delta:
                        yield sse_chunk(delta)
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

    prompt_tokens = sum(len(m.content.split()) for m in request.messages) * 4 // 3
    completion_tokens = len(response_text.split()) * 4 // 3

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
    return ModelsResponse(
        data=[ModelInfo(id=MODEL_ID, created=int(time.time()), owned_by=MODEL_ID)]
    )


@app.get("/v1/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str) -> ModelInfo:
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


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    log.info("HTTP /ingest chars=%d", len(request.text))
    agent = _get_agent()
    async with _get_lock():
        ess = await asyncio.to_thread(
            partial(agent.ingest, request.text, topic_override=request.topic_override)
        )
    return IngestResponse(
        success=True,
        score=ess.score,
        reasoning_type=ess.reasoning_type,
        belief_update_recommended=ess.belief_update_recommended,
        urgency=ess.urgency,
        topics=list(ess.topics),
        summary=ess.summary,
    )


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
