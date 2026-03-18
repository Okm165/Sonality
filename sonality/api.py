"""FastAPI wrapper for programmatic Sonality access.

Provides HTTP endpoints for data ingestion, belief querying, probability estimation,
and OpenAI-compatible chat completions for integration with external LLM tools.
Uses a singleton SonalityAgent with a threading lock to prevent concurrent mutation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent import SonalityAgent

log = logging.getLogger(__name__)

_agent_store: dict[str, SonalityAgent] = {}
# asyncio.Lock avoids blocking the event loop during long ingest/respond operations,
# keeping the /health endpoint responsive even when the agent is busy.
_agent_lock: asyncio.Lock


def _get_agent() -> SonalityAgent:
    agent = _agent_store.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _agent_lock
    _agent_lock = asyncio.Lock()
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
    version="0.1.0",
    lifespan=lifespan,
)


# --- OpenAI-Compatible Chat Completions API ---


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    model: str = Field(
        default="sonality", description="Model identifier (ignored, always uses Sonality)"
    )
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    stream: bool = Field(default=False, description="Stream responses (not yet supported)")


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
    """OpenAI-compatible chat completions endpoint.

    Enables Sonality to be used as an LLM model by external tools like
    LangChain, AutoGen, OpenAI SDK, and other OpenAI-compatible clients.

    The last user message is processed through Sonality's respond() method,
    which includes ESS classification, belief updates, and personality-aware responses.
    """
    agent = _get_agent()
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")
    user_message = user_messages[-1].content

    if request.stream:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        def sse_chunk(delta: dict[str, str], finish: str | None = None) -> str:
            return f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'sonality', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]})}\n\n"

        def generate() -> Iterator[str]:
            for content, reasoning in agent.respond_stream(user_message):
                delta = {k: v for k, v in [("content", content), ("reasoning_content", reasoning)] if v}
                if delta:
                    yield sse_chunk(delta)
            yield sse_chunk({}, "stop")
            yield "data: [DONE]\n\n"

        async with _agent_lock:
            return StreamingResponse(generate(), media_type="text/event-stream")

    async with _agent_lock:
        response_text = await asyncio.to_thread(agent.respond, user_message)

    prompt_tokens = sum(len(m.content.split()) for m in request.messages) * 4 // 3
    completion_tokens = len(response_text.split()) * 4 // 3

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model="sonality",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
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
        data=[
            ModelInfo(
                id="sonality",
                created=int(time.time()),
                owned_by="sonality",
            )
        ]
    )


@app.get("/v1/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str) -> ModelInfo:
    """Get model info (OpenAI-compatible)."""
    if model_id != "sonality":
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return ModelInfo(
        id="sonality",
        created=int(time.time()),
        owned_by="sonality",
    )


# --- Embeddings API (OpenAI-compatible) ---


class EmbeddingRequest(BaseModel):
    input: str | list[str] = Field(..., description="Text(s) to embed")
    model: str = Field(default="sonality-embed", description="Model identifier")


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: ChatCompletionUsage


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Create embeddings for text (OpenAI-compatible).

    Uses the same embedding model as Sonality's internal retrieval system.
    """
    agent = _get_agent()
    texts = [request.input] if isinstance(request.input, str) else request.input
    embeddings = agent._embedder.embed_documents(texts)
    total_tokens = sum(len(t.split()) for t in texts) * 4 // 3
    return EmbeddingResponse(
        data=[EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)],
        model="sonality-embed",
        usage=ChatCompletionUsage(
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens,
        ),
    )


# --- Simple Chat Endpoint ---


class SimpleChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to Sonality")


class SimpleChatResponse(BaseModel):
    response: str
    ess_score: float
    reasoning_type: str
    topics: list[str]


@app.post("/chat", response_model=SimpleChatResponse)
async def simple_chat(request: SimpleChatRequest) -> SimpleChatResponse:
    """Simple chat endpoint for quick interactions.

    Simpler than the OpenAI-compatible endpoint, returns response with ESS metadata.
    """
    agent = _get_agent()
    async with _agent_lock:
        response_text = await asyncio.to_thread(agent.respond, request.message)
        ess = agent.last_ess
    return SimpleChatResponse(
        response=response_text,
        ess_score=ess.score,
        reasoning_type=str(ess.reasoning_type),
        topics=list(ess.topics),
    )


# --- Sonality-Specific Endpoints ---


class IngestRequest(BaseModel):
    text: str = Field(..., description="Content to ingest (news article, social media post, etc.)")
    topic_override: str = Field(
        default="",
        description="Optional canonical topic name to use instead of LLM-extracted topics",
    )


class IngestResponse(BaseModel):
    success: bool
    score: float
    reasoning_type: str
    belief_update_recommended: bool
    urgency: str
    topics: list[str]
    summary: str


class BeliefResponse(BaseModel):
    topic: str
    position: float
    confidence: float
    evidence_count: int
    uncertainty: float


class ProbabilityRequest(BaseModel):
    base_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Prior probability from general knowledge or caller's context",
    )


class ProbabilityResponse(BaseModel):
    topic: str
    probability: float
    evidence_weight: float
    opinion: float
    confidence: float
    evidence_count: int
    raw_probability: float


class CorrelationResponse(BaseModel):
    source_topic: str
    target_topic: str
    correlation_type: str
    strength: float
    reasoning: str


class HealthResponse(BaseModel):
    version: int
    interaction_count: int
    belief_count: int
    topic_count: int
    staged_updates: int


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Ingest non-conversational content (news, social media, reports).

    Bypasses response generation and runs the evidence assessment pipeline directly.
    """
    agent = _get_agent()
    async with _agent_lock:
        ess = await asyncio.to_thread(agent.ingest, request.text, request.topic_override)
    return IngestResponse(
        success=True,
        score=ess.score,
        reasoning_type=str(ess.reasoning_type),
        belief_update_recommended=ess.belief_update_recommended,
        urgency=str(ess.urgency),
        topics=list(ess.topics),
        summary=ess.summary,
    )


@app.get("/beliefs", response_model=list[BeliefResponse])
async def get_beliefs() -> list[BeliefResponse]:
    """Return all current belief states, sorted by absolute position strength."""
    agent = _get_agent()
    beliefs = [
        BeliefResponse(
            topic=topic,
            position=b.position,
            confidence=b.confidence,
            evidence_count=b.evidence_count,
            uncertainty=b.uncertainty,
        )
        for topic in agent.sponge.opinion_vectors
        for b in (agent.sponge.get_belief(topic),)
    ]
    return sorted(beliefs, key=lambda b: abs(b.position), reverse=True)


@app.get("/beliefs/{topic}", response_model=BeliefResponse)
async def get_belief_endpoint(topic: str) -> BeliefResponse:
    """Return belief state for a specific topic."""
    agent = _get_agent()
    b = agent.sponge.get_belief(topic)
    return BeliefResponse(
        topic=b.topic,
        position=b.position,
        confidence=b.confidence,
        evidence_count=b.evidence_count,
        uncertainty=b.uncertainty,
    )


@app.post("/beliefs/{topic}/probability", response_model=ProbabilityResponse)
async def estimate_probability(topic: str, request: ProbabilityRequest) -> ProbabilityResponse:
    """Estimate calibrated probability for a topic using Platt scaling."""
    agent = _get_agent()
    estimate = agent.sponge.estimate_probability(topic, base_rate=request.base_rate)
    return ProbabilityResponse(
        topic=estimate.topic,
        probability=estimate.probability,
        evidence_weight=estimate.evidence_weight,
        opinion=estimate.opinion,
        confidence=estimate.confidence,
        evidence_count=estimate.evidence_count,
        raw_probability=estimate.raw_probability,
    )


@app.get("/probability/{topic}", response_model=ProbabilityResponse)
async def estimate_probability_compat(topic: str, base_rate: float = 0.5) -> ProbabilityResponse:
    """Plan-compatible probability endpoint alias."""
    return await estimate_probability(topic, ProbabilityRequest(base_rate=base_rate))


@app.get("/beliefs/{topic}/correlations", response_model=list[CorrelationResponse])
async def get_correlations(topic: str) -> list[CorrelationResponse]:
    """Return all belief correlations for a topic."""
    agent = _get_agent()
    correlations = agent._run_async(agent._graph.get_belief_correlations(topic))
    return [
        CorrelationResponse(
            source_topic=c.source_topic,
            target_topic=c.target_topic,
            correlation_type=str(c.correlation_type),
            strength=c.strength,
            reasoning=c.reasoning,
        )
        for c in correlations
    ]


@app.get("/correlations/{topic}", response_model=list[CorrelationResponse])
async def get_correlations_compat(topic: str) -> list[CorrelationResponse]:
    """Plan-compatible correlations endpoint alias."""
    return await get_correlations(topic)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return agent health status."""
    agent = _get_agent()
    return HealthResponse(
        version=agent.sponge.version,
        interaction_count=agent.sponge.interaction_count,
        belief_count=len(agent.sponge.opinion_vectors),
        topic_count=len(agent.sponge.behavioral_signature.topic_engagement),
        staged_updates=len(agent.sponge.staged_opinion_updates),
    )
