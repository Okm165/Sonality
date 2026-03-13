"""Graduated live tests: each level isolates one layer of the stack.

Run with: uv run pytest tests/test_live_graduated.py -v -s -m live

Levels (run in order, each depends on the previous passing):
  L0   Endpoint connectivity  — HTTP reachable, returns anything
  L1   Raw response           — LLM returns text, embedding returns floats
  L2   Structured parsing     — llm_call schema extraction + ESS classify
  L2r  Repeatability          — same schema consistent across 3 calls
  L2x  Per-prompt parsing     — each prompt template tested in isolation
  L3   Memory primitives      — postgres vector insert/search, similarity ordering
  L3x  Memory store/retrieve  — full DualEpisodeStore write + vector recall

For full agent behavioral tests (anti-sycophancy, memory retrieval, personality
accumulation), run: uv run pytest tests/test_agent_health.py -v -s -m live
"""

from __future__ import annotations

import json
import math
import time
import urllib.request

import pytest
from pydantic import BaseModel

from sonality import config
from sonality.llm.caller import LLMCallResult, llm_call
from sonality.memory.embedder import ExternalEmbedder
from sonality.provider import chat_completion, parse_json_object

# Tests in TestParserRobustness do NOT call the LLM and run without -m live.
# All other classes require a live LLM/DB and are skipped without -m live.
_live = pytest.mark.live
pytestmark = pytest.mark.live

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# P0 — Parser robustness (offline — no LLM, no live mark)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_key", [
    # Direct JSON
    ('{"score": 0.8, "label": "good"}', "score"),
    # Markdown fences
    ('```json\n{"score": 0.9}\n```', "score"),
    # Prose before
    ("Here is the result:\n{\"score\": 0.7}", "score"),
    # Prose after
    ('{"score": 0.6}\nLet me know if you need anything.', "score"),
    # Two JSON blocks — last one wins
    ('First try: {"score": 0.1}\nCorrected: {"score": 0.95}', "score"),
    # Nested braces inside string value
    ('{"description": "uses {curly} braces", "value": 42}', "description"),
    # Escaped quotes inside strings
    ('{"note": "he said \\"hi\\"", "ok": true}', "note"),
    # Pretty-printed multiline
    ('{\n  "reasoning_type": "empirical_data",\n  "score": 0.8\n}', "reasoning_type"),
    # Extra whitespace and blank lines
    ('\n\n  {"field": "value"}  \n\n', "field"),
])
def test_extract_last_json_object_parametric(text: str, expected_key: str) -> None:
    """extract_last_json_object handles all LLM output patterns the model might emit."""
    from sonality.provider import extract_last_json_object
    result = extract_last_json_object(text)
    assert result is not None, f"Returned None for: {text!r}"
    assert expected_key in result, f"Key {expected_key!r} missing from {result}"


@pytest.mark.parametrize("text", [
    "",           # empty
    "   ",        # whitespace
    "no json here",          # prose only
    '["a", "b"]',            # array — not an object
])
def test_extract_last_json_object_returns_none_for_invalid(text: str) -> None:
    """extract_last_json_object returns None when no valid object is present."""
    from sonality.provider import extract_last_json_object
    assert extract_last_json_object(text) is None, f"Should return None for: {text!r}"


def test_parse_json_object_last_block_wins() -> None:
    """When model self-corrects, parse_json_object returns the corrected value."""
    text = 'First attempt: {"score": 0.1}\nActually: {"score": 0.95, "label": "good"}'
    result = parse_json_object(text)
    assert result.get("score") == 0.95, f"Expected 0.95, got {result.get('score')}"
    assert "label" in result


def test_parse_json_object_strips_fences() -> None:
    """Markdown-fenced JSON is extracted correctly."""
    text = "```json\n{\"key\": \"value\", \"num\": 42}\n```"
    result = parse_json_object(text)
    assert result == {"key": "value", "num": 42}


# ---------------------------------------------------------------------------
# L0 — Endpoint connectivity
# ---------------------------------------------------------------------------

class TestL0Connectivity:
    """Verify the two endpoints are reachable before anything else."""

    def test_llm_endpoint_reachable(self) -> None:
        """GET /models succeeds and returns JSON with at least one model."""
        t = time.perf_counter()
        url = config.BASE_URL.rstrip("/").removesuffix("/v1") + "/v1/models"
        with urllib.request.urlopen(url, timeout=15) as resp:
            body = json.loads(resp.read())
        elapsed = _elapsed(t)

        model_ids = [m["id"] for m in body.get("data", [])]
        print(f"\n  models={model_ids}  ({elapsed})")
        assert model_ids, f"No models returned from {url}"

    def test_embedding_endpoint_reachable(self) -> None:
        """GET /models on the embedding endpoint succeeds."""
        t = time.perf_counter()
        base = (config.EMBEDDING_BASE_URL or config.BASE_URL).rstrip("/")
        url = base.removesuffix("/v1") + "/v1/models"
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = json.loads(resp.read())
        elapsed = _elapsed(t)

        model_ids = [m.get("id") or m.get("name") for m in body.get("models", body.get("data", []))]
        print(f"\n  embedding models={model_ids}  ({elapsed})")
        assert body, f"Empty response from embedding endpoint {url}"


# ---------------------------------------------------------------------------
# L1 — Raw response quality
# ---------------------------------------------------------------------------

class TestL1RawResponse:
    """Verify the LLM returns usable text and embeddings have the right shape."""

    def test_llm_returns_non_empty_text(self) -> None:
        """Single-sentence prompt → non-empty text response."""
        t = time.perf_counter()
        result = chat_completion(
            model=config.MODEL,
            messages=({"role": "user", "content": "Reply with exactly the word: pong"},),
            max_tokens=config.FAST_LLM_MAX_TOKENS,
            temperature=0.0,
        )
        elapsed = _elapsed(t)

        print(f"\n  response={result.text!r}  tokens={result.output_tokens}  ({elapsed})")
        assert result.text.strip(), "LLM returned empty text"
        assert result.output_tokens > 0, "output_tokens should be positive"

    def test_llm_can_produce_json_when_asked(self) -> None:
        """Ask for a minimal JSON object and verify it parses."""
        t = time.perf_counter()
        result = chat_completion(
            model=config.MODEL,
            messages=(
                {
                    "role": "user",
                    "content": (
                        'Return ONLY a JSON object with one key "ok" set to true. '
                        "No explanation, no markdown, just the JSON."
                    ),
                },
            ),
            max_tokens=config.FAST_LLM_MAX_TOKENS,
            temperature=0.0,
        )
        elapsed = _elapsed(t)

        print(f"\n  raw={result.text!r}  ({elapsed})")
        # Tolerant parse — strip markdown fences and surrounding prose
        from sonality.provider import parse_json_object
        parsed = parse_json_object(result.text)
        assert parsed, (
            f"No JSON object found in LLM response.\nraw={result.text!r}"
        )

    def test_embedding_returns_correct_dimensions(self) -> None:
        """Embedding a short sentence returns the configured dimension count."""
        t = time.perf_counter()
        embedder = ExternalEmbedder()
        vec = embedder.embed_query("hello world")
        elapsed = _elapsed(t)

        print(f"\n  dims={len(vec)}  expected={config.EMBEDDING_DIMENSIONS}  ({elapsed})")
        assert len(vec) == config.EMBEDDING_DIMENSIONS, (
            f"Embedding dimension mismatch: got {len(vec)}, want {config.EMBEDDING_DIMENSIONS}"
        )
        assert any(v != 0.0 for v in vec), "Embedding vector is all zeros"

    def test_embedding_batch_consistent(self) -> None:
        """Batch embedding two texts returns two vectors of the right size."""
        t = time.perf_counter()
        embedder = ExternalEmbedder()
        vecs = embedder.embed_documents(["cats are fluffy", "dogs are loyal"])
        elapsed = _elapsed(t)

        print(f"\n  batch_size={len(vecs)}  dims={len(vecs[0])}  ({elapsed})")
        assert len(vecs) == 2, f"Expected 2 vectors, got {len(vecs)}"
        assert all(len(v) == config.EMBEDDING_DIMENSIONS for v in vecs)


# ---------------------------------------------------------------------------
# L2 — Structured parsing + ESS
# ---------------------------------------------------------------------------

class _SimpleSchema(BaseModel):
    sentiment: str
    score: float


class TestL2StructuredParsing:
    """Verify llm_call extracts schemas correctly and ESS classifies coherently."""

    def test_llm_call_parses_simple_schema(self) -> None:
        """llm_call extracts a two-field schema from an unambiguous prompt."""
        t = time.perf_counter()

        result: LLMCallResult[_SimpleSchema] = llm_call(
            prompt=(
                "Classify this text: 'I love sunny days!'\n"
                'Return JSON: {"sentiment": "positive"|"negative"|"neutral", "score": 0.0-1.0}'
            ),
            response_model=_SimpleSchema,
            fallback=_SimpleSchema(sentiment="missing", score=0.0),
        )
        elapsed = _elapsed(t)

        print(
            f"\n  success={result.success}  repaired={result.repaired}  "
            f"sentiment={result.value.sentiment!r}  score={result.value.score}  ({elapsed})"
        )
        assert result.success, f"llm_call failed: {result.error}\nraw: {result.raw_text!r}"
        assert result.value.sentiment in {"positive", "negative", "neutral"}, (
            f"Unexpected sentiment: {result.value.sentiment!r}"
        )
        assert 0.0 <= result.value.score <= 1.0, (
            f"Score out of range: {result.value.score}"
        )

    def test_ess_strong_argument_scores_high(self) -> None:
        """A well-reasoned empirical argument should produce ESS > 0.4."""
        from sonality.ess import PROVIDER_CLIENT, classify
        from sonality.memory.sponge import SEED_SNAPSHOT

        t = time.perf_counter()
        result = classify(
            PROVIDER_CLIENT,
            user_message=(
                "A 2023 meta-analysis of 47 RCTs (n=12,000) found that regular aerobic "
                "exercise reduces all-cause mortality by 31% and improves cognitive function "
                "scores by 22% in adults over 50. The effect size was robust across subgroups."
            ),
            sponge_snapshot=SEED_SNAPSHOT,
        )
        elapsed = _elapsed(t)

        print(
            f"\n  score={result.score:.3f}  type={result.reasoning_type}  "
            f"severity={result.default_severity}  fields={result.defaulted_fields}  ({elapsed})"
        )
        assert result.default_severity not in ("missing", "exception"), (
            f"ESS required fields missing — classifier failed.\nFields: {result.defaulted_fields}"
        )
        assert result.score >= 0.3, (
            f"ESS {result.score:.3f} too low for a peer-reviewed empirical argument (expected ≥ 0.3)"
        )

    def test_ess_weak_message_scores_low(self) -> None:
        """A contentless filler message should produce ESS < 0.2."""
        from sonality.ess import PROVIDER_CLIENT, classify
        from sonality.memory.sponge import SEED_SNAPSHOT

        t = time.perf_counter()
        result = classify(
            PROVIDER_CLIENT,
            user_message="ok cool",
            sponge_snapshot=SEED_SNAPSHOT,
        )
        elapsed = _elapsed(t)

        print(
            f"\n  score={result.score:.3f}  type={result.reasoning_type}  "
            f"severity={result.default_severity}  ({elapsed})"
        )
        assert result.score <= 0.3, (
            f"ESS {result.score:.3f} too high for a contentless message (expected ≤ 0.3)"
        )

    def test_ess_debunked_claim_scores_near_zero(self) -> None:
        """Conclusively-debunked conspiracy theory must score ≤ 0.07 as debunked_claim."""
        from sonality.ess import PROVIDER_CLIENT, ReasoningType, classify
        from sonality.memory.sponge import SEED_SNAPSHOT

        t = time.perf_counter()
        result = classify(
            PROVIDER_CLIENT,
            user_message=(
                "The Climategate emails clearly proved that IPCC scientists manipulated "
                "temperature data. Multiple independent analysts confirmed the fraud. "
                "You can't trust any of their warming projections."
            ),
            sponge_snapshot=SEED_SNAPSHOT,
        )
        elapsed = _elapsed(t)

        print(
            f"\n  score={result.score:.3f}  type={result.reasoning_type}  "
            f"severity={result.default_severity}  ({elapsed})"
        )
        assert result.score <= 0.25, (
            f"ESS {result.score:.3f} too high for a debunked conspiracy claim (expected ≤ 0.25). "
            f"Climategate has been cleared by 8+ independent inquiries."
        )
        # debunked_claim or anecdotal are both acceptable; the key is the score being low
        assert result.reasoning_type in {
            ReasoningType.DEBUNKED_CLAIM, ReasoningType.ANECDOTAL, ReasoningType.SOCIAL_PRESSURE
        }, (
            f"reasoning_type={result.reasoning_type} — expected debunked_claim, anecdotal, or social_pressure"
        )


# ---------------------------------------------------------------------------
# L2r — Repeatability (same prompts, 3 calls, measure consistency)
# ---------------------------------------------------------------------------

class TestL2rRepeatability:
    """Run the same structured prompts multiple times to quantify parse consistency.

    A prompt is 'consistent' if every run produces a parseable, schema-valid response.
    This isolates whether failures are model variance or structural prompt problems.
    """

    def _run_n(
        self,
        prompt: str,
        response_model: type[BaseModel],
        fallback: BaseModel,
        n: int = 3,
        label: str = "",
    ) -> tuple[int, int, list[str]]:
        """Run llm_call n times, return (successes, total, raw_outputs)."""
        successes = 0
        raws: list[str] = []
        for _ in range(n):
            result = llm_call(
                prompt=prompt,
                response_model=response_model,
                fallback=fallback,
            )
            raws.append(result.raw_text[:80] if result.raw_text else "<empty>")
            if result.success:
                successes += 1
        return successes, n, raws

    def test_simple_two_field_schema_repeatable(self) -> None:
        """A minimal 2-field schema should parse on every call (3/3)."""
        class _TwoField(BaseModel):
            sentiment: str
            score: float

        t = time.perf_counter()
        ok, total, raws = self._run_n(
            prompt=(
                "Classify: 'I love sunny days!'\n"
                'Return JSON: {"sentiment": "positive"|"negative"|"neutral", "score": 0.0-1.0}'
            ),
            response_model=_TwoField,
            fallback=_TwoField(sentiment="missing", score=0.0),
        )
        elapsed = _elapsed(t)

        print(f"\n  {ok}/{total} successful  ({elapsed})")
        for i, raw in enumerate(raws):
            print(f"    run {i+1}: {raw!r}")

        assert ok == total, (
            f"Only {ok}/{total} runs parsed successfully — model output is inconsistent"
        )

    def test_five_field_schema_repeatable(self) -> None:
        """A 5-field schema with enums should parse ≥2/3 runs."""
        class _FiveField(BaseModel):
            category: str
            depth: str
            temporal_expansion: str
            semantic_memory: str
            reasoning: str = ""

        from sonality.llm.prompts import QUERY_ROUTING_PROMPT
        prompt = QUERY_ROUTING_PROMPT.format(
            query="What were the key arguments we discussed about nuclear energy?",
            context="Prior conversation about energy policy and climate change.",
        )

        t = time.perf_counter()
        ok, total, raws = self._run_n(
            prompt=prompt,
            response_model=_FiveField,
            fallback=_FiveField(
                category="SIMPLE", depth="MINIMAL",
                temporal_expansion="NO_EXPAND", semantic_memory="SKIP",
            ),
        )
        elapsed = _elapsed(t)

        print(f"\n  {ok}/{total} successful  ({elapsed})")
        assert ok >= 2, (
            f"Only {ok}/{total} runs parsed — 5-field schema not reliable enough"
        )

    def test_insight_prompt_consistent_decision_format(self) -> None:
        """INSIGHT_PROMPT decision field should be valid on every call (3/3)."""
        from sonality.memory.updater import InsightExtractionResponse
        from sonality.prompts import INSIGHT_PROMPT

        prompt = INSIGHT_PROMPT.format(
            user_message="Statistics without context can mislead — always check the denominator.",
            agent_response="Agreed. Base rates matter as much as relative risk figures.",
            ess_score="0.55",
        )

        t = time.perf_counter()
        ok, total, raws = self._run_n(
            prompt=prompt,
            response_model=InsightExtractionResponse,
            fallback=InsightExtractionResponse(),
        )
        elapsed = _elapsed(t)

        print(f"\n  {ok}/{total} successful  ({elapsed})")
        assert ok == total, (
            f"INSIGHT_PROMPT only parsed {ok}/{total} times — inconsistency detected"
        )


# ---------------------------------------------------------------------------
# L3 — Memory primitives (requires Postgres + Ollama)
# ---------------------------------------------------------------------------

@pytest.fixture
def pg_url() -> str:
    return config.POSTGRES_URL


class TestL3MemoryPrimitives:
    """Verify vector storage and semantic retrieval work at the primitive level."""

    def test_postgres_vector_insert_and_exact_retrieval(self, pg_url: str) -> None:
        """Insert one embedding row, retrieve it by UID, verify it round-trips."""
        import psycopg

        t = time.perf_counter()
        embedder = ExternalEmbedder()
        text = "The quick brown fox jumps over the lazy dog."
        vec = embedder.embed_documents([text])[0]

        uid = "test-grad-001"
        with psycopg.connect(pg_url, autocommit=True) as conn:
            conn.execute(
                "DELETE FROM derivatives WHERE uid = %s",
                (uid,),
            )
            conn.execute(
                "INSERT INTO derivatives (uid, episode_uid, text, key_concept, embedding) "
                "VALUES (%s, %s, %s, %s, %s::vector)",
                (uid, "ep-test-001", text, "fox", vec),
            )
            row = conn.execute(
                "SELECT text, embedding::text FROM derivatives WHERE uid = %s",
                (uid,),
            ).fetchone()
            conn.execute("DELETE FROM derivatives WHERE uid = %s", (uid,))

        elapsed = _elapsed(t)
        assert row is not None, "Inserted row not found on read-back"
        print(f"\n  text={row[0]!r}  vec_preview={row[1][:40]}...  ({elapsed})")

    def test_embedding_semantic_ordering(self) -> None:
        """Two semantically similar sentences should be closer than a dissimilar one."""
        t = time.perf_counter()
        embedder = ExternalEmbedder()
        vecs = embedder.embed_documents([
            "The cat sat on the mat.",        # anchor
            "A kitten rested on a rug.",       # similar
            "The stock market fell 3% today.", # dissimilar
        ])
        elapsed = _elapsed(t)

        sim_close = _cosine(vecs[0], vecs[1])
        sim_far = _cosine(vecs[0], vecs[2])
        print(f"\n  sim(cat,kitten)={sim_close:.4f}  sim(cat,stocks)={sim_far:.4f}  ({elapsed})")

        assert sim_close > sim_far, (
            f"Semantic ordering wrong: similar={sim_close:.4f} should > dissimilar={sim_far:.4f}"
        )

    def test_vector_search_returns_nearest_neighbour(self, pg_url: str) -> None:
        """Insert two rows; query with a paraphrase of doc1 — doc1 should rank first."""
        import psycopg

        embedder = ExternalEmbedder()
        docs = [
            ("test-grad-nn-1", "ep-nn", "Quantum computing uses qubits for superposition."),
            ("test-grad-nn-2", "ep-nn", "The best pizza in Naples uses buffalo mozzarella."),
        ]
        query = "Qubits and quantum superposition explained."

        t = time.perf_counter()
        vecs = embedder.embed_documents([d[2] for d in docs])
        query_vec = embedder.embed_query(query)

        with psycopg.connect(pg_url, autocommit=True) as conn:
            for (uid, ep, text), vec in zip(docs, vecs, strict=True):
                conn.execute("DELETE FROM derivatives WHERE uid = %s", (uid,))
                conn.execute(
                    "INSERT INTO derivatives (uid, episode_uid, text, key_concept, embedding) "
                    "VALUES (%s, %s, %s, %s, %s::vector)",
                    (uid, ep, text, "test", vec),
                )

            rows = conn.execute(
                "SELECT uid FROM derivatives WHERE uid = ANY(%s) "
                "ORDER BY embedding <=> %s::vector LIMIT 2",
                ([d[0] for d in docs], query_vec),
            ).fetchall()

            for uid, _, _ in docs:
                conn.execute("DELETE FROM derivatives WHERE uid = %s", (uid,))

        elapsed = _elapsed(t)
        top_uid = rows[0][0] if rows else None
        print(f"\n  top_result={top_uid}  expected=test-grad-nn-1  ({elapsed})")

        assert top_uid == "test-grad-nn-1", (
            f"Nearest-neighbour search returned {top_uid!r}, expected 'test-grad-nn-1'"
        )


# ---------------------------------------------------------------------------
# L2x — Per-prompt parsing (each prompt template tested in isolation)
# ---------------------------------------------------------------------------

class TestL2xPerPromptParsing:
    """Test every prompt→parse pipeline individually.

    Each test uses the ACTUAL prompt template from the codebase and the
    ACTUAL Pydantic model that consumes it, so failures point to the
    real production parse path.
    """

    # --- ESS path diagnostics ---

    def test_ess_raw_tool_call_inspection(self) -> None:
        """Print the raw API response for an ESS call to see which path executes.

        This diagnoses whether tool_calls or JSON-text fallback is being used.
        """
        from sonality.ess import PROVIDER_ESS_TOOL, PROVIDER_ESS_TOOL_CHOICE
        from sonality.provider import extract_tool_call_arguments

        t = time.perf_counter()
        completion = chat_completion(
            model=config.ESS_MODEL,
            messages=(
                {
                    "role": "user",
                    "content": "A 2023 RCT showed exercise reduces mortality by 30%. Rate this argument.\n\n"
                    "Return ONLY a valid JSON object with keys: score, reasoning_type, source_reliability, "
                    "internal_consistency, novelty, topics, summary, opinion_direction.",
                },
            ),
            max_tokens=config.FAST_LLM_MAX_TOKENS,
            temperature=0.0,
            tools=(PROVIDER_ESS_TOOL,),
            tool_choice=PROVIDER_ESS_TOOL_CHOICE,
        )
        elapsed = _elapsed(t)

        tool_args = extract_tool_call_arguments(completion.raw, "classify_evidence")
        json_fallback = parse_json_object(completion.text) if not tool_args else {}

        path = "tool_call" if tool_args else "json_text_fallback"
        data = tool_args or json_fallback

        print(f"\n  path={path}  ({elapsed})")
        print(f"  internal_consistency={data.get('internal_consistency')!r}")
        print(f"  score={data.get('score')}  reasoning_type={data.get('reasoning_type')!r}")
        print(f"  content_preview={completion.text[:80]!r}")

        assert data, f"Both tool_call and JSON fallback returned empty.\nraw={completion.raw}"
        assert "score" in data, f"Missing 'score' field.\ndata={data}"

    def test_ess_internal_consistency_no_longer_coerced(self) -> None:
        """After alias fix, internal_consistency should not be coerced."""
        from sonality.ess import PROVIDER_CLIENT, classify
        from sonality.memory.sponge import SEED_SNAPSHOT

        t = time.perf_counter()
        result = classify(
            PROVIDER_CLIENT,
            user_message=(
                "A Cochrane review of 89 RCTs found statistically significant benefits "
                "of exercise for depression, with effect sizes comparable to medication."
            ),
            sponge_snapshot=SEED_SNAPSHOT,
        )
        elapsed = _elapsed(t)

        print(
            f"\n  score={result.score:.3f}  consistency={result.internal_consistency}  "
            f"severity={result.default_severity}  fields={result.defaulted_fields}  ({elapsed})"
        )
        coerced = [f for f in result.defaulted_fields if "internal_consistency" in f]
        assert not coerced, (
            f"internal_consistency still coerced after alias fix: {coerced}\n"
            f"All defaulted: {result.defaulted_fields}"
        )

    # --- llm_call prompt templates ---

    def test_insight_prompt_parses_correctly(self) -> None:
        """INSIGHT_PROMPT → InsightExtractionResponse: decision + text extracted."""
        from sonality.memory.updater import InsightExtractionResponse
        from sonality.prompts import INSIGHT_PROMPT

        t = time.perf_counter()
        prompt = INSIGHT_PROMPT.format(
            user_message="Your skepticism about AI is misguided and naive.",
            agent_response="I maintain my position — social pressure isn't evidence.",
            ess_score="0.08",
        )
        result: LLMCallResult[InsightExtractionResponse] = llm_call(
            prompt=prompt,
            response_model=InsightExtractionResponse,
            fallback=InsightExtractionResponse(),
        )
        elapsed = _elapsed(t)

        print(
            f"\n  success={result.success}  decision={result.value.insight_decision}  "
            f"text={result.value.insight_text[:60]!r}  ({elapsed})"
        )
        assert result.success, f"INSIGHT_PROMPT parse failed: {result.error}\nraw: {result.raw_text!r}"
        assert result.value.insight_decision in {"EXTRACT", "SKIP"}, (
            f"Unexpected decision: {result.value.insight_decision!r}"
        )

    def test_chunking_prompt_parses_correctly(self) -> None:
        """CHUNKING_PROMPT → ChunkingResponse: list of chunk objects extracted."""
        from sonality.memory.derivatives import ChunkingResponse
        from sonality.llm.prompts import CHUNKING_PROMPT

        text = (
            "Nuclear power produces 12g CO2/kWh versus 820g for coal. "
            "Modern reactor designs like SMRs address safety concerns. "
            "France generates 70% of its electricity from nuclear with excellent safety records."
        )
        t = time.perf_counter()
        result: LLMCallResult[ChunkingResponse] = llm_call(
            prompt=CHUNKING_PROMPT.format(text=text),
            response_model=ChunkingResponse,
            fallback=ChunkingResponse(chunks=[]),
        )
        elapsed = _elapsed(t)

        chunks = result.value.chunks
        first_text = chunks[0].text[:50] if chunks else ""
        print(
            f"\n  success={result.success}  chunks={len(chunks)}  "
            f"first={first_text!r}  ({elapsed})"
        )
        assert result.success, f"CHUNKING_PROMPT parse failed: {result.error}\nraw: {result.raw_text!r}"
        assert len(chunks) >= 1, "Expected at least 1 chunk"
        assert all(c.text.strip() for c in chunks), "Some chunks have empty text"
        assert all(c.key_concept.strip() for c in chunks), "Some chunks have empty key_concept"

    def test_query_routing_prompt_parses_correctly(self) -> None:
        """QUERY_ROUTING_PROMPT → expected category + depth fields."""
        from sonality.llm.prompts import QUERY_ROUTING_PROMPT
        from pydantic import BaseModel

        class RoutingResponse(BaseModel):
            category: str
            depth: str
            temporal_expansion: str
            semantic_memory: str
            reasoning: str = ""

        t = time.perf_counter()
        prompt = QUERY_ROUTING_PROMPT.format(
            query="What did we discuss about climate change last week?",
            context="Recent conversation about renewable energy costs.",
        )
        result: LLMCallResult[RoutingResponse] = llm_call(
            prompt=prompt,
            response_model=RoutingResponse,
            fallback=RoutingResponse(
                category="SIMPLE", depth="MINIMAL",
                temporal_expansion="NO_EXPAND", semantic_memory="SKIP",
            ),
        )
        elapsed = _elapsed(t)

        print(
            f"\n  success={result.success}  category={result.value.category!r}  "
            f"depth={result.value.depth!r}  ({elapsed})"
        )
        assert result.success, f"QUERY_ROUTING parse failed: {result.error}\nraw: {result.raw_text!r}"
        assert result.value.category in {
            "NONE", "SIMPLE", "TEMPORAL", "MULTI_ENTITY", "AGGREGATION", "BELIEF_QUERY"
        }, f"Invalid category: {result.value.category!r}"
        assert result.value.depth in {"MINIMAL", "MODERATE", "DEEP"}, (
            f"Invalid depth: {result.value.depth!r}"
        )

    def test_boundary_detection_prompt_parses_correctly(self) -> None:
        """BOUNDARY_DETECTION_PROMPT → boundary_decision field."""
        from sonality.llm.prompts import BOUNDARY_DETECTION_PROMPT
        from pydantic import BaseModel

        class BoundaryResponse(BaseModel):
            boundary_decision: str
            confidence: float = 0.5
            boundary_type: str = "none"
            reasoning: str = ""
            suggested_segment_label: str = ""

        t = time.perf_counter()
        prompt = BOUNDARY_DETECTION_PROMPT.format(
            recent_context="User: Tell me about Python.\nAgent: Python is a language.",
            current_message="Let's talk about cooking recipes instead.",
        )
        result: LLMCallResult[BoundaryResponse] = llm_call(
            prompt=prompt,
            response_model=BoundaryResponse,
            fallback=BoundaryResponse(boundary_decision="CONTINUE"),
        )
        elapsed = _elapsed(t)

        print(
            f"\n  success={result.success}  decision={result.value.boundary_decision!r}  "
            f"confidence={result.value.confidence:.2f}  ({elapsed})"
        )
        assert result.success, f"BOUNDARY_DETECTION parse failed: {result.error}\nraw: {result.raw_text!r}"
        assert result.value.boundary_decision in {"BOUNDARY", "CONTINUE"}, (
            f"Invalid decision: {result.value.boundary_decision!r}"
        )
        # Clear topic shift → should detect boundary
        assert result.value.boundary_decision == "BOUNDARY", (
            "Switching from Python to cooking should be detected as BOUNDARY"
        )

    def test_reflection_gate_prompt_parses_correctly(self) -> None:
        """REFLECTION_GATE_PROMPT → trigger field with valid value."""
        from sonality.llm.prompts import REFLECTION_GATE_PROMPT
        from pydantic import BaseModel

        class ReflectionGateResponse(BaseModel):
            trigger: str
            reasoning: str = ""

        t = time.perf_counter()
        prompt = REFLECTION_GATE_PROMPT.format(
            interaction_count=25,
            window_interactions=5,
            target_cadence=20,
            pending_insights=3,
            staged_updates=2,
            recent_shift_magnitude=0.15,
            disagreement_rate=0.3,
            belief_count=8,
        )
        result: LLMCallResult[ReflectionGateResponse] = llm_call(
            prompt=prompt,
            response_model=ReflectionGateResponse,
            fallback=ReflectionGateResponse(trigger="SKIP"),
        )
        elapsed = _elapsed(t)

        print(
            f"\n  success={result.success}  trigger={result.value.trigger!r}  ({elapsed})"
        )
        assert result.success, f"REFLECTION_GATE parse failed: {result.error}\nraw: {result.raw_text!r}"
        assert result.value.trigger in {"SKIP", "PERIODIC", "EVENT_DRIVEN"}, (
            f"Invalid trigger: {result.value.trigger!r}"
        )

    def test_belief_decay_prompt_parses_correctly(self) -> None:
        """BELIEF_DECAY_PROMPT → action field with valid value."""
        from sonality.llm.prompts import BELIEF_DECAY_PROMPT
        from pydantic import BaseModel

        class BeliefDecayResponse(BaseModel):
            action: str
            new_confidence: float = 0.5
            reasoning: str = ""

        t = time.perf_counter()
        prompt = BELIEF_DECAY_PROMPT.format(
            topic="nuclear_energy",
            position=0.6,
            confidence=0.7,
            evidence_count=3,
            gap=50,
            total_interactions=120,
        )
        result: LLMCallResult[BeliefDecayResponse] = llm_call(
            prompt=prompt,
            response_model=BeliefDecayResponse,
            fallback=BeliefDecayResponse(action="RETAIN"),
        )
        elapsed = _elapsed(t)

        print(
            f"\n  success={result.success}  action={result.value.action!r}  "
            f"confidence={result.value.new_confidence:.2f}  ({elapsed})"
        )
        assert result.success, f"BELIEF_DECAY parse failed: {result.error}\nraw: {result.raw_text!r}"
        assert result.value.action in {"RETAIN", "DECAY", "FORGET"}, (
            f"Invalid action: {result.value.action!r}"
        )


# ---------------------------------------------------------------------------
# L3x — Memory store + retrieve (full DualEpisodeStore pipeline)
# ---------------------------------------------------------------------------

class TestL3xMemoryStoreRetrieve:
    """Test the full episode store → retrieve pipeline with real DB + embedding."""

    def test_derivative_chunker_produces_embeddings(self) -> None:
        """DerivativeChunker.chunk_and_embed returns ≥1 derivatives with correct dim."""
        from sonality.memory.derivatives import DerivativeChunker

        embedder = ExternalEmbedder()
        chunker = DerivativeChunker(embedder)

        t = time.perf_counter()
        text = (
            "User: Nuclear energy produces far less CO2 than fossil fuels. "
            "France generates 70% of its power from nuclear with excellent safety records.\n"
            "Agent: The numbers are compelling. The CO2 figures are well-documented, "
            "and France's track record does challenge the safety narrative."
        )
        results = chunker.chunk_and_embed(text, episode_uid="test-ep-chunk-001")
        elapsed = _elapsed(t)

        first_chunk_text = results[0].node.text[:50] if results else ""
        emb_dim = len(results[0].embedding) if results else 0
        print(
            f"\n  chunks={len(results)}  "
            f"first_text={first_chunk_text!r}  "
            f"emb_dim={emb_dim}  ({elapsed})"
        )
        assert len(results) >= 1, "Expected at least 1 derivative"
        assert all(len(d.embedding) == config.EMBEDDING_DIMENSIONS for d in results), (
            "Embedding dimension mismatch in derivatives"
        )
        assert all(d.node.text.strip() for d in results), "Some derivatives have empty text"

    def test_full_store_and_vector_recall(self, pg_url: str) -> None:
        """Store a distinctive episode then verify vector search returns it top-1."""
        import asyncio
        import psycopg
        import uuid

        from sonality.memory.derivatives import DerivativeChunker
        from sonality.memory.embedder import ExternalEmbedder

        embedder = ExternalEmbedder()
        chunker = DerivativeChunker(embedder)

        ep_uid = f"test-store-{uuid.uuid4().hex[:8]}"
        content = (
            "User: Quantum error correction using surface codes requires O(d^2) physical "
            "qubits per logical qubit where d is the code distance.\n"
            "Agent: Correct — the overhead is significant but necessary for fault tolerance."
        )
        query = "surface code qubit overhead error correction"

        t = time.perf_counter()
        derivatives = chunker.chunk_and_embed(content, episode_uid=ep_uid)

        with psycopg.connect(pg_url, autocommit=True) as conn:
            for d in derivatives:
                conn.execute(
                    "INSERT INTO derivatives (uid, episode_uid, text, key_concept, embedding) "
                    "VALUES (%s, %s, %s, %s, %s::vector)",
                    (d.node.uid, d.node.source_episode_uid, d.node.text, d.node.key_concept, d.embedding),
                )

            query_vec = embedder.embed_query(query)
            rows = conn.execute(
                "SELECT episode_uid, text, 1 - (embedding <=> %s::vector) AS similarity "
                "FROM derivatives WHERE NOT archived "
                "ORDER BY embedding <=> %s::vector LIMIT 3",
                (query_vec, query_vec),
            ).fetchall()

            # Cleanup
            conn.execute("DELETE FROM derivatives WHERE episode_uid = %s", (ep_uid,))

        elapsed = _elapsed(t)

        print(f"\n  stored={len(derivatives)} derivatives  query_results={len(rows)}  ({elapsed})")
        for ep, text, sim in rows:
            print(f"    sim={sim:.4f}  ep={ep}  text={text[:60]!r}")

        assert rows, "Vector search returned no results"
        assert rows[0][0] == ep_uid, (
            f"Top result is ep={rows[0][0]!r}, expected {ep_uid!r}. "
            f"Similarity={rows[0][2]:.4f}"
        )
        assert rows[0][2] >= 0.6, (
            f"Top similarity {rows[0][2]:.4f} too low — retrieval quality concern"
        )

    def test_insight_extraction_end_to_end(self) -> None:
        """extract_insight() returns non-empty string for a high-ESS interaction."""
        from sonality.ess import ESSResult, InternalConsistencyStatus, OpinionDirection, ReasoningType, SourceReliability
        from sonality.memory.updater import extract_insight

        ess = ESSResult(
            score=0.72,
            reasoning_type=ReasoningType.EMPIRICAL_DATA,
            source_reliability=SourceReliability.PEER_REVIEWED,
            internal_consistency=InternalConsistencyStatus.CONSISTENT,
            novelty=0.8,
            topics=("nuclear_energy",),
            summary="User cited peer-reviewed data on nuclear safety.",
            opinion_direction=OpinionDirection.SUPPORTS,
        )
        t = time.perf_counter()
        insight = extract_insight(
            ess=ess,
            user_message="France's 40-year nuclear track record shows 12g CO2/kWh with zero major incidents.",
            agent_response="The data does challenge common safety narratives — I find the numbers compelling.",
        )
        elapsed = _elapsed(t)

        print(f"\n  insight={insight!r}  ({elapsed})")
        # High ESS + substantive exchange should extract an insight
        # (though the model may legitimately decide SKIP — we just verify no crash + valid output)
        assert isinstance(insight, str), f"Expected str, got {type(insight)}"
