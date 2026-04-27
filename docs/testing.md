# Testing

## Test Layers

| Layer | API Key | Command |
|-------|---------|---------|
| L0 Lint/Type | No | `make check-ci` |
| L1 Unit Tests | No | `pytest tests/` |
| L2 Benchmark Contracts | No | `pytest benches -m "bench and not live"` |
| L3 Live Benchmarks | Yes | `make bench-memory` / `make bench-personality` |
| L4 Full Teaching | Yes | `make bench-teaching` |

## Running Tests

```bash
pytest tests/                     # Unit (local DBs)
pytest tests/ --use-containers    # With testcontainers
pytest benches/ -v                # Benchmarks (mocked)
pytest benches/ --live -v         # Benchmarks (live LLM)
```

## Key Fixtures

### `mock_llm_call`

```python
def test_router(mock_llm_call):
    mock_llm_call({"Classify": {"category": "SIMPLE"}})
    assert route_query("...").category == QueryCategory.SIMPLE
```

### `db_containers`

Session-scoped Neo4j + Qdrant via testcontainers.

## Test Categories

| Category | Location | Purpose |
|----------|----------|---------|
| Unit | `tests/` | ESS, chunking, boundaries |
| Integration | `tests/memory/` | Router, retrieval, forgetting |
| Live | `tests/test_live_graduated.py` | Full agent |
| Benchmarks | `benches/` | Multi-dimensional evaluation |

## Live Preconditions

```bash
SONALITY_BASE_URL=http://localhost:11434/v1
SONALITY_MODEL=qwen2.5:14b-instruct
make preflight-live
```

## Benchmark Artifacts

Output in `data/teaching_bench/`:
- `summary.json` — Top-level outcome
- `release_readiness.json` — Gate decisions
- `observer_verdict_trace.jsonl` — Per-step verdicts

## Triage

1. Read `release_readiness.json`
2. If blocked → inspect hard-gate failures
3. If unstable → inspect `health_summary.json`
4. For step-level → `observer_verdict_trace.jsonl`
