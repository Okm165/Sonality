# Testing

Sonality follows a testing philosophy that prioritizes meaningful behavior verification over coverage metrics. Each test earns its place by catching real bugs or verifying invariants that the type system cannot guarantee.

## Test Architecture

```
tests/
├── conftest.py                          # Environment defaults for isolation
├── shared/llm/test_parse.py             # LLM output parsing (49 tests)
├── sonality/test_api.py                 # API contract verification
├── sonality/memory/test_derivatives.py  # Chunk normalization + fallback
└── fathom/test_models.py                # Pydantic coercion edge cases
```

## Test Categories

### LLM Output Parsing (49 tests)

The largest test suite covers the output normalization pipeline in `shared/llm/parse.py`. These tests verify handling of:

- **Thinking removal** — `<think>` blocks, reasoning fences, XML tags from chain-of-thought models
- **JSON extraction** — Markdown fences, buried JSON in prose, escaped quote handling
- **Tool call parsing** — Standard format, malformed calls, missing parameters
- **Response sanitization** — Fake XML injection, leaked system prompts, bare tool call artifacts
- **Pydantic normalization** — List wrapping for single items, field type coercion

These tests exist because quantized local models produce non-standard output. Each test case represents a real artifact encountered in production with models ranging from 2-bit to 8-bit quantization.

### API Contract (test_api.py)

Verifies the FastAPI server behavior:

- `/v1/chat/completions` endpoint contract (request/response shapes)
- `/health` endpoint availability
- `/ingest` endpoint for external content feeding
- Error handling (invalid requests, missing fields)

### Derivative Normalization (test_derivatives.py)

Tests the LLM semantic chunking pipeline:

- Chunk count within bounds (1–15 per episode)
- Handling of empty/minimal input
- Fallback behavior when LLM produces invalid chunk structures

### Model Coercion (test_models.py)

Tests Fathom's Pydantic model resilience:

- `Checklist` item parsing from various LLM output formats
- `Fact` extraction with partial/missing fields
- Type coercion for numeric fields returned as strings

## Running Tests

```bash
make test              # All tests (excludes live API tests)
make test-report       # Tests with JSON/XML report output
uv run pytest tests -v # Direct pytest invocation
```

### Live Tests

Tests requiring external API access are marked `live` and excluded from default runs:

```bash
uv run pytest tests -m "live" -v  # Only live tests (requires configured .env)
```

## Testing Philosophy

**Integration over mocks.** Tests use real Pydantic models, real parsing functions, and real normalization pipelines. Mocking is used only for external I/O boundaries (HTTP calls, database connections).

**Edge cases over happy paths.** The type system already guarantees happy-path behavior for most operations. Tests focus on boundary conditions: malformed input, partial failures, quantization artifacts.

**Fewer tests, more value.** Each test must justify its existence by catching a category of bugs. Tests that verify obvious behavior or implementation details are removed.

**Reproducibility.** All tests run without external dependencies (no API keys, no databases, no network). Environment defaults in `conftest.py` ensure isolation.

## Test Configuration

From `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-v --tb=short -m 'not live'"
timeout = 1800
```

The 1800-second timeout accommodates slow CI environments. Local runs typically complete in under 30 seconds.

## Adding Tests

When adding tests:

1. Place tests in the appropriate subdirectory matching the source module structure
2. Use the `live` marker for anything requiring external services
3. Prefer testing at module boundaries rather than internal implementation
4. Each test should fail if the behavior it tests is broken, and pass otherwise
5. Name test functions to describe the specific behavior being verified
