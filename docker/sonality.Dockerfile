FROM python:3.13-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV FASTEMBED_CACHE_PATH=/app/models

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Pre-download BAAI/bge-large-en-v1.5 ONNX into the image (~300 MB).
RUN uv run --no-dev python -c \
    "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-large-en-v1.5').embed(['warmup']))"

RUN mkdir -p data

ENTRYPOINT ["uv", "run", "--no-dev", "sonality-server"]
