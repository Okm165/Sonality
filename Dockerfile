# Redirects to docker/sonality.Dockerfile.
# Use `docker build -f docker/sonality.Dockerfile .` or `docker compose up`.
#
# This file copies docker/sonality.Dockerfile verbatim so that bare
# `docker build .` still works for CI scripts that reference the root path.

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

RUN uv run --no-dev python -c \
    "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-large-en-v1.5').embed(['warmup']))"

RUN mkdir -p data

ENTRYPOINT ["uv", "run", "--no-dev", "sonality-server"]
