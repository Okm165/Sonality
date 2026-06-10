FROM python:3.13-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_PROGRESS=1
ENV RUST_LOG=warn

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Playwright connects to remote browserless service via CDP — no local browser needed.
# Install only the pip package for the API (deps already satisfied by uv sync).

EXPOSE 8010

ENTRYPOINT ["uv", "run", "--no-dev", "fathom-server", "--host", "0.0.0.0", "--port", "8010"]
