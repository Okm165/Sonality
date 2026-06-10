"""Unified tool system — symmetric schema + executor for every agent tool.

Architecture: each tool module (memory, web) exports two parallel
registries: DEFINITIONS (OpenAI-compatible function schemas for the LLM) and
EXECUTORS (name → callable). This module lazily merges them at first use.

ToolContext carries all runtime dependencies (graph, embedder, web client, etc.)
so tool executors never import from agent.py — the dependency arrow is one-way.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

import structlog

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient

    from shared.embedder import Embedder

    from ..memory import DualEpisodeStore, MemoryGraph
    from ..request_identity import IdentityBundle
    from ..web_client import ResearchClient

log = structlog.get_logger(__name__)

_T = TypeVar("_T")


class RunAsync(Protocol):
    def __call__(self, coro: Coroutine[object, object, _T], /) -> _T: ...


ProgressCallback = Callable[[str], None]


def _noop_progress(_detail: str) -> None:
    pass


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Runtime dependencies for tool execution, created per dispatch call.

    The agent populates this from its own state. Tools never import from agent.
    ``research_transcript`` provides a read-only view of the agent's current
    memory state as a formatted string.
    """

    run_async: RunAsync
    web_client: ResearchClient | None
    graph: MemoryGraph
    dual_store: DualEpisodeStore
    qdrant: AsyncQdrantClient
    embedder: Embedder
    identity: IdentityBundle | None
    retrieve: Callable[[str], list[str]]
    research_transcript: Callable[[], str] = lambda: ""
    short_term_memory: str = ""
    progress: ProgressCallback = _noop_progress


ToolLabeler = Callable[[dict[str, object]], str]

_DISPATCH: dict[str, Callable[[dict[str, object], ToolContext], str]] = {}
_LABELS: dict[str, ToolLabeler] = {}
_DEFINITIONS: list[dict[str, object]] = []
_LOADED = False


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    from . import memory, web

    for mod in (memory, web):
        _DEFINITIONS.extend(mod.DEFINITIONS)
        _DISPATCH.update(mod.EXECUTORS)
        _LABELS.update(mod.LABELS)
    _LOADED = True


def get_definitions() -> list[dict[str, object]]:
    """All tool schemas for the LLM."""
    _load()
    return list(_DEFINITIONS)


def tool_label(name: str, args: dict[str, object]) -> str:
    """Extract a concise display label from tool args (e.g. the query or goal)."""
    _load()
    labeler = _LABELS.get(name)
    if labeler is None:
        return name
    return labeler(args)[:120]


def dispatch_tool(name: str, args: dict[str, object], ctx: ToolContext) -> str:
    """Execute a tool by name. Returns text result; never raises."""
    _load()
    executor = _DISPATCH.get(name)
    if executor is None:
        log.error("unknown_tool", name=name)
        return f"Unknown tool: {name}"
    try:
        return executor(args, ctx)
    except Exception as exc:
        log.error("tool_error", tool=name, error=str(exc), exc_info=True)
        return f"Tool error ({name}): {exc}"
