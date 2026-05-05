"""Unified tool system — symmetric schema + executor for every agent tool.

Architecture: each tool module (memory, web, synthesize) exports two parallel
registries: DEFINITIONS (OpenAI-compatible function schemas for the LLM) and
EXECUTORS (name → callable). This module lazily merges them at first use.

ToolContext carries all runtime dependencies (graph, embedder, web client, etc.)
so tool executors never import from agent.py — the dependency arrow is one-way.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient

    from ..memory import DualEpisodeStore, Embedder, MemoryGraph
    from ..request_identity import IdentityBundle
    from ..web import WebSearchClient

log = logging.getLogger(__name__)

_T = TypeVar("_T")


class RunAsync(Protocol):
    def __call__(self, coro: Coroutine[object, object, _T], /) -> _T: ...


_ToolExecutor = Callable[["dict[str, object]", "ToolContext"], str]


@dataclass(frozen=True)
class ToolContext:
    """Runtime dependencies for tool execution, created per dispatch call.

    The agent populates this from its own state. Tools never import from agent.
    """

    run_async: RunAsync
    web_client: WebSearchClient | None
    graph: MemoryGraph
    dual_store: DualEpisodeStore
    qdrant: AsyncQdrantClient
    embedder: Embedder
    identity: IdentityBundle | None
    llm_messages: list[dict[str, object]]
    retrieve: Callable[[str], list[str]]

    def build_research_transcript(
        self, *, tool_tail: int = 6, assistant_tail: int = 3, char_limit: int = 1200
    ) -> str:
        """Collect recent tool results and assistant reasoning into a single transcript."""
        from ..schema import ChatRole

        parts: list[str] = []
        for m in self.llm_messages[-20:]:
            if m.get("role") == ChatRole.TOOL and len(parts) < tool_tail:
                parts.append(str(m.get("content", ""))[:char_limit])
        for m in self.llm_messages[-10:]:
            if m.get("role") == ChatRole.ASSISTANT and assistant_tail > 0:
                content = str(m.get("content", ""))
                if content:
                    parts.append(f"[Your reasoning]: {content[:600]}")
                    assistant_tail -= 1
        return "\n---\n".join(parts)


_DISPATCH: dict[str, _ToolExecutor] = {}
_DEFINITIONS: list[dict[str, object]] = []
_LOADED = False


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    from . import memory, synthesize, web

    for mod in (memory, web, synthesize):
        _DEFINITIONS.extend(mod.DEFINITIONS)
        _DISPATCH.update(mod.EXECUTORS)
    _LOADED = True


def get_definitions() -> list[dict[str, object]]:
    """All tool schemas for the LLM."""
    _load()
    return list(_DEFINITIONS)


def dispatch_tool(name: str, args: dict[str, object], ctx: ToolContext) -> str:
    """Execute a tool by name. Returns text result; never raises."""
    _load()
    executor = _DISPATCH.get(name)
    if executor is None:
        log.error("Unknown tool requested: %s", name)
        return f"Unknown tool: {name}"
    try:
        return executor(args, ctx)
    except Exception as exc:
        log.error("Tool %s failed: %s", name, exc, exc_info=True)
        return f"Tool error ({name}): {exc}"
