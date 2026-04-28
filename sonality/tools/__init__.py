"""Unified tool system — symmetric schema + executor for every agent tool.

Each tool module exports DEFINITIONS (schemas) and EXECUTORS (name -> function).
ToolContext provides all runtime dependencies without coupling to SonalityAgent.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeVar

from ..schema import ToolName

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient

    from ..memory import DualEpisodeStore, Embedder, MemoryGraph
    from ..request_identity import IdentityBundle
    from ..web import WebSearchClient

log = logging.getLogger(__name__)

_T = TypeVar("_T")

WEB_TOOL_NAMES: Final = frozenset({ToolName.WEB_SEARCH, ToolName.WEB_EXTRACT})


_ToolExecutor = Callable[["dict[str, object]", "ToolContext"], str]


@dataclass(frozen=True)
class ToolContext:
    """Runtime dependencies for tool execution, created per dispatch call.

    The agent populates this from its own state. Tools never import from agent.
    """

    run_async: Callable[[Coroutine[object, object, _T]], _T]
    web_client: WebSearchClient | None
    graph: MemoryGraph
    dual_store: DualEpisodeStore
    qdrant: AsyncQdrantClient
    embedder: Embedder
    identity: IdentityBundle | None
    llm_messages: list[dict[str, object]]
    retrieve: Callable[[str], list[str]]


_DISPATCH: dict[str, _ToolExecutor] = {}
_DEFINITIONS: list[dict[str, object]] = []
_LOADED = False


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    from . import assess, consolidate, memory, reflect, web

    for mod in (memory, web, assess, consolidate, reflect):
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
