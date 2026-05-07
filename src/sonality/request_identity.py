"""Request-scoped identity cache (ContextVar) to avoid duplicate Neo4j reads per turn."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from .memory.graph import BeliefNode


@dataclass(frozen=True, slots=True)
class IdentityBundle:
    """Snapshot + beliefs loaded once per user-facing request."""

    snapshot_text: str
    beliefs_prompt_text: str
    all_beliefs: tuple[BeliefNode, ...]


_identity_ctx: ContextVar[IdentityBundle | None] = ContextVar(
    "sonality_request_identity", default=None
)


def get_request_identity() -> IdentityBundle | None:
    """Return cached identity for the current context, if any."""
    return _identity_ctx.get()


def set_request_identity(bundle: IdentityBundle) -> Token[IdentityBundle | None]:
    """Bind identity for the current context; returns token for ``reset_request_identity``."""
    return _identity_ctx.set(bundle)


def reset_request_identity(token: Token[IdentityBundle | None]) -> None:
    """Restore previous identity context."""
    _identity_ctx.reset(token)
