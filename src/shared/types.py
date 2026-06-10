"""Shared type definitions used across multiple packages."""

from __future__ import annotations

import uuid
from enum import StrEnum


def new_id() -> str:
    """Generate a random UUID4 string. Standard identifier across all modules."""
    return str(uuid.uuid4())


def deterministic_id(seed: str) -> str:
    """Generate a deterministic UUID5 from a seed string.

    Same seed always produces the same ID — used for derivative chunks,
    knowledge propositions, and semantic features where re-processing
    the same input should yield identical UIDs.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


class ChatRole(StrEnum):
    """Message roles in chat completions."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
