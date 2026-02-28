from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from .episodes import EpisodeStore
    from .sponge import BeliefMeta, SpongeState, StagedOpinionUpdate
    from .updater import compute_magnitude, extract_insight, validate_snapshot

__all__ = [
    "BeliefMeta",
    "EpisodeStore",
    "SpongeState",
    "StagedOpinionUpdate",
    "compute_magnitude",
    "extract_insight",
    "validate_snapshot",
]

_EXPORT_MODULES: Final[dict[str, str]] = {
    "EpisodeStore": ".episodes",
    "BeliefMeta": ".sponge",
    "SpongeState": ".sponge",
    "StagedOpinionUpdate": ".sponge",
    "compute_magnitude": ".updater",
    "extract_insight": ".updater",
    "validate_snapshot": ".updater",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)
