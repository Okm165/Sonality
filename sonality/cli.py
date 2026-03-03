from __future__ import annotations

import argparse
import difflib
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from . import config
from .agent import SonalityAgent
from .memory import SpongeState


def _print_status(agent: SonalityAgent) -> None:
    """Print a one-line post-turn summary of ESS and sponge state."""
    ess = agent.last_ess
    score_str = f"{ess.score:.2f}" if ess else "n/a"
    topics = ", ".join(ess.topics) if ess and ess.topics else ""
    updated = ess and ess.score > config.ESS_THRESHOLD
    parts = [
        f"ESS {score_str}",
        f"v{agent.sponge.version}",
        f"#{agent.sponge.interaction_count}",
        f"staged={len(agent.sponge.staged_opinion_updates)}",
    ]
    if topics:
        parts.append(topics)
    if updated:
        parts.append("\033[33mSPONGE UPDATED\033[0m")
    print(f"  [{' | '.join(parts)}]")


def _show_health(agent: SonalityAgent) -> None:
    """Print human-readable personality health diagnostics."""
    s = agent.sponge
    metas = list(s.belief_meta.values())
    ic = s.interaction_count or 1
    strong = sum(1 for m in metas if m.confidence > 0.5)
    stale = sum(1 for m in metas if ic - m.last_reinforced > 30)

    if ic < 20:
        maturity = "Layer 1 (linguistic mimicry)"
    elif ic < 50 or len(s.opinion_vectors) < 10:
        maturity = "Layer 2 (structured accumulation)"
    else:
        maturity = "Layer 3 (autonomous expansion)"

    entrenched = s.detect_entrenched_beliefs()
    contradictions = agent._collect_unresolved_contradictions()
    contradiction_line = ", ".join(contradictions[:3]) if contradictions else "none"
    recent = s.recent_shifts[-1] if s.recent_shifts else None
    last_line = f"#{recent.interaction} — {recent.description[:50]}" if recent else "none"

    print(f"  Maturity:    {maturity} ({ic} interactions, {len(s.opinion_vectors)} beliefs)")
    print(f"  Beliefs:     {len(s.opinion_vectors)} total, {strong} strong, {stale} stale")
    print(f"  Disagree:    {s.behavioral_signature.disagreement_rate:.0%}")
    print(f"  Insights:    {len(s.pending_insights)} pending")
    print(f"  Staged:      {len(s.staged_opinion_updates)} pending commits")
    print(f"  Entrenched:  {', '.join(entrenched) if entrenched else 'none'}")
    print(f"  Contradict:  {contradiction_line}")
    print(f"  Snapshot:    {len(s.snapshot)} chars, v{s.version}")
    print(f"  Last shift:  {last_line}")


def _show_diff(agent: SonalityAgent) -> None:
    """Print unified diff between current and previous personality snapshot."""
    old = agent.previous_snapshot or "(no previous snapshot)"
    new = agent.sponge.snapshot
    if old == new:
        print("  No changes since last interaction.")
        return
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"sponge v{max(agent.sponge.version - 1, 0)}",
        tofile=f"sponge v{agent.sponge.version}",
        lineterm="",
    )
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            print(f"  \033[32m{line}\033[0m")
        elif line.startswith("-") and not line.startswith("---"):
            print(f"  \033[31m{line}\033[0m")
        else:
            print(f"  {line}")


BANNER = """\
============================================================
  SONALITY v0.1
============================================================
  Sponge v{version} | {interactions} prior interactions
  Base URL: {base_url}
  Model: {model}
  ESS model: {ess_model}

  Commands:
    /sponge    full personality state (JSON)
    /snapshot  narrative snapshot text
    /beliefs   opinion vectors with confidence
    /insights  pending personality insights
    /staged    staged opinion updates (cooling period)
    /topics    topic engagement counts
    /shifts    recent personality shifts
    /health    personality health metrics
    /models    current base-url/model configuration
    /diff      diff of last snapshot change
    /reset     reset to seed personality
    /quit      exit
============================================================"""


def _show_sponge(agent: SonalityAgent) -> None:
    """Print full sponge state as indented JSON."""
    print(json.dumps(agent.sponge.model_dump(), indent=2))


def _show_snapshot(agent: SonalityAgent) -> None:
    """Print the current narrative snapshot text."""
    print(f"\n  {agent.sponge.snapshot}")


def _show_beliefs(agent: SonalityAgent) -> None:
    """Print tracked belief vectors with confidence metadata."""
    if not agent.sponge.opinion_vectors:
        print("  No beliefs formed yet.")
        return
    for topic, pos in sorted(agent.sponge.opinion_vectors.items(), key=lambda x: -abs(x[1])):
        meta = agent.sponge.belief_meta.get(topic)
        conf = f"conf={meta.confidence:.2f} ev={meta.evidence_count}" if meta else "no meta"
        print(f"  {topic:30s} {pos:+.3f}  ({conf})")


def _show_insights(agent: SonalityAgent) -> None:
    """Print pending personality insights queued for reflection."""
    if not agent.sponge.pending_insights:
        print("  No pending insights (cleared at last reflection).")
        return
    for index, insight in enumerate(agent.sponge.pending_insights, 1):
        print(f"  {index}. {insight}")


def _show_staged_updates(agent: SonalityAgent) -> None:
    """Print staged opinion updates awaiting cooldown commit."""
    if not agent.sponge.staged_opinion_updates:
        print("  No staged opinion updates.")
        return
    for update in agent.sponge.staged_opinion_updates:
        print(
            "  "
            f"{update.topic:30s} {update.signed_magnitude:+.4f} "
            f"(due #{update.due_interaction}, staged #{update.staged_at})"
        )


def _show_topics(agent: SonalityAgent) -> None:
    """Print topic-engagement counters tracked in the behavioral signature."""
    engagement = agent.sponge.behavioral_signature.topic_engagement
    if not engagement:
        print("  No topics tracked yet.")
        return
    for topic, count in sorted(engagement.items(), key=lambda x: -x[1]):
        print(f"  {topic:30s} {count}")


def _show_shifts(agent: SonalityAgent) -> None:
    """Print recent recorded personality shifts."""
    if not agent.sponge.recent_shifts:
        print("  No shifts recorded yet.")
        return
    for shift in agent.sponge.recent_shifts:
        print(f"  #{shift.interaction} ({shift.magnitude:.3f}): {shift.description}")


def _reset(agent: SonalityAgent) -> None:
    """Reset sponge state to seed personality and clear in-memory context."""
    agent.sponge = SpongeState()
    agent.sponge.save(config.SPONGE_FILE, config.SPONGE_HISTORY_DIR)
    agent.conversation.clear()
    agent.previous_snapshot = None
    print("  Sponge reset to seed state.")


def _show_models(agent: SonalityAgent) -> None:
    """Print active base URL and model selections for this runtime."""
    print(f"  API variant:{config.API_VARIANT}")
    print(f"  Base URL:   {config.BASE_URL}")
    print(f"  Model:      {agent.model}")
    print(f"  ESS model:  {agent.ess_model}")


CommandResult = Literal["handled", "chat", "quit"]
CommandHandler = Callable[[SonalityAgent], None]

COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "/sponge": _show_sponge,
    "/snapshot": _show_snapshot,
    "/beliefs": _show_beliefs,
    "/insights": _show_insights,
    "/staged": _show_staged_updates,
    "/topics": _show_topics,
    "/shifts": _show_shifts,
    "/health": _show_health,
    "/models": _show_models,
    "/diff": _show_diff,
    "/reset": _reset,
}


@dataclass(frozen=True, slots=True)
class CliArgs:
    """CLI runtime overrides for model selection."""

    model: str
    ess_model: str


def _parse_args() -> CliArgs:
    """Parse CLI arguments for runtime model selection overrides."""
    parser = argparse.ArgumentParser(
        prog="sonality",
        description="Interactive Sonality REPL",
    )
    parser.add_argument(
        "--model",
        default=config.MODEL,
        help="Main response-generation model ID.",
    )
    parser.add_argument(
        "--ess-model",
        default=config.ESS_MODEL,
        help="ESS/insight/reflection model ID.",
    )
    parsed = parser.parse_args()
    return CliArgs(
        model=parsed.model,
        ess_model=parsed.ess_model,
    )


def _configure_logging() -> None:
    """Initialize CLI logging from environment-backed configuration."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


def _require_api_config() -> None:
    """Exit process when required API configuration is not configured."""
    missing = config.missing_live_api_config()
    if not missing:
        return
    print(f"Error: set {', '.join(missing)} in .env or environment.")
    print("  cp .env.example .env && $EDITOR .env")
    sys.exit(1)


def _print_banner(agent: SonalityAgent) -> None:
    """Print startup banner with persisted sponge metadata."""
    print(
        BANNER.format(
            version=agent.sponge.version,
            interactions=agent.sponge.interaction_count,
            base_url=config.BASE_URL,
            model=agent.model,
            ess_model=agent.ess_model,
        )
    )


def _handle_command(agent: SonalityAgent, command: str) -> CommandResult:
    """Handle slash-command input and return next REPL action."""
    if command == "/quit":
        print("Goodbye.")
        return "quit"
    if not command.startswith("/"):
        return "chat"
    handler = COMMAND_HANDLERS.get(command)
    if handler is None:
        print(f"  Unknown command: {command}")
        return "handled"
    handler(agent)
    return "handled"


def main() -> None:
    """Run the interactive Sonality REPL."""
    _configure_logging()
    cli_args = _parse_args()
    _require_api_config()

    agent = SonalityAgent(
        model=cli_args.model,
        ess_model=cli_args.ess_model,
    )
    _print_banner(agent)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        command_result = _handle_command(agent, user_input.lower())
        if command_result == "quit":
            break
        if command_result == "handled":
            continue

        print()
        try:
            response = agent.respond(user_input)
        except Exception as exc:
            print(f"\033[31mError: {exc}\033[0m")
            continue
        print(f"Sonality: {response}")
        _print_status(agent)


if __name__ == "__main__":
    main()
