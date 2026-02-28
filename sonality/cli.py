from __future__ import annotations

import difflib
import json
import logging
import sys

from . import config
from .agent import SonalityAgent
from .memory import SpongeState


def _print_status(agent: SonalityAgent) -> None:
    ess = agent.last_ess
    score_str = f"{ess.score:.2f}" if ess else "n/a"
    topics = ", ".join(ess.topics) if ess and ess.topics else ""
    updated = ess and ess.score > config.ESS_THRESHOLD
    parts = [
        f"ESS {score_str}",
        f"v{agent.sponge.version}",
        f"#{agent.sponge.interaction_count}",
    ]
    if topics:
        parts.append(topics)
    if updated:
        parts.append("\033[33mSPONGE UPDATED\033[0m")
    print(f"  [{' | '.join(parts)}]")


def _show_diff(agent: SonalityAgent) -> None:
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

  Commands:
    /sponge    full personality state (JSON)
    /snapshot  narrative snapshot text
    /beliefs   opinion vectors with confidence
    /insights  pending personality insights
    /topics    topic engagement counts
    /shifts    recent personality shifts
    /diff      diff of last snapshot change
    /reset     reset to seed personality
    /quit      exit
============================================================"""


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not config.ANTHROPIC_API_KEY:
        print("Error: set ANTHROPIC_API_KEY in .env or environment.")
        print("  cp .env.example .env && $EDITOR .env")
        sys.exit(1)

    agent = SonalityAgent()
    print(
        BANNER.format(
            version=agent.sponge.version,
            interactions=agent.sponge.interaction_count,
        )
    )

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "/quit":
            print("Goodbye.")
            break

        if cmd == "/sponge":
            print(json.dumps(agent.sponge.model_dump(), indent=2))
            continue

        if cmd == "/snapshot":
            print(f"\n  {agent.sponge.snapshot}")
            continue

        if cmd == "/beliefs":
            if not agent.sponge.opinion_vectors:
                print("  No beliefs formed yet.")
            else:
                for topic, pos in sorted(
                    agent.sponge.opinion_vectors.items(), key=lambda x: -abs(x[1])
                ):
                    meta = agent.sponge.belief_meta.get(topic)
                    conf = (
                        f"conf={meta.confidence:.2f} ev={meta.evidence_count}"
                        if meta
                        else "no meta"
                    )
                    print(f"  {topic:30s} {pos:+.3f}  ({conf})")
            continue

        if cmd == "/insights":
            if not agent.sponge.pending_insights:
                print("  No pending insights (cleared at last reflection).")
            else:
                for i, insight in enumerate(agent.sponge.pending_insights, 1):
                    print(f"  {i}. {insight}")
            continue

        if cmd == "/topics":
            eng = agent.sponge.behavioral_signature.topic_engagement
            if not eng:
                print("  No topics tracked yet.")
            else:
                for topic, count in sorted(eng.items(), key=lambda x: -x[1]):
                    print(f"  {topic:30s} {count}")
            continue

        if cmd == "/shifts":
            if not agent.sponge.recent_shifts:
                print("  No shifts recorded yet.")
            else:
                for s in agent.sponge.recent_shifts:
                    print(f"  #{s.interaction} ({s.magnitude:.3f}): {s.description}")
            continue

        if cmd == "/diff":
            _show_diff(agent)
            continue

        if cmd == "/reset":
            agent.sponge = SpongeState()
            agent.sponge.save(config.SPONGE_FILE, config.SPONGE_HISTORY_DIR)
            agent.conversation.clear()
            agent.previous_snapshot = None
            print("  Sponge reset to seed state.")
            continue

        if cmd.startswith("/"):
            print(f"  Unknown command: {cmd}")
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
