"""Interactive Sonality REPL with client-side conversation history."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable

from . import __version__, config
from .agent import SonalityAgent
from .schema import ChatRole

log = logging.getLogger(__name__)

BANNER = """\
============================================================
  SONALITY v{version} (stateless, graph-backed)
============================================================
  Base URL: {base_url}
  Model: {model}
  ESS model: {ess_model}

  Commands:
    /snapshot  narrative personality snapshot
    /beliefs   current belief states from graph
    /models    current base-url/model configuration
    /clear     clear conversation history
    /quit      exit
============================================================"""


def _show_snapshot(agent: SonalityAgent) -> None:
    snapshot = agent.get_snapshot()
    print(f"  [v{snapshot.version}] {snapshot.text}")


def _show_beliefs(agent: SonalityAgent) -> None:
    beliefs = agent.get_all_beliefs()
    if not beliefs:
        print("  No beliefs formed yet.")
        return
    for b in beliefs:
        entry = f"  {b.topic:30s} {b.valence:+.3f}  (conf={b.confidence:.2f} ev={b.evidence_count})"
        if b.belief_text:
            entry += f"  {b.belief_text[:60]}"
        print(entry)


def _show_models(agent: SonalityAgent) -> None:
    print(f"  Base URL:   {config.BASE_URL}")
    print(f"  Model:      {agent.model}")
    print(f"  ESS model:  {agent.ess_model}")


CommandHandler = Callable[[SonalityAgent], None]

COMMAND_HANDLERS: dict[str, CommandHandler] = {
    "/snapshot": _show_snapshot,
    "/beliefs": _show_beliefs,
    "/models": _show_models,
}


def main() -> None:
    """Run the interactive Sonality REPL."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="sonality", description="Interactive Sonality REPL")
    parser.add_argument("--model", default=config.MODEL, help="Main response model ID.")
    parser.add_argument("--ess-model", default=config.ESS_MODEL, help="ESS model ID.")
    args = parser.parse_args()
    missing = config.missing_live_api_config()
    if missing:
        print(f"Error: set {', '.join(missing)} in .env or environment.")
        sys.exit(1)

    agent = SonalityAgent(model=args.model, ess_model=args.ess_model)
    print(BANNER.format(version=__version__, base_url=config.BASE_URL, model=agent.model, ess_model=agent.ess_model))

    conversation: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        command = user_input.lower()
        if command == "/quit":
            print("Goodbye.")
            break
        if command == "/clear":
            conversation.clear()
            print("  Conversation cleared.")
            continue
        if command.startswith("/"):
            handler = COMMAND_HANDLERS.get(command)
            if handler is None:
                print(f"  Unknown command: {command}")
            else:
                handler(agent)
            continue

        conversation.append({"role": ChatRole.USER, "content": user_input})

        print()
        try:
            response = agent.respond(list(conversation))
        except Exception as exc:
            log.exception("REPL respond failed")
            print(f"\033[31mError: {exc}\033[0m")
            continue

        conversation.append({"role": ChatRole.ASSISTANT, "content": response})
        print(f"Sonality: {response}")

        ess = agent.last_ess
        parts = [f"ESS {ess.score:.2f}"]
        if ess.topics:
            parts.append(", ".join(ess.topics))
        print(f"  [{' | '.join(parts)}]")


if __name__ == "__main__":
    main()
