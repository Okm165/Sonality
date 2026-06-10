"""Entry point for chat module: python -m chat [terminal|telegram]."""

from __future__ import annotations

import sys


def main() -> None:
    """Dispatch to terminal or telegram based on command line argument."""
    if len(sys.argv) < 2:
        print("Usage: python -m chat [terminal|telegram]")
        print("  terminal  - Start terminal chat interface")
        print("  telegram  - Start Telegram bot")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "terminal":
        from .terminal import main as terminal_main

        terminal_main()
    elif mode == "telegram":
        from .telegram import main as telegram_main

        telegram_main()
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'terminal' or 'telegram'")
        sys.exit(1)


if __name__ == "__main__":
    main()
