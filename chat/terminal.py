"""Async Rich terminal interface for Sonality chat."""

from __future__ import annotations

import asyncio
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import config
from .client import Belief, SonalityClient

log = logging.getLogger(__name__)

console = Console()

HELP_TEXT = """
[bold]Available Commands:[/bold]

  [cyan]/beliefs[/cyan]  - Show current beliefs
  [cyan]/health[/cyan]   - Check API health status
  [cyan]/clear[/cyan]    - Clear conversation history
  [cyan]/help[/cyan]     - Show this help message
  [cyan]/quit[/cyan]     - Exit the chat

[dim]Type any message to chat with Sonality.[/dim]
"""


def _display_beliefs(beliefs: list[Belief]) -> None:
    """Display beliefs in a formatted table."""
    if not beliefs:
        console.print("[yellow]No beliefs formed yet.[/yellow]")
        return

    table = Table(title="Current Beliefs", show_header=True, header_style="bold magenta")
    table.add_column("Topic", style="cyan", no_wrap=True)
    table.add_column("Position", justify="right", style="green")
    table.add_column("Confidence", justify="right", style="yellow")

    for belief in beliefs[:15]:
        table.add_row(belief.topic, f"{belief.position:+.2f}", f"{belief.confidence:.2f}")

    console.print(table)


async def _chat_loop(client: SonalityClient) -> None:
    """Main chat loop."""
    while True:
        try:
            user_input = await asyncio.to_thread(console.input, "[bold cyan]You:[/bold cyan] ")
            user_input = user_input.strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if cmd == "/help":
            console.print(HELP_TEXT)
            continue

        if cmd == "/clear":
            console.print("[green]✓ Conversation cleared[/green]")
            continue

        if cmd == "/beliefs":
            try:
                beliefs = await client.beliefs()
                _display_beliefs(beliefs)
            except Exception as e:
                log.error("Failed to fetch beliefs: %s", e, exc_info=True)
                console.print(f"[red]Error fetching beliefs: {e}[/red]")
            continue

        if cmd == "/health":
            try:
                health = await client.health()
                console.print(f"[green]Version:[/green] {health.version}")
                console.print(f"[green]Interactions:[/green] {health.interaction_count}")
                console.print(f"[green]Beliefs:[/green] {health.belief_count}")
                if health.staged_updates:
                    console.print(f"[yellow]Staged:[/yellow] {health.staged_updates}")
            except Exception as e:
                log.error("Failed to fetch health: %s", e, exc_info=True)
                console.print(f"[red]Error fetching health: {e}[/red]")
            continue

        log.debug("User input: %.100s", user_input)

        try:
            console.print("[bold green]Sonality:[/bold green] ", end="")
            full_response = ""
            async for chunk in client.chat_stream(user_input):
                console.print(chunk, end="")
                full_response += chunk
            console.print()
            log.info("Response: %.100s", full_response)
        except Exception as e:
            log.error("Chat error: %s", e, exc_info=True)
            console.print(f"[red]Error: {e}[/red]")

        console.print()


async def _main() -> None:
    """Async main entry point."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    console.print(
        Panel.fit(
            f"[bold blue]Sonality Chat[/bold blue]\n[dim]Connected to: {config.SONALITY_URL}[/dim]",
            border_style="blue",
        )
    )
    console.print("[dim]Type /help for available commands[/dim]")
    console.print()

    async with SonalityClient() as client:
        try:
            await client.health()
            console.print("[green]✓ Connected to Sonality[/green]")
        except Exception as e:
            log.error("Cannot connect to Sonality: %s", e, exc_info=True)
            console.print(f"[red]✗ Cannot connect to Sonality: {e}[/red]")
            console.print(
                f"[yellow]Make sure Sonality is running at {config.SONALITY_URL}[/yellow]"
            )
            sys.exit(1)

        console.print()
        await _chat_loop(client)


def main() -> None:
    """Entry point for terminal chat."""
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    main()
