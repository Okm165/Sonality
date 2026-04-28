"""Async Rich terminal interface for Sonality chat with live progress display."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time as _time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import config
from .client import Belief, ProgressEvent, SonalityClient, TOOL_LABELS, pipeline_summary

log = logging.getLogger(__name__)

console = Console()

HELP_TEXT = """[bold]Commands[/bold]

  /beliefs   Current beliefs with confidence
  /snapshot  Personality overview
  /health    Check API health
  /clear     Clear conversation history
  /help      Show this message
  /quit      Exit

[dim]Type anything else to chat.[/dim]
"""


def _display_beliefs(beliefs: list[Belief]) -> None:
    if not beliefs:
        console.print("[yellow]No beliefs formed yet.[/yellow]")
        return

    sorted_beliefs = sorted(beliefs, key=lambda b: b.confidence, reverse=True)

    table = Table(
        title=f"Beliefs ({len(sorted_beliefs)})",
        show_header=True,
        header_style="bold magenta",
        expand=True,
        padding=(0, 1),
        border_style="dim",
    )
    table.add_column("Topic", style="bold cyan", no_wrap=True, max_width=24)
    table.add_column("Val", justify="right", width=6)
    table.add_column("Conf", width=8)
    table.add_column("Belief", ratio=1, style="white")

    for b in sorted_beliefs[:20]:
        val_style = "green" if b.valence > 0.1 else "red" if b.valence < -0.1 else "dim"
        table.add_row(
            b.topic,
            Text(f"{b.valence:+.1f}", style=val_style),
            Text(f"{b.confidence:.0%}", style="yellow"),
            Text(b.belief_text[:90] + ("..." if len(b.belief_text or "") > 90 else ""))
            if b.belief_text
            else Text("-", style="dim"),
        )
    console.print(table)


def _progress_action(event: ProgressEvent) -> str:
    """Convert a progress event into a short action string."""
    if event.type == "thinking":
        return "Thinking..."
    if event.type == "context_build":
        return event.detail if event.detail else "Loading context..."
    if event.type == "summarizing":
        return "Compressing context..."
    if event.type == "tool_call":
        label = TOOL_LABELS.get(event.tool_name, event.tool_name.replace("_", " "))
        query = ""
        if event.tool_args:
            try:
                parsed = json.loads(event.tool_args)
                query = parsed.get("query", parsed.get("url", parsed.get("text", "")))[:60]
            except json.JSONDecodeError:
                query = event.tool_args[:60]
        return f"{label}: {query}" if query else label
    if event.type == "tool_result":
        label = TOOL_LABELS.get(event.tool_name, event.tool_name.replace("_", " "))
        detail = ""
        if event.sources_count:
            detail = f" ({event.sources_count} sources)"
        elif event.tool_result_summary:
            summary = event.tool_result_summary[:60]
            if len(event.tool_result_summary) > 60:
                summary += "..."
            detail = f" - {summary}"
        return f"{label}{detail}"
    return ""


def _build_status(completed: list[str], current: str, elapsed: float) -> Text:
    """Render completed steps, current action, and elapsed time."""
    t = Text()
    for line in completed[-8:]:
        t.append(f"  [done] {line}\n", style="dim green")
    if current:
        t.append(f"  >> {current}", style="bold cyan")
    if elapsed > 0.5:
        t.append(f"  [{elapsed:.0f}s]", style="dim")
    return t


def _format_done_metadata(detail: str, tool_names: list[str]) -> Text | None:
    """Parse DONE event detail into a styled metadata footer."""
    try:
        d = json.loads(detail)
    except (json.JSONDecodeError, TypeError):
        return None

    t = Text()
    pipeline = pipeline_summary(tool_names)
    if pipeline:
        t.append(pipeline, style="dim")
    if el := d.get("elapsed"):
        if t.plain:
            t.append(" | ", style="dim")
        t.append(f"{el}s", style="dim yellow")
    if ess := d.get("ess_score"):
        if t.plain:
            t.append(" | ", style="dim")
        ess_f = float(ess)
        ess_style = "bold green" if ess_f >= 0.5 else "yellow" if ess_f >= 0.2 else "dim"
        t.append(f"ESS {ess}", style=ess_style)
    if rtype := d.get("reasoning_type"):
        t.append(f" ({rtype})", style="dim")
    if topics := d.get("topics"):
        if t.plain:
            t.append(" | ", style="dim")
        t.append(" ".join(f"#{tp}" for tp in topics[:4]), style="dim cyan")
    return t if t.plain else None


async def _chat_loop(client: SonalityClient) -> None:
    while True:
        try:
            user_input = await asyncio.to_thread(console.input, "\n[bold cyan]You:[/bold cyan] ")
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
            client.clear_history()
            console.print("[green]Conversation cleared.[/green]")
            continue

        if cmd == "/beliefs":
            try:
                beliefs = await client.beliefs()
                _display_beliefs(beliefs)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            continue

        if cmd == "/snapshot":
            try:
                h = await client.health()
                uptime_min = int(h.uptime_seconds // 60)
                beliefs = await client.beliefs()
                snap = Text()
                snap.append(f"v{h.snapshot_version}", style="bold")
                snap.append(f" | {h.belief_count} beliefs", style="white")
                snap.append(f" | up {uptime_min}m", style="dim")
                if h.version:
                    snap.append(f" | {h.version}", style="dim")
                if beliefs:
                    top = ", ".join(
                        b.topic for b in sorted(
                            beliefs, key=lambda b: b.confidence, reverse=True
                        )[:5]
                    )
                    snap.append(f"\nTop: {top}", style="cyan")
                console.print(Panel(snap, title="Personality Snapshot", border_style="magenta"))
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            continue

        if cmd == "/health":
            try:
                h = await client.health()
                console.print(f"[green]Healthy[/green] v{h.snapshot_version} | {h.belief_count} beliefs")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            continue

        t0 = _time.perf_counter()
        log.info("User message: %d chars", len(user_input))
        completed_steps: list[str] = []
        tool_names_used: list[str] = []
        current_action = ""
        response_chars = 0
        done_meta: Text | None = None

        try:
            with Live(
                _build_status(completed_steps, current_action, 0),
                console=console,
                transient=True,
                refresh_per_second=4,
            ) as live:
                response_started = False
                async for item in client.chat_stream(user_input):
                    elapsed = _time.perf_counter() - t0

                    if isinstance(item, ProgressEvent):
                        log.debug("Progress: type=%s detail=%s", item.type, item.detail)

                        if item.type == "done":
                            done_meta = _format_done_metadata(item.detail, tool_names_used)
                            continue

                        if item.type == "tool_call" and item.tool_name:
                            tool_names_used.append(item.tool_name)

                        action = _progress_action(item)
                        if item.type == "tool_result":
                            if action:
                                completed_steps.append(action)
                            elif current_action:
                                completed_steps.append(current_action)
                            current_action = ""
                        elif action:
                            if current_action and item.type != "thinking":
                                completed_steps.append(current_action)
                            current_action = action
                        live.update(_build_status(completed_steps, current_action, elapsed))
                    elif isinstance(item, str):
                        if not response_started:
                            live.stop()
                            console.print()
                            console.print("[bold green]Sonality:[/bold green] ", end="")
                            response_started = True
                        response_chars += len(item)
                        console.print(item, end="", highlight=False)

            if response_started:
                console.print()

            if done_meta:
                footer = Text("  --- ", style="dim")
                footer.append_text(done_meta)
                console.print(footer)

            log.info(
                "Response complete: %d chars, %d tools, elapsed=%.1fs",
                response_chars,
                len(tool_names_used),
                _time.perf_counter() - t0,
            )
        except Exception as e:
            log.error("Chat error: %s", e, exc_info=True)
            console.print(f"[red]Error: {e}[/red]")


async def _main() -> None:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        log.debug("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    async with SonalityClient() as client:
        try:
            h = await client.health()
            beliefs = await client.beliefs()
            banner = Text()
            banner.append("Sonality", style="bold blue")
            banner.append(" - a mind that grows\n", style="dim")
            banner.append(f"v{h.snapshot_version}", style="bold")
            banner.append(f" | {h.belief_count} beliefs", style="white")
            if h.version:
                banner.append(f" | {h.version}", style="dim")
            if beliefs:
                top = ", ".join(
                    b.topic for b in sorted(
                        beliefs, key=lambda b: b.confidence, reverse=True
                    )[:4]
                )
                banner.append(f"\n{top}", style="cyan")
            console.print(Panel(banner, border_style="blue", padding=(0, 2)))
        except Exception as e:
            console.print(
                Panel.fit(
                    "[bold blue]Sonality[/bold blue]\n"
                    f"[red]Cannot connect: {e}[/red]\n"
                    f"[yellow]Ensure Sonality is running at {config.SONALITY_URL}[/yellow]",
                    border_style="red",
                    padding=(0, 2),
                )
            )
            sys.exit(1)

        console.print("[dim]Type /help for commands[/dim]")
        await _chat_loop(client)
    console.print("[dim]Goodbye![/dim]")


def main() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    main()
