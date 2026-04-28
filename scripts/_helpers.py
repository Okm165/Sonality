"""Shared display helpers for feed scripts."""

from __future__ import annotations

import httpx
from rich.console import Console
from rich.markup import escape
from rich.table import Table

console = Console()


def print_result(res: dict[str, object]) -> None:
    """Display ingest result with full ESS feedback."""
    score = float(res.get("score", 0) or 0)
    rtype = res.get("reasoning_type", "?")
    urgency = res.get("urgency", "?")
    updated = res.get("belief_update_recommended", False)
    raw_topics = res.get("topics")
    topics = ", ".join(raw_topics[:3]) if isinstance(raw_topics, list) else "-"
    summary = res.get("summary") or ""

    color = "green" if updated else "yellow" if score else "dim"
    marker = "▲" if updated else "·"
    console.print(
        f"  [{color}]{marker}[/{color}] ESS={score:.2f} {rtype} "
        f"[dim]urgency={urgency}[/dim] | {topics}"
    )
    if summary and isinstance(summary, str):
        console.print(f"    [dim italic]{escape(summary[:200])}[/dim italic]")


def print_error(exc: httpx.HTTPStatusError) -> None:
    """Display ingest error with response body."""
    if exc.response is not None:
        body = exc.response.text[:300]
        console.print(f"  [red]✗ HTTP {exc.response.status_code}[/red]")
        if body:
            console.print(f"    [dim red]{escape(body)}[/dim red]")
    else:
        console.print("  [red]✗ HTTP error (no response)[/red]")


def show_beliefs(beliefs: list[dict[str, object]]) -> None:
    """Display a belief table from the /beliefs API response."""
    if not beliefs:
        console.print("[yellow]No beliefs yet.[/yellow]")
        return
    table = Table(title="Beliefs", header_style="bold magenta")
    table.add_column("Topic", style="cyan")
    table.add_column("Valence", justify="right", style="green")
    table.add_column("Confidence", justify="right", style="yellow")
    for b in beliefs[:10]:
        val = b.get("valence", 0)
        conf = b.get("confidence", 0)
        table.add_row(
            str(b.get("topic", "")),
            f"{val:+.2f}" if isinstance(val, (int, float)) else str(val),
            f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf),
        )
    console.print(table)
