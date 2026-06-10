"""Shared helpers for feed scripts (feed.py, x_feed.py).

Provides: queue_ingest (fire-and-forget POST to /ingest with backpressure),
print_error (HTTP error display), show_beliefs (Rich table of current beliefs),
and connect_or_exit (health-check guard).
"""

from __future__ import annotations

import time

import httpx
from rich.console import Console
from rich.markup import escape
from rich.table import Table

console = Console()


def connect_or_exit(client: httpx.Client, base: str) -> bool:
    """Check /health and print connection status. Returns False on failure."""
    try:
        client.get(f"{base}/health").raise_for_status()
    except Exception as exc:
        console.print(f"[red]\u2717 Sonality unreachable: {exc}[/red]")
        return False
    console.print("[green]\u2713 Connected[/green]\n")
    return True


def fetch_and_show_beliefs(client: httpx.Client, base: str) -> None:
    """Fetch beliefs from API and display them."""
    try:
        r = client.get(f"{base}/beliefs")
        r.raise_for_status()
        beliefs = r.json()
        show_beliefs(beliefs if isinstance(beliefs, list) else beliefs.get("beliefs", []))
    except Exception as exc:
        console.print(f"[yellow]Could not fetch beliefs: {exc}[/yellow]")


def queue_ingest(
    client: httpx.Client,
    base: str,
    text: str,
    topic: str,
    throttle: float,
) -> tuple[bool, int]:
    """POST text to /ingest, handle errors and back-off.

    Returns (success, queue_depth). Sleeps for throttle seconds on success,
    or extra time if queue_depth > 50.
    """
    try:
        r = client.post(f"{base}/ingest", json={"text": text, "topic_override": topic})
        r.raise_for_status()
        data = r.json()
        job_id = data.get("job_id", "")
        depth = data.get("queue_depth", 0)
        console.print(f"  [green]✓ queued[/green] job={job_id[:8]} depth={depth}")
        if depth > 50 and throttle > 0:
            extra = min(depth * 2, 120)
            console.print(f"  [yellow]queue depth {depth} — pausing {extra}s[/yellow]")
            time.sleep(extra)
        elif throttle > 0:
            time.sleep(throttle)
        return True, depth
    except httpx.HTTPStatusError as exc:
        print_error(exc)
        if throttle > 0:
            time.sleep(throttle)
        return False, 0
    except Exception as exc:
        console.print(f"  [red]✗ {exc}[/red]")
        if throttle > 0:
            time.sleep(throttle)
        return False, 0


def print_error(exc: httpx.HTTPStatusError) -> None:
    """Display HTTP error with response body."""
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
