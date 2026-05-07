"""ResearchDashboard (rich) for live progress display."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import ChecklistItem, SessionMemory


@dataclass
class DashboardState:
    """Typed snapshot of research session progress."""

    round: int = 0
    pages_fetched: int = 0
    total_facts: int = 0
    checklist: list[ChecklistItem] = field(default_factory=list)
    memory: SessionMemory | None = None
    concentration: float = 0.0


class ResearchDashboard:
    """Live-updating console display for research progress."""

    def __init__(self, goal: str, max_pages: int) -> None:
        self.goal = goal
        self.max_pages = max_pages
        self._live: Live | None = None
        self._last_action: str = ""
        self._last_state = DashboardState()

    def start(self) -> None:
        self._live = Live(self._build_display(self._last_state), refresh_per_second=4)
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def update(self, state: DashboardState) -> None:
        self._last_state = state
        if self._live is not None:
            self._live.update(self._build_display(state))

    def set_action(self, action: str) -> None:
        self._last_action = action
        if self._live is not None:
            self._live.update(self._build_display(self._last_state))

    def _build_display(self, state: DashboardState) -> Panel:
        layout = Layout()

        round_num = state.round
        pages = state.pages_fetched
        total_facts = state.total_facts
        checklist = state.checklist
        memory = state.memory
        concentration = state.concentration

        answered = sum(1 for item in checklist if item.answered)
        total_items = len(checklist)
        pct = (pages / self.max_pages * 100) if self.max_pages else 0

        # Header
        header = Text()
        header.append(f"  Round {round_num}    ", style="bold")
        header.append(f"Pages: {pages}/{self.max_pages}    ", style="cyan")
        header.append(f"Facts: {total_facts}    ", style="green")
        header.append(f"Stalls: {memory.stall_rounds if memory else 0}", style="yellow")
        header.append(f"\n  {'━' * 40} {pct:.0f}% complete")

        # Checklist
        cl_table = Table(show_header=False, box=None, padding=(0, 1))
        cl_table.add_column("status", width=3)
        cl_table.add_column("question")
        for item in checklist:
            mark = "✓" if item.answered else "◌"
            style = "green" if item.answered else "dim"
            cl_table.add_row(mark, item.question, style=style)

        # Health
        health = Text()
        if memory:
            fpr = memory.facts_per_round[-7:] if memory.facts_per_round else []
            health.append(f"  facts/round: {fpr}\n")
            prod = len(memory.productive_urls)
            total_visited = prod + len(memory.unproductive_urls)
            rate = (prod / total_visited * 100) if total_visited else 0
            health.append(f"  productive: {prod}/{total_visited} ({rate:.0f}%)\n")
        if concentration:
            health.append(f"  concentration: {concentration}")

        # Action
        action_text = Text(f"  {self._last_action}", style="bold cyan")

        layout.split_column(
            Layout(Panel(header, title=f'Research: "{self.goal[:60]}"'), size=5),
            Layout(
                Panel(cl_table, title=f"CHECKLIST [{answered}/{total_items} answered]"),
                size=max(total_items + 3, 5),
            ),
            Layout(Panel(health, title="HEALTH"), size=7),
            Layout(Panel(action_text, title="LAST ACTION"), size=3),
        )

        return Panel(layout, border_style="blue")

    def __enter__(self) -> ResearchDashboard:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
