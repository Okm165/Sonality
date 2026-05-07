"""Fathom CLI — run research from the terminal with a live dashboard."""

from __future__ import annotations

import asyncio
import sys

from shared.logging import setup_logging

from .config import settings
from .observe import ResearchDashboard


async def _run_research(goal: str, seeds: list[str]) -> None:
    from . import browser, db, session

    setup_logging(settings.log_level)
    driver = await db.get_driver()
    await browser.launch()

    session_id = await db.create_session(driver, goal)
    dashboard = ResearchDashboard(goal, settings.max_pages)

    try:
        with dashboard:
            result = await session.run(
                driver,
                session_id,
                goal,
                seeds,
                dashboard=dashboard,
            )
        print("\n" + result.document)
        print(
            f"\n--- {len(result.sources)} sources visited, "
            f"{sum(s.facts_extracted for s in result.sources)} facts extracted ---"
        )
    finally:
        await browser.close()
        await db.close_driver()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: fathom <research goal> [--seed URL ...]")
        sys.exit(1)

    args = sys.argv[1:]
    seeds: list[str] = []
    goal_parts: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--seed" and i + 1 < len(args):
            seeds.append(args[i + 1])
            i += 2
        else:
            goal_parts.append(args[i])
            i += 1

    goal = " ".join(goal_parts)
    asyncio.run(_run_research(goal, seeds))


if __name__ == "__main__":
    main()
