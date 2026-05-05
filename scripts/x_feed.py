#!/usr/bin/env python3
"""Feed X posts to Sonality — fetch one, ingest one, repeat.

Uses fire-and-forget /ingest (202 queue). Same pattern as feed.py.
Pay-per-use: $0.005/post + $0.01/user. Set a cap at developer.x.com.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import httpx
from _helpers import connect_or_exit, fetch_and_show_beliefs, queue_ingest
from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

X_API: Final = "https://api.x.com/2"
TWEET_FIELDS: Final = (
    "created_at,public_metrics,author_id,entities,context_annotations,note_tweet,conversation_id"
)
USER_FIELDS: Final = "username,name,public_metrics,verified"
MIN_TEXT_LEN: Final = 80
MIN_LIKES: Final = 5

QUERIES: Final[dict[str, tuple[str, str]]] = {
    "geopolitics": (
        '("foreign policy" OR sanctions OR diplomacy OR "security council" OR NATO)'
        " is:verified -is:retweet lang:en",
        "geopolitics",
    ),
    "conflict": (
        '("military" OR "troops" OR "missile" OR "ceasefire" OR "airstrike")'
        " is:verified has:links -is:retweet lang:en",
        "military_conflict",
    ),
    "ai_tech": (
        '("artificial intelligence" OR "large language model" OR GPT OR "machine learning"'
        ' OR "AI safety" OR "foundation model") has:links -is:retweet lang:en',
        "artificial_intelligence",
    ),
    "crypto": (
        "(bitcoin OR ethereum OR crypto OR DeFi OR blockchain OR $BTC OR $ETH)"
        " has:links -is:retweet lang:en -giveaway -airdrop -scam",
        "cryptocurrency",
    ),
    "markets": (
        '("stock market" OR "S&P 500" OR "Wall Street" OR earnings'
        ' OR "market crash" OR "bull market" OR "bear market")'
        " is:verified -is:retweet lang:en",
        "financial_markets",
    ),
    "economics": (
        '("interest rate" OR inflation OR GDP OR "central bank"'
        ' OR "Federal Reserve" OR tariff OR "trade deficit")'
        " is:verified -is:retweet lang:en",
        "economics",
    ),
    "science": (
        '("peer reviewed" OR "study finds" OR "researchers"'
        ' OR "published in" OR "clinical trial") has:links -is:retweet lang:en',
        "",
    ),
    "politics": (
        '(legislation OR congress OR parliament OR "executive order"'
        ' OR "policy change" OR election) is:verified has:links -is:retweet lang:en',
        "politics",
    ),
}

console = Console()


@dataclass(frozen=True, slots=True)
class Config:
    """Runtime configuration from environment variables (X_FEED_* prefix)."""

    max_results: int
    max_pages: int
    sort_order: str
    throttle: float


# ---------------------------------------------------------------------------
# X API
# ---------------------------------------------------------------------------


def _get(client: httpx.Client, path: str, params: dict[str, str | int]) -> dict[str, object]:
    """GET with 429 back-off."""
    r = client.get(f"{X_API}{path}", params=params)
    if r.status_code == 429:
        reset = int(r.headers.get("x-rate-limit-reset", "0"))
        wait = max(reset - int(time.time()), 1) if reset else 60
        console.print(f"[yellow]Rate limited — {wait}s[/yellow]")
        time.sleep(wait)
        r = client.get(f"{X_API}{path}", params=params)
    r.raise_for_status()
    return r.json()  # type: ignore[no-any-return]


def _search(
    client: httpx.Client, query: str, cfg: Config
) -> Iterator[tuple[dict[str, object], dict[str, object] | None]]:
    """Yield (post, author_or_none) one at a time."""
    params: dict[str, str | int] = {
        "query": query,
        "max_results": cfg.max_results,
        "sort_order": cfg.sort_order,
        "tweet.fields": TWEET_FIELDS,
        "expansions": "author_id",
        "user.fields": USER_FIELDS,
    }
    for _ in range(cfg.max_pages):
        try:
            resp = _get(client, "/tweets/search/recent", params)
        except httpx.HTTPStatusError as exc:
            console.print(f"  [red]API {exc.response.status_code}[/red]")
            return

        data = resp.get("data")
        if not isinstance(data, list) or not data:
            return

        users: dict[str, dict[str, object]] = {}
        includes = resp.get("includes")
        if isinstance(includes, dict):
            ul = includes.get("users")
            if isinstance(ul, list):
                users = {u["id"]: u for u in ul if isinstance(u, dict) and "id" in u}

        for post in data:
            if not isinstance(post, dict):
                continue
            aid = post.get("author_id")
            yield post, users.get(aid) if isinstance(aid, str) else None

        meta = resp.get("meta")
        nt = meta.get("next_token") if isinstance(meta, dict) else None
        if not isinstance(nt, str):
            return
        params["next_token"] = nt


# ---------------------------------------------------------------------------
# Post parsing & quality gate
# ---------------------------------------------------------------------------


def _text(post: dict[str, object]) -> str:
    """Full text, preferring note_tweet for long-form."""
    note = post.get("note_tweet")
    if isinstance(note, dict):
        t = note.get("text")
        if isinstance(t, str):
            return t
    t = post.get("text")
    return t if isinstance(t, str) else ""


def _metrics(raw: dict[str, object]) -> dict[str, int]:
    m = raw.get("public_metrics")
    if not isinstance(m, dict):
        return {}
    return {k: v for k, v in m.items() if isinstance(k, str) and isinstance(v, int)}


def _entities(post: dict[str, object]) -> dict[str, object]:
    """Prefer note_tweet.entities for long-form posts."""
    note = post.get("note_tweet")
    src = (
        note.get("entities")
        if isinstance(note, dict) and isinstance(note.get("entities"), dict)
        else post.get("entities")
    )
    return src if isinstance(src, dict) else {}


def _worth_ingesting(
    text: str, m: dict[str, int], user: dict[str, object] | None, post: dict[str, object]
) -> bool:
    """Quality gate: accept if text is long enough AND has substance, likes, or verification."""
    if len(text) < MIN_TEXT_LEN:
        return False
    ent = _entities(post)
    substance = bool(ent.get("urls") or ent.get("annotations") or post.get("context_annotations"))
    verified = bool(user and user.get("verified"))
    return substance or m.get("like_count", 0) >= MIN_LIKES or verified


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _k(n: int) -> str:
    """Format a number compactly: 1500 → '1.5K', 2000000 → '2.0M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _enrich(
    text: str, m: dict[str, int], user: dict[str, object] | None, post: dict[str, object]
) -> str:
    """Enriched text with credibility markers for ESS classification."""
    parts: list[str] = []
    if user:
        v = "verified" if user.get("verified") else "unverified"
        um = _metrics(user)
        parts.append(
            f"[X/{v}/@{user.get('username', '?')}, "
            f"{_k(um.get('followers_count', 0))} followers, "
            f"{_k(m.get('like_count', 0))} likes, "
            f"{_k(m.get('bookmark_count', 0))} saves]"
        )
    parts.append(text)

    ent = _entities(post)
    urls = ent.get("urls")
    if isinstance(urls, list):
        for u in urls[:3]:
            if isinstance(u, dict) and isinstance(u.get("title"), str) and u["title"]:
                parts.append(f"Linked: {u['title']} ({u.get('display_url', '')})")

    uid = user.get("username", "?") if user else "?"
    parts.append(f"Source: x.com/{uid}/status/{post.get('id', '?')}")
    return "\n\n".join(parts)


def _panel(text: str, m: dict[str, int], user: dict[str, object] | None) -> Panel:
    """Build a Rich panel for displaying a tweet in the terminal."""
    lines: list[str] = []
    if user:
        um = _metrics(user)
        tick = " ✓" if user.get("verified") else ""
        lines.append(
            f"[bold]@{user.get('username', '?')}[/bold]{tick}"
            f" · {_k(um.get('followers_count', 0))} followers"
        )
        lines.append("")
    body = escape(text[:300])
    if len(text) > 300:
        body += "…"
    lines.append(body)
    lines.append(
        f"[dim]{_k(m.get('like_count', 0))} likes"
        f" · {_k(m.get('retweet_count', 0))} RT"
        f" · {_k(m.get('bookmark_count', 0))} saves[/dim]"
    )
    return Panel("\n".join(lines), border_style="blue", padding=(0, 1))


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> None:
    """Fetch posts from X API, ingest quality-gated content into Sonality."""
    load_dotenv()
    token = os.getenv("X_BEARER_TOKEN", "")
    base = os.getenv("SONALITY_URL", "http://localhost:8000")

    if not token:
        console.print("[red]X_BEARER_TOKEN not set — developer.x.com[/red]")
        return

    cfg = Config(
        max_results=int(os.getenv("X_FEED_MAX_RESULTS", "10")),
        max_pages=int(os.getenv("X_FEED_MAX_PAGES", "1")),
        sort_order=os.getenv("X_FEED_SORT_ORDER", "relevancy"),
        throttle=float(os.getenv("X_FEED_THROTTLE", "2")),
    )

    queries = dict(QUERIES)
    tag_filter = os.getenv("X_FEED_QUERIES", "")
    if tag_filter:
        wanted = {t.strip() for t in tag_filter.split(",") if t.strip()}
        queries = {k: v for k, v in queries.items() if k in wanted}
        if not queries:
            console.print(f"[red]No matching queries: {tag_filter}[/red]")
            return

    console.print(f"[bold blue]X Feed[/bold blue] → {base}")
    console.print(
        f"[dim]{len(queries)} queries · {cfg.sort_order} · throttle {cfg.throttle}s[/dim]\n"
    )

    with (
        httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=60.0) as xc,
        httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as sc,
    ):
        if not connect_or_exit(sc, base):
            return

        seen: set[str] = set()
        ok = total = 0

        for tag, (query, topic) in queries.items():
            console.print(f"[bold cyan]{tag}[/bold cyan]")

            for post, user in _search(xc, query, cfg):
                pid = post.get("id")
                if not isinstance(pid, str) or pid in seen:
                    continue
                seen.add(pid)

                text = _text(post)
                m = _metrics(post)
                if not _worth_ingesting(text, m, user, post):
                    continue

                total += 1
                console.print(_panel(text, m, user))
                success, _ = queue_ingest(
                    sc, base, _enrich(text, m, user, post), topic, cfg.throttle
                )
                if success:
                    ok += 1

        console.print(f"\n[bold]Done — {ok}/{total} queued[/bold]\n")
        fetch_and_show_beliefs(sc, base)


if __name__ == "__main__":
    main()
