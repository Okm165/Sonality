#!/usr/bin/env python3
"""Feed news articles to Sonality — fetch one, ingest one, repeat.

Sources: topic-organized RSS feeds + GNews, mirroring x_feed's topic tags.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import feedparser  # type: ignore[import-untyped]
import httpx
from _helpers import connect_or_exit, fetch_and_show_beliefs, queue_ingest
from dotenv import load_dotenv
from gnews import GNews  # type: ignore[import-untyped]
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

# tag → tuple of (source_name, url) — mirrors x_feed's tag → (query, topic_override)
FEEDS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "geopolitics": (
        ("BBC World", "http://feeds.bbci.co.uk/news/world/rss.xml"),
        ("France24", "https://www.france24.com/en/rss"),
        ("DW World", "https://rss.dw.com/rdf/rss-en-all"),
        ("VOA News", "https://www.voanews.com/api/zq_opetivm"),
        ("NPR World", "https://feeds.npr.org/1004/rss.xml"),
        ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("Guardian World", "https://www.theguardian.com/world/rss"),
        ("UN News", "https://news.un.org/feed/subscribe/en/news/all/rss.xml"),
        ("Al-Monitor", "https://www.al-monitor.com/rss"),
        ("Middle East Eye", "https://www.middleeasteye.net/rss"),
        ("Times of Israel", "https://www.timesofisrael.com/feed/"),
        ("SCMP Asia", "https://www.scmp.com/rss/91/feed"),
        ("Straits Times", "https://www.straitstimes.com/news/world/rss.xml"),
        ("Africanews", "https://www.africanews.com/feed/"),
        ("MercoPress", "https://en.mercopress.com/rss"),
    ),
    "conflict": (
        ("Defense One", "https://www.defenseone.com/rss/all/"),
        ("War on the Rocks", "https://warontherocks.com/feed/"),
        ("Breaking Defense", "https://breakingdefense.com/feed/"),
        ("The War Zone", "https://www.thedrive.com/the-war-zone/feed"),
        ("Bellingcat", "https://www.bellingcat.com/feed/"),
        ("Kyiv Independent", "https://kyivindependent.com/feed/"),
        ("Crisis Group", "https://www.crisisgroup.org/rss.xml"),
    ),
    "ai_tech": (
        ("MIT Tech Review", "https://www.technologyreview.com/feed/"),
        ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
        ("Wired", "https://www.wired.com/feed/rss"),
        ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ),
    "crypto": (
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
        ("Decrypt", "https://decrypt.co/feed"),
        ("The Block", "https://www.theblock.co/rss.xml"),
    ),
    "markets": (
        ("CNBC Markets", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
        ("FT World", "https://www.ft.com/world?format=rss"),
        ("Nikkei Asia", "https://asia.nikkei.com/rss/feed/nar"),
    ),
    "economics": (
        ("OilPrice", "https://oilprice.com/rss/main"),
        ("World Bank", "https://blogs.worldbank.org/feed"),
        ("RAND", "https://www.rand.org/news/rss.xml"),
        ("Atlantic Council", "https://www.atlanticcouncil.org/feed/"),
    ),
    "science": (
        ("Nature", "https://www.nature.com/nature.rss"),
        ("Science Daily", "https://www.sciencedaily.com/rss/all.xml"),
        ("Phys.org", "https://phys.org/rss-feed/"),
        ("WHO News", "https://www.who.int/rss-feeds/news-english.xml"),
    ),
    "politics": (
        ("POLITICO EU", "https://www.politico.eu/feed/"),
        ("The Intercept", "https://theintercept.com/feed/?rss"),
        ("ProPublica", "https://www.propublica.org/feeds/propublica/main"),
        ("OCCRP", "https://www.occrp.org/en/component/ocrss/?format=feed"),
        ("DW Europe", "https://rss.dw.com/rdf/rss-en-eu"),
    ),
}

# Same topic_override values as x_feed's QUERIES
TOPIC_OVERRIDES: Final[dict[str, str]] = {
    "geopolitics": "geopolitics",
    "conflict": "military_conflict",
    "ai_tech": "artificial_intelligence",
    "crypto": "cryptocurrency",
    "markets": "financial_markets",
    "economics": "economics",
    "science": "",
    "politics": "politics",
}

# GNews topic → topic_override (same mapping)
GNEWS_MAP: Final[dict[str, str]] = {
    "WORLD": "geopolitics",
    "NATION": "politics",
    "BUSINESS": "financial_markets",
    "TECHNOLOGY": "artificial_intelligence",
    "SCIENCE": "",
    "HEALTH": "",
}
GNEWS_COUNTRIES: Final = ("US", "GB", "IN", "AU", "CA")

MIN_DESC_LEN: Final = 40

console = Console()


@dataclass(frozen=True, slots=True)
class Article:
    """Single news article from RSS or GNews."""

    title: str
    description: str
    source: str
    link: str
    topic: str


# ---------------------------------------------------------------------------
# Source generators
# ---------------------------------------------------------------------------


def _rss(source: str, url: str, topic: str, limit: int) -> Iterator[Article]:
    """Yield articles from a single RSS feed."""
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:limit]:
            yield Article(
                title=str(entry.get("title") or ""),
                description=str(entry.get("summary") or entry.get("description") or ""),
                source=source,
                link=str(entry.get("link") or ""),
                topic=topic,
            )
    except Exception as exc:
        console.print(f"  [dim yellow]{source}: {exc}[/dim yellow]")


def _gnews(country: str, gnews_topic: str, topic: str, limit: int) -> Iterator[Article]:
    """Yield articles from GNews for one country+topic pair."""
    try:
        gn = GNews(language="en", country=country, max_results=limit)
        for item in gn.get_news_by_topic(gnews_topic) or []:
            yield Article(
                title=item.get("title", ""),
                description=item.get("description", ""),
                source=f"GNews/{country}/{gnews_topic}",
                link=item.get("url", ""),
                topic=topic,
            )
    except Exception as exc:
        console.print(f"  [dim yellow]GNews {country}/{gnews_topic}: {exc}[/dim yellow]")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _panel(article: Article) -> Panel:
    """Build a Rich panel for displaying an article in the terminal."""
    body = escape(article.description[:300])
    if len(article.description) > 300:
        body += "…"
    content = (
        f"[bold]{escape(article.title)}[/bold]\n\n{body}"
        f"\n\n[dim]{article.source} · {article.link}[/dim]"
    )
    return Panel(content, border_style="green", padding=(0, 1))


_HTML_TAG = re.compile(r"<[^>]+>")


def _enrich(article: Article) -> str:
    """Enriched text with source credibility marker for ESS classification."""
    desc = _HTML_TAG.sub("", article.description).strip()
    return f"[News/{article.source}]\n\n{article.title}\n\n{desc}\n\nSource: {article.link}"


def _ingest_one(
    client: httpx.Client,
    base: str,
    article: Article,
    topic: str,
    seen: set[str],
    ok: int,
    total: int,
    throttle: float,
) -> tuple[int, int]:
    """Filter, display, and fire-and-forget ingest a single article."""
    key = article.link or article.title
    if not key or key in seen:
        return ok, total
    if not article.title or len(article.description) < MIN_DESC_LEN:
        return ok, total
    seen.add(key)

    total += 1
    console.print(_panel(article))
    success, _ = queue_ingest(client, base, _enrich(article), topic, throttle)
    if success:
        ok += 1
    return ok, total


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> None:
    """Fetch news from RSS + GNews, ingest each article into Sonality."""
    load_dotenv()
    base = os.getenv("SONALITY_URL", "http://localhost:8000")
    throttle = float(os.getenv("FEED_THROTTLE", "5"))
    gnews_limit = int(os.getenv("GNEWS_LIMIT", "5"))
    rss_entries = int(os.getenv("RSS_ENTRIES", "5"))
    tag_filter = os.getenv("FEED_TAGS", "")

    feeds = dict(FEEDS)
    if tag_filter:
        wanted = {t.strip() for t in tag_filter.split(",") if t.strip()}
        feeds = {k: v for k, v in feeds.items() if k in wanted}
        if not feeds:
            console.print(f"[red]No matching tags: {tag_filter}[/red]")
            return

    n_sources = sum(len(sources) for sources in feeds.values())
    console.print(f"[bold blue]News Feed[/bold blue] → {base}")
    console.print(
        f"[dim]{len(feeds)} tags · {n_sources} RSS + GNews "
        f"({len(GNEWS_COUNTRIES)}×{len(GNEWS_MAP)}) · throttle {throttle}s[/dim]\n"
    )

    with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        if not connect_or_exit(client, base):
            return

        seen: set[str] = set()
        ok = total = 0

        # Phase 1: RSS feeds by topic tag (mirrors x_feed's query loop)
        for tag, sources in feeds.items():
            topic = TOPIC_OVERRIDES.get(tag, "")
            console.print(f"[bold cyan]{tag}[/bold cyan]")

            for name, url in sources:
                for article in _rss(name, url, topic, rss_entries):
                    ok, total = _ingest_one(client, base, article, topic, seen, ok, total, throttle)

        # Phase 2: GNews (same topics, mapped to same overrides)
        for gnews_topic, topic in GNEWS_MAP.items():
            for country in GNEWS_COUNTRIES:
                console.print(f"[bold cyan]GNews/{country}/{gnews_topic}[/bold cyan]")

                for article in _gnews(country, gnews_topic, topic, gnews_limit):
                    ok, total = _ingest_one(client, base, article, topic, seen, ok, total, throttle)

        console.print(f"\n[bold]Done — {ok}/{total} queued[/bold]\n")
        fetch_and_show_beliefs(client, base)


if __name__ == "__main__":
    main()
