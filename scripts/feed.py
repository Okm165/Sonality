#!/usr/bin/env python3
"""Feed news articles to Sonality for belief formation."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
import httpx
from dotenv import load_dotenv
from gnews import GNews
from rich.console import Console
from rich.table import Table

# Comprehensive global RSS feeds organized by category (58 feeds, ~40 typically work)
RSS_FEEDS = {
    # === Wire Services ===
    "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "France24": "https://www.france24.com/en/rss",
    "DW World": "https://rss.dw.com/rdf/rss-en-all",
    "VOA News": "https://www.voanews.com/api/zq_opetivm",
    "NPR World": "https://feeds.npr.org/1004/rss.xml",
    # === Middle East ===
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "Al-Monitor": "https://www.al-monitor.com/rss",
    "Jerusalem Post": "http://www.jpost.com/RSS/RssFeedsHeadlines.aspx",
    "Middle East Eye": "https://www.middleeasteye.net/rss",
    "Times of Israel": "https://www.timesofisrael.com/feed/",
    "Haaretz": "https://www.haaretz.com/srv/haaretz-latest-headlines",
    # === Asia ===
    "SCMP Asia": "https://www.scmp.com/rss/91/feed",
    "Japan Times": "https://www.japantimes.co.jp/feed/",
    "Nikkei Asia": "https://asia.nikkei.com/rss/feed/nar",
    "Bangkok Post": "https://www.bangkokpost.com/rss/data/topstories.xml",
    "Straits Times": "https://www.straitstimes.com/news/world/rss.xml",
    "The Hindu World": "https://www.thehindu.com/news/international/feeder/default.rss",
    "NDTV World": "https://feeds.feedburner.com/ndtvnews-world-news",
    "Xinhua": "http://www.news.cn/english/rss/worldrss.xml",
    "CGTN": "https://www.cgtn.com/subscribe/rss/section/world.xml",
    "Channel News Asia": "https://www.channelnewsasia.com/rssfeeds/8395986",
    # === Europe ===
    "POLITICO EU": "https://www.politico.eu/feed/",
    "DW Europe": "https://rss.dw.com/rdf/rss-en-eu",
    "Guardian World": "https://www.theguardian.com/world/rss",
    "The Local EU": "https://www.thelocal.com/feeds/rss.php",
    "EUobserver": "https://euobserver.com/rss.xml",
    "Baltic Times": "https://www.baltictimes.com/rss/",
    # === Russia/Eurasia ===
    "RT News": "https://www.rt.com/rss/news/",
    "Moscow Times": "https://www.themoscowtimes.com/rss/news",
    "Kyiv Independent": "https://kyivindependent.com/feed/",
    "Meduza": "https://meduza.io/rss/en/all",
    # === Africa ===
    "Africanews": "https://www.africanews.com/feed/",
    "Daily Maverick": "https://www.dailymaverick.co.za/dmrss/",
    "Punch Nigeria": "https://punchng.com/feed/",
    "News24 SA": "https://feeds.news24.com/articles/news24/TopStories/rss",
    "Ghana Web": "https://www.ghanaweb.com/GhanaHomePage/rss/",
    # === Latin America ===
    "MercoPress": "https://en.mercopress.com/rss",
    "Buenos Aires Times": "https://www.batimes.com.ar/feed",
    "Mexico News Daily": "https://mexiconewsdaily.com/feed/",
    "Rio Times": "https://riotimesonline.com/feed/",
    "Tico Times": "https://ticotimes.net/feed",
    # === Think Tanks & Analysis ===
    "RAND": "https://www.rand.org/news/rss.xml",
    "Crisis Group": "https://www.crisisgroup.org/rss.xml",
    "Atlantic Council": "https://www.atlanticcouncil.org/feed/",
    "Stimson Center": "https://www.stimson.org/feed/",
    "Wilson Center": "https://www.wilsoncenter.org/rss.xml",
    # === OSINT & Investigative ===
    "Bellingcat": "https://www.bellingcat.com/feed/",
    "The Intercept": "https://theintercept.com/feed/?rss",
    "ProPublica": "https://www.propublica.org/feeds/propublica/main",
    "OCCRP": "https://www.occrp.org/en/component/ocrss/?format=feed",
    # === Defense & Security ===
    "Defense One": "https://www.defenseone.com/rss/all/",
    "War on the Rocks": "https://warontherocks.com/feed/",
    "Breaking Defense": "https://breakingdefense.com/feed/",
    "The War Zone": "https://www.thedrive.com/the-war-zone/feed",
    # === Official Sources ===
    "UN News": "https://news.un.org/feed/subscribe/en/news/all/rss.xml",
    "WHO News": "https://www.who.int/rss-feeds/news-english.xml",
    "World Bank": "https://blogs.worldbank.org/feed",
    # === Energy & Economics ===
    "OilPrice": "https://oilprice.com/rss/main",
    "FT World": "https://www.ft.com/world?format=rss",
}

GNEWS_TOPICS = ["WORLD", "NATION", "BUSINESS", "TECHNOLOGY", "SCIENCE", "HEALTH"]
GNEWS_COUNTRIES = ["US", "GB", "IN", "AU", "CA"]

console = Console()


def fetch_rss_feed(source: str, url: str, max_entries: int = 10) -> list[dict[str, str]]:
    """Fetch articles from a single RSS feed."""
    articles = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_entries]:
            articles.append({
                "title": entry.get("title", ""),
                "description": entry.get("summary", entry.get("description", "")),
                "source": source,
                "link": entry.get("link", ""),
            })
    except Exception as e:
        console.print(f"[dim yellow]RSS {source}: {e}[/dim yellow]")
    return articles


def fetch_gnews(limit: int = 10) -> list[dict[str, str]]:
    """Fetch articles from GNews across multiple topics and countries."""
    articles = []
    for country in GNEWS_COUNTRIES:
        try:
            gn = GNews(language="en", country=country, max_results=limit)
            for topic in GNEWS_TOPICS:
                for item in gn.get_news_by_topic(topic) or []:
                    articles.append({
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "source": f"GNews/{country}/{topic}",
                        "link": item.get("url", ""),
                    })
        except Exception as e:
            console.print(f"[dim yellow]GNews {country}: {e}[/dim yellow]")
    return articles


def fetch_all_articles(gnews_limit: int = 5, rss_entries: int = 5) -> list[dict[str, str]]:
    """Fetch articles from all sources with parallel RSS fetching."""
    articles: list[dict[str, str]] = []
    seen: set[str] = set()

    def add_unique(items: list[dict[str, str]]) -> int:
        added = 0
        for item in items:
            key = item.get("link") or item.get("title", "")
            if key and key not in seen:
                seen.add(key)
                articles.append(item)
                added += 1
        return added

    # Fetch GNews
    console.print("[dim]Fetching GNews...[/dim]")
    gnews_articles = fetch_gnews(gnews_limit)
    gnews_count = add_unique(gnews_articles)
    console.print(f"[dim]  GNews: {gnews_count} articles[/dim]")

    # Fetch RSS feeds in parallel
    console.print(f"[dim]Fetching {len(RSS_FEEDS)} RSS feeds...[/dim]")
    rss_count = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(fetch_rss_feed, source, url, rss_entries): source
            for source, url in RSS_FEEDS.items()
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                feed_articles = future.result()
                count = add_unique(feed_articles)
                rss_count += count
                if count > 0:
                    console.print(f"[dim]  {source}: {count}[/dim]")
            except Exception as e:
                console.print(f"[dim yellow]  {source}: {e}[/dim yellow]")

    console.print(f"[dim]  RSS total: {rss_count} articles[/dim]")
    return articles


def display_beliefs(beliefs: list[dict[str, object]]) -> None:
    """Display beliefs in a table."""
    if not beliefs:
        console.print("[yellow]No beliefs yet.[/yellow]")
        return

    table = Table(title="Beliefs", header_style="bold magenta")
    table.add_column("Topic", style="cyan")
    table.add_column("Position", justify="right", style="green")
    table.add_column("Confidence", justify="right", style="yellow")

    for b in beliefs[:10]:
        pos = b.get("position", 0)
        conf = b.get("confidence", 0)
        table.add_row(
            str(b.get("topic", "")),
            f"{pos:+.2f}" if isinstance(pos, (int, float)) else str(pos),
            f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf),
        )
    console.print(table)


def main() -> None:
    load_dotenv()
    base_url = os.getenv("SONALITY_URL", "http://localhost:8000")
    throttle = float(os.getenv("FEED_THROTTLE", "5"))
    gnews_limit = int(os.getenv("GNEWS_LIMIT", "5"))
    rss_entries = int(os.getenv("RSS_ENTRIES", "5"))

    console.print(f"[bold blue]News Feed[/bold blue] → {base_url}")
    console.print(f"[dim]Config: throttle={throttle}s, gnews_limit={gnews_limit}, rss_entries={rss_entries}[/dim]\n")

    articles = fetch_all_articles(gnews_limit, rss_entries)
    console.print(f"\n[green]Found {len(articles)} unique articles[/green]\n")

    if not articles:
        console.print("[red]No articles fetched. Check network/feeds.[/red]")
        return

    with httpx.Client(timeout=300.0) as client:
        try:
            client.get(f"{base_url}/health").raise_for_status()
            console.print("[green]✓ Connected to Sonality[/green]\n")
        except Exception as e:
            console.print(f"[red]✗ Cannot connect: {e}[/red]")
            return

        ok = 0
        for i, art in enumerate(articles, 1):
            title = art["title"][:50] if art["title"] else "(no title)"
            text = f"{art['title']}\n\n{art['description']}\n\nSource: {art['source']}"

            try:
                r = client.post(f"{base_url}/ingest", json={"text": text})
                r.raise_for_status()
                data = r.json()
                score = data.get("score", 0)
                topics = ", ".join(data.get("topics", [])[:2]) or "-"
                console.print(f"  [green]✓[/green] {i}/{len(articles)} {title}... (ESS={score:.2f} {topics})")
                ok += 1
            except Exception as e:
                console.print(f"  [red]✗[/red] {i}/{len(articles)} {title}... ({e})")

            if i < len(articles) and throttle > 0:
                time.sleep(throttle)

        console.print(f"\n[bold]Ingested {ok}/{len(articles)}[/bold]\n")

        try:
            beliefs = client.get(f"{base_url}/beliefs").json()
            display_beliefs(beliefs if isinstance(beliefs, list) else beliefs.get("beliefs", []))
        except Exception as e:
            console.print(f"[yellow]Could not fetch beliefs: {e}[/yellow]")


if __name__ == "__main__":
    main()
