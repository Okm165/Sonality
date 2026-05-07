"""trafilatura + selectolax — mechanical content and link extraction."""

from __future__ import annotations

from selectolax.parser import HTMLParser
from trafilatura import extract as traf_extract

from .models import ExtractedPage, Link


def extract_content(html: str, url: str) -> ExtractedPage:
    """Strip HTML to clean markdown + structured links. No LLM, purely mechanical."""
    markdown = traf_extract(html, output_format="txt", include_links=False) or ""
    links = _extract_links(html, url)
    title = _extract_title(html)
    return ExtractedPage(
        markdown=markdown,
        links=links,
        title=title,
        has_content=bool(markdown.strip()),
    )


def _extract_links(html: str, base_url: str) -> list[Link]:
    """Extract <a> tags with href, anchor text, and surrounding context."""
    tree = HTMLParser(html)
    links: list[Link] = []
    for node in tree.css("a[href]"):
        href = node.attributes.get("href", "")
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        # Resolve relative URLs
        if href.startswith("/"):
            from urllib.parse import urlparse

            parsed = urlparse(base_url)
            href = f"{parsed.scheme}://{parsed.netloc}{href}"
        elif not href.startswith("http"):
            continue
        anchor = node.text(strip=True) or ""
        # Context: parent node text (trimmed)
        parent = node.parent
        context = parent.text(strip=True)[:200] if parent else ""
        links.append(Link(url=href, anchor_text=anchor, context=context))
    return links


def _extract_title(html: str) -> str:
    tree = HTMLParser(html)
    title_node = tree.css_first("title")
    return title_node.text(strip=True) if title_node else ""
