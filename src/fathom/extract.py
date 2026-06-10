"""trafilatura + selectolax — mechanical content and link extraction."""

from __future__ import annotations

import json as _json
from urllib.parse import urljoin

from selectolax.parser import HTMLParser
from trafilatura import extract as traf_extract

from .models import ExtractedPage, Link

_PREVIEW_CHARS = 800


def extract_content(html: str, url: str) -> ExtractedPage:
    """Strip HTML to clean text + structured links. No LLM, purely mechanical."""
    text = traf_extract(html, output_format="txt", include_links=False) or ""
    tree = HTMLParser(html)
    links = _extract_links(tree, url)
    title_node = tree.css_first("title")
    title = title_node.text(strip=True) if title_node else ""
    return ExtractedPage(
        markdown=text,
        links=links,
        title=title,
        has_content=bool(text.strip()),
    )


def extract_preview(html: str) -> str:
    """Extract a structured content preview from raw HTML for batch gating.

    Pulls title, meta descriptions, Open Graph / article metadata, JSON-LD
    summaries, heading outlines, and leading paragraph text. Capped at ~800
    chars — URLs with previews are preferred when selecting the fetch batch.
    """
    tree = HTMLParser(html)
    parts: list[str] = []

    title_node = tree.css_first("title")
    if title_node:
        parts.append(title_node.text(strip=True))

    desc = _meta_content(tree, 'meta[name="description"]') or _meta_content(
        tree, 'meta[property="og:description"]'
    )
    if desc:
        parts.append(desc)

    meta_tags: list[str] = []
    site_name = _meta_content(tree, 'meta[property="og:site_name"]')
    if site_name:
        meta_tags.append(f"site:{site_name}")
    og_type = _meta_content(tree, 'meta[property="og:type"]')
    if og_type:
        meta_tags.append(f"type:{og_type}")
    author = _meta_content(tree, 'meta[name="author"]') or _meta_content(
        tree, 'meta[property="article:author"]'
    )
    if author:
        meta_tags.append(f"author:{author}")
    pub_date = _meta_content(tree, 'meta[property="article:published_time"]') or _meta_content(
        tree, 'meta[name="date"]'
    )
    if pub_date:
        meta_tags.append(f"date:{pub_date[:10]}")
    if meta_tags:
        parts.append("[" + ", ".join(meta_tags) + "]")

    jsonld = _extract_jsonld_summary(tree)
    if jsonld:
        parts.append(f"schema: {jsonld}")

    headings = _extract_headings(tree, max_headings=4)
    if headings:
        parts.append("sections: " + " / ".join(headings))

    char_count = sum(len(p) for p in parts)
    if char_count < 400:
        lead = _extract_lead_paragraphs(tree, budget=_PREVIEW_CHARS - char_count)
        if lead:
            parts.append(lead)

    return " | ".join(parts)[:_PREVIEW_CHARS]


def _meta_content(tree: HTMLParser, selector: str) -> str:
    node = tree.css_first(selector)
    if node:
        return (node.attributes.get("content") or "").strip()[:200]
    return ""


def _extract_headings(tree: HTMLParser, *, max_headings: int = 4) -> list[str]:
    """Return first few h1/h2 texts — gives a content outline."""
    headings: list[str] = []
    for tag in ("h1", "h2"):
        for node in tree.css(tag):
            text = node.text(strip=True)[:80]
            if text and text not in headings:
                headings.append(text)
                if len(headings) >= max_headings:
                    return headings
    return headings


def _extract_lead_paragraphs(tree: HTMLParser, *, budget: int = 400) -> str:
    """Extract text from the first real <p> tags — avoids nav/footer noise."""
    parts: list[str] = []
    total = 0
    for node in tree.css("p"):
        text = node.text(strip=True)
        if len(text) < 40:
            continue
        parts.append(text)
        total += len(text)
        if total >= budget:
            break
    return " ".join(parts)[:budget]


def _extract_jsonld_summary(tree: HTMLParser) -> str:
    """Pull a one-line summary from the first JSON-LD block (headline + description)."""
    for script in tree.css('script[type="application/ld+json"]'):
        raw = script.text(strip=True)
        if not raw:
            continue
        try:
            data = _json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            continue
        bits: list[str] = []
        for key in ("headline", "name", "description"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                bits.append(val.strip()[:120])
        if bits:
            return " — ".join(bits)[:200]
    return ""


def _extract_links(tree: HTMLParser, base_url: str) -> list[Link]:
    """Extract <a> tags with href, anchor text, and surrounding context."""
    links: list[Link] = []
    for node in tree.css("a[href]"):
        href = node.attributes.get("href", "")
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        resolved = urljoin(base_url, href)
        if not resolved.startswith("http"):
            continue
        anchor = node.text(strip=True) or ""
        parent = node.parent
        context = parent.text(strip=True)[:200] if parent else ""
        links.append(Link(url=resolved, anchor_text=anchor, context=context))
    return links
