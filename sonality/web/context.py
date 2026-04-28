"""Web context formatting and prompt injection defense.

Sanitizes untrusted web content and formats it for safe injection into LLM
prompts using dual delimiter pattern: outer <web_data> XML boundary + inner
[SOURCE N START/END] numbered delimiters.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from .search import SearchResult

_INJECTION_PATTERNS: Final = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(previous|above|all|prior)\s+(instructions?|prompts?|rules?)",
        r"you\s+are\s+now\s+",
        r"system\s*:\s*",
        r"<\s*/?(?:system|instruction|prompt|admin|override)",
        r"forget\s+(?:everything|all|your|previous)",
        r"new\s+(?:instruction|role|persona|identity|system)",
        r"disregard\s+(?:previous|above|all|prior)",
        r"override\s+(?:instructions?|prompt|rules?|mode)",
        r"\bDAN\b.*\bmode\b",
        r"jailbreak",
        r"act\s+as\s+(?:a|an|the)\s+(?:different|new)",
    ]
]

_DELIMITER_PATTERNS: Final = [
    re.compile(r"\[SOURCE\s+\d+\s+(?:START|END)\]", re.IGNORECASE),
    re.compile(r"</?web_data[^>]*>", re.IGNORECASE),
]

_UNICODE_TAG_RANGE: Final = range(0xE0001, 0xE007F + 1)
_ZERO_WIDTH_CHARS: Final = frozenset({"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u00ad"})
_HTML_TAG_RE: Final = re.compile(r"<[^>]{1,200}>")
_BLANK_LINES_RE: Final = re.compile(r"\n{3,}")
_TRAILING_WS_RE: Final = re.compile(r"[ \t]+$", re.MULTILINE)


def sanitize_web_content(text: str, *, max_chars: int = 500) -> str:
    """Strip prompt injection patterns and dangerous artifacts from web content.

    Preserves markdown line structure (newlines, indentation for code blocks)
    while collapsing excessive blank lines and stripping injection payloads.
    """
    text = "".join(
        c for c in text if ord(c) not in _UNICODE_TAG_RANGE and c not in _ZERO_WIDTH_CHARS
    )
    text = unicodedata.normalize("NFC", text)
    text = _HTML_TAG_RE.sub(" ", text)
    for pattern in _INJECTION_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    for pattern in _DELIMITER_PATTERNS:
        text = pattern.sub("", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    text = _TRAILING_WS_RE.sub("", text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0] + "\n..."
    return text


def format_web_context(
    results: list[SearchResult],
    *,
    max_chars: int = 4000,
) -> str:
    """Format search results with defense-in-depth delimiters.

    Uses dual delimiter pattern: outer <web_data> XML boundary for
    instruction-level isolation, inner [SOURCE N] numbered delimiters
    for per-result boundaries. Markdown content is preferred over
    description snippets. Callers should pre-slice results.
    """
    if not results:
        return ""

    lines: list[str] = ["<web_data>"]
    total_chars = 0
    n = len(results)

    for i, result in enumerate(results[:n], 1):
        per_limit = max_chars // n
        remaining = max_chars - total_chars
        if remaining <= 0:
            break

        best_content = result.markdown or result.description
        sanitized = sanitize_web_content(best_content, max_chars=min(per_limit, remaining))

        lines.append(f"[SOURCE {i} START]")
        lines.append(f"Domain: {result.domain}")
        if result.title:
            lines.append(f"Title: {result.title}")
        lines.append(sanitized)
        lines.append(f"[SOURCE {i} END]")
        lines.append("")
        total_chars += len(sanitized)

    lines.append("</web_data>")
    return "\n".join(lines)
