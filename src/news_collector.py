"""
news_collector.py
-----------------
Fetches top financial market news headlines.

Priority:
  1. NewsAPI.org  (if NEWS_API_KEY is set — free tier: 100 req/day)
  2. RSS fallback (no key needed) — MarketWatch + Yahoo Finance RSS feeds

Each item returned:
    {"headline": str, "source": str, "url": str, "published_at": str}

Cache TTL: 1 hour (module-level).
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger       = logging.getLogger(__name__)
_NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ── Module-level cache ─────────────────────────────────────────────────────────
_cache: list[dict] | None = None
_cache_ts: datetime | None = None
_CACHE_TTL_MIN = 60

_HEADERS = {
    "User-Agent": "WhaleTracker research@whaletracker.ai",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

# RSS feeds (no API key needed) — newest headlines first
_RSS_FEEDS = [
    ("MarketWatch",  "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("Yahoo Finance","https://finance.yahoo.com/rss/topstories"),
    ("Reuters",      "https://feeds.reuters.com/reuters/businessNews"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_market_news(n: int = 5) -> list[dict[str, Any]]:
    """Return up to `n` top financial news headlines.

    Returns list of {headline, source, url, published_at}.
    Returns empty list on complete failure (never raises).
    """
    global _cache, _cache_ts

    now = datetime.utcnow()
    if (_cache is not None and _cache_ts is not None
            and (now - _cache_ts).total_seconds() < _CACHE_TTL_MIN * 60):
        return _cache[:n]

    items: list[dict] = []

    if _NEWS_API_KEY:
        items = _fetch_newsapi(n * 2)
    if not items:
        items = _fetch_rss(n * 2)

    _cache    = items[:n]
    _cache_ts = now
    return _cache


# ---------------------------------------------------------------------------
# NewsAPI.org
# ---------------------------------------------------------------------------

def _fetch_newsapi(n: int) -> list[dict]:
    try:
        resp = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={
                "category": "business",
                "language": "en",
                "pageSize": min(n, 20),
                "apiKey":   _NEWS_API_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        items = []
        for a in articles:
            headline = a.get("title", "").split(" - ")[0].strip()  # strip source suffix
            if not headline or headline.lower() == "[removed]":
                continue
            items.append({
                "headline":    headline,
                "source":      a.get("source", {}).get("name", "NewsAPI"),
                "url":         a.get("url", ""),
                "published_at":_fmt_date(a.get("publishedAt", "")),
            })
        logger.debug("NewsAPI returned %d headlines", len(items))
        return items
    except Exception as exc:
        logger.warning("NewsAPI fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# RSS fallback
# ---------------------------------------------------------------------------

def _fetch_rss(n: int) -> list[dict]:
    items: list[dict] = []

    for source_name, feed_url in _RSS_FEEDS:
        if len(items) >= n:
            break
        try:
            resp = requests.get(feed_url, headers=_HEADERS, timeout=10)
            if not resp.ok:
                continue
            root = ET.fromstring(resp.content)
            entries = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
            for entry in entries:
                if len(items) >= n:
                    break
                title = (
                    _elem_text(entry, "title")
                    or _elem_text(entry, "{http://www.w3.org/2005/Atom}title")
                    or ""
                ).strip()
                link = (
                    _elem_text(entry, "link")
                    or entry.get("href", "")
                    or _elem_text(entry, "{http://www.w3.org/2005/Atom}link")
                    or ""
                ).strip()
                pub = (
                    _elem_text(entry, "pubDate")
                    or _elem_text(entry, "{http://www.w3.org/2005/Atom}updated")
                    or ""
                )
                if not title:
                    continue
                items.append({
                    "headline":    title,
                    "source":      source_name,
                    "url":         link,
                    "published_at":_fmt_date(pub),
                })
        except Exception as exc:
            logger.debug("RSS fetch failed (%s): %s", source_name, exc)

    logger.debug("RSS fallback returned %d headlines", len(items))
    return items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elem_text(el: ET.Element, tag: str) -> str:
    child = el.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _fmt_date(raw: str) -> str:
    """Normalise various date formats to 'MMM DD, YYYY'."""
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z",
                "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z"):
        try:
            return datetime.strptime(raw[:30], fmt).strftime("%b %d, %Y")
        except (ValueError, TypeError):
            pass
    return raw[:10] if raw else ""
