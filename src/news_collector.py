"""
news_collector.py
-----------------
Fetches top financial market news headlines, filtered for market relevance.

Priority:
  1. NewsAPI.org  (if NEWS_API_KEY is set — free tier: 100 req/day)
  2. RSS fallback (no key needed) — finance-focused feeds

Each item returned:
    {"headline": str, "source": str, "url": str, "published_at": str}

Cache TTL: 1 hour (module-level).
"""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger        = logging.getLogger(__name__)
_NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ── Module-level cache ─────────────────────────────────────────────────────────
_cache: list[dict] | None = None
_cache_ts: datetime | None = None
_CACHE_TTL_MIN = 60

_HEADERS = {
    "User-Agent": "WhaleTracker research@whaletracker.ai",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

# Finance-focused RSS feeds (no API key required)
_RSS_FEEDS = [
    ("CNBC Markets",   "https://www.cnbc.com/id/20910258/device/rss/rss.html"),
    ("MarketWatch",    "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("Reuters Biz",    "https://feeds.reuters.com/reuters/businessNews"),
    ("Yahoo Finance",  "https://finance.yahoo.com/rss/topstories"),
    ("Seeking Alpha",  "https://seekingalpha.com/market_currents.xml"),
    ("Investopedia",   "https://www.investopedia.com/feedbuilder/feed/getfeed?feedName=rss_headline"),
]

# ── Market relevance filter ────────────────────────────────────────────────────

# Presence of ANY of these → headline is market-relevant
_MARKET_KEYWORDS: frozenset[str] = frozenset([
    # indices & general market
    "stock", "stocks", "equity", "equities", "market", "markets",
    "s&p", "s&p 500", "nasdaq", "dow", "dow jones", "nyse", "ftse",
    "nikkei", "hang seng", "index", "indices", "wall street", "trading",
    # macro / monetary policy
    "fed", "federal reserve", "fomc", "rate cut", "rate hike",
    "interest rate", "inflation", "cpi", "pce", "ppi", "deflation",
    "gdp", "jobs report", "payroll", "unemployment", "nonfarm", "nfp",
    "yield", "treasury", "bond", "bonds", "yield curve", "spread", "credit",
    "debt ceiling", "fiscal", "monetary", "quantitative",
    # corporate events
    "earnings", "revenue", "profit", "loss", "eps", "guidance",
    "outlook", "quarterly", "dividend", "buyback", "share repurchase",
    "ipo", "merger", "acquisition", "m&a", "takeover", "deal", "divestiture",
    "shares", "share price", "valuation", "writedown", "restructuring",
    "bankruptcy", "default", "layoffs", "hiring freeze",
    # commodities
    "oil", "crude", "wti", "brent", "natural gas", "gold", "silver",
    "copper", "commodity", "commodities", "energy prices",
    # sectors
    "bank", "banking", "semiconductor", "pharma", "pharmaceutical",
    "healthcare", "biotech", "fintech", "tech sector", "ai stocks",
    # market moves
    "rally", "selloff", "sell-off", "surge", "plunge", "tumble",
    "soar", "climb", "rebound", "correction", "bull market", "bear market",
    "volatile", "volatility", "drawdown", "all-time high", "52-week",
    # analyst activity
    "analyst", "upgrade", "downgrade", "price target", "overweight",
    "underweight", "outperform", "underperform", "buy rating", "sell rating",
    "forecast", "estimate", "consensus",
    # institutional / regulatory
    "sec", "cftc", "imf", "world bank", "ecb", "boe", "bank of japan",
    "hedge fund", "private equity", "asset manager", "institutional",
    "13f", "13d", "activist investor",
    # crypto (market-moving)
    "bitcoin", "ethereum", "crypto", "cryptocurrency", "digital asset",
    # trade / geopolitical with market framing
    "tariff", "trade war", "sanctions", "supply chain", "recession",
])

# Longer phrases — safe to match as substrings
_NOISE_PHRASES: frozenset[str] = frozenset([
    # sports (longer, no substring risk)
    "super bowl", "world cup", "premier league", "la liga", "champions league",
    "championship game", "playoffs", "quarterback", "touchdown",
    "slam dunk", "home run", "hat trick", "transfer fee",
    "soccer match", "basketball game", "football game",
    # entertainment / celebrity
    "grammy", "golden globe", "box office", "kardashian",
    "taylor swift", "concert tour", "movie review", "tv show",
    "netflix series", "disney plus", "streaming show",
    # crime (non-financial)
    "shooting", "murder", "homicide", "robbery", "kidnap",
    # weather (no economic framing)
    "hurricane season", "tornado warning", "earthquake hits",
    "wildfire spreading", "flood damages homes",
    # lifestyle
    "recipe", "best diet", "workout tips", "travel guide",
    "fashion week", "beauty tips", "zodiac", "horoscope",
])

# Short acronyms — must match as whole words (word-boundary) to avoid false positives
# e.g. "nfl" must NOT match inside "inflation" or "inflows"
_NOISE_ACRONYMS: tuple[re.Pattern, ...] = tuple(
    re.compile(rf"\b{re.escape(kw)}\b")
    for kw in ("nba", "nfl", "nhl", "mlb", "fifa", "mls", "ufc")
)


def _is_market_relevant(headline: str) -> bool:
    """Return True only if the headline is relevant to stock/market analysis.

    Logic:
      1. Hard-block obvious noise topics (sports, entertainment, crime, etc.)
      2. Require at least one market/finance keyword to pass.
    """
    h = headline.lower()
    if any(phrase in h for phrase in _NOISE_PHRASES):
        return False
    if any(pat.search(h) for pat in _NOISE_ACRONYMS):
        return False
    return any(kw in h for kw in _MARKET_KEYWORDS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_ticker_news(ticker: str, n: int = 5) -> list[dict[str, Any]]:
    """Return up to `n` recent headlines mentioning `ticker` (case-insensitive).

    Searches the cached general news first; falls back to a fresh NewsAPI
    query if available.  Never raises.
    """
    ticker_upper = ticker.upper()

    # 1. Check already-cached general headlines (free, instant)
    cached = fetch_market_news(50)
    matches = [
        item for item in cached
        if ticker_upper in item.get("headline", "").upper()
    ]
    if len(matches) >= n:
        return matches[:n]

    # 2. Targeted NewsAPI query if key is available
    if _NEWS_API_KEY:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q":        ticker_upper,
                    "language": "en",
                    "pageSize": min(n, 10),
                    "sortBy":   "publishedAt",
                    "apiKey":   _NEWS_API_KEY,
                },
                timeout=8,
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            items = []
            for a in articles:
                headline = a.get("title", "").split(" - ")[0].strip()
                if not headline or headline.lower() == "[removed]":
                    continue
                if not _is_market_relevant(headline):
                    continue
                items.append({
                    "headline":    headline,
                    "source":      a.get("source", {}).get("name", "NewsAPI"),
                    "url":         a.get("url", ""),
                    "published_at": _fmt_date(a.get("publishedAt", "")),
                })
            if items:
                return items[:n]
        except Exception as exc:
            logger.debug("search_ticker_news(%s) NewsAPI failed: %s", ticker, exc)

    return matches[:n]


def fetch_market_news(n: int = 5) -> list[dict[str, Any]]:
    """Return up to `n` market-relevant financial news headlines.

    Returns list of {headline, source, url, published_at}.
    Returns empty list on complete failure (never raises).
    """
    global _cache, _cache_ts

    now = datetime.utcnow()
    if (_cache is not None and _cache_ts is not None
            and (now - _cache_ts).total_seconds() < _CACHE_TTL_MIN * 60):
        return _cache[:n]

    # Fetch a larger pool so the relevance filter has enough candidates
    fetch_target = max(n * 5, 30)

    items: list[dict] = []
    if _NEWS_API_KEY:
        items = _fetch_newsapi(fetch_target)
    if not items:
        items = _fetch_rss(fetch_target)

    _cache    = items
    _cache_ts = now
    return _cache[:n]


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
                "pageSize": min(n, 100),
                "apiKey":   _NEWS_API_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        items = []
        for a in articles:
            headline = a.get("title", "").split(" - ")[0].strip()
            if not headline or headline.lower() == "[removed]":
                continue
            if not _is_market_relevant(headline):
                logger.debug("NewsAPI filtered out: %s", headline[:60])
                continue
            items.append({
                "headline":    headline,
                "source":      a.get("source", {}).get("name", "NewsAPI"),
                "url":         a.get("url", ""),
                "published_at":_fmt_date(a.get("publishedAt", "")),
            })
        logger.debug("NewsAPI: %d/%d passed relevance filter", len(items), len(articles))
        return items
    except Exception as exc:
        logger.warning("NewsAPI fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# RSS fallback
# ---------------------------------------------------------------------------

def _fetch_rss(n: int) -> list[dict]:
    """Fetch from finance RSS feeds, apply relevance filter, return up to n items."""
    raw: list[dict] = []

    for source_name, feed_url in _RSS_FEEDS:
        if len(raw) >= n * 2:   # gather a generous pool before filtering
            break
        try:
            resp = requests.get(feed_url, headers=_HEADERS, timeout=10)
            if not resp.ok:
                continue
            root = ET.fromstring(resp.content)
            entries = (root.findall(".//item")
                       or root.findall(".//{http://www.w3.org/2005/Atom}entry"))
            for entry in entries:
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
                raw.append({
                    "headline":    title,
                    "source":      source_name,
                    "url":         link,
                    "published_at":_fmt_date(pub),
                })
        except Exception as exc:
            logger.debug("RSS fetch failed (%s): %s", source_name, exc)

    # Apply relevance filter
    filtered = [item for item in raw if _is_market_relevant(item["headline"])]
    logger.debug("RSS: %d/%d passed relevance filter", len(filtered), len(raw))
    return filtered[:n]


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
