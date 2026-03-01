"""
market_data.py
--------------
Live price and company overview (sector) fetching for the portfolio editor.

Priority for quotes:
  1. FMP /quote/{ticker}       (fast, reliable — uses FMP_API_KEY)
  2. Alpha Vantage GLOBAL_QUOTE (fallback — uses ALPHA_VANTAGE_API_KEY)

Sector detection:
  Alpha Vantage OVERVIEW → mapped to GICS sector names.

Caching:
  - Prices: 15-minute TTL (module-level dict)
  - Sectors: 24-hour TTL (module-level dict)
  Both caches survive for the lifetime of the process only.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_FMP_KEY = os.getenv("FMP_API_KEY", "")
_AV_KEY  = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# ── Caches ─────────────────────────────────────────────────────────────────────
_price_cache:  dict[str, tuple[float, datetime]] = {}   # ticker → (price, fetched_at)
_sector_cache: dict[str, tuple[str, datetime]]   = {}   # ticker → (sector, fetched_at)

_PRICE_TTL  = timedelta(minutes=15)
_SECTOR_TTL = timedelta(hours=24)

# ── Alpha Vantage → GICS sector name mapping ───────────────────────────────────
_AV_TO_GICS: dict[str, str] = {
    "technology":              "Technology",
    "information technology":  "Technology",
    "healthcare":              "Healthcare",
    "health care":             "Healthcare",
    "finance":                 "Financials",
    "financial services":      "Financials",
    "financials":              "Financials",
    "consumer cyclical":       "Consumer Discretionary",
    "consumer discretionary":  "Consumer Discretionary",
    "consumer defensive":      "Consumer Staples",
    "consumer staples":        "Consumer Staples",
    "industrials":             "Industrials",
    "industrial conglomerates":"Industrials",
    "energy":                  "Energy",
    "basic materials":         "Materials",
    "materials":               "Materials",
    "utilities":               "Utilities",
    "real estate":             "Real Estate",
    "communication services":  "Communication Services",
    "telecommunications":      "Communication Services",
}

# Static fallback for common ETFs (AV OVERVIEW doesn't return sector for ETFs)
_ETF_SECTOR: dict[str, str] = {
    "SPY": "Diversified", "IVV": "Diversified", "VOO": "Diversified",
    "QQQ": "Technology",  "VTI": "Diversified", "IWM": "Diversified",
    "XLF": "Financials",  "XLK": "Technology",  "XLE": "Energy",
    "XLV": "Healthcare",  "XLI": "Industrials", "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples", "XLU": "Utilities", "XLRE": "Real Estate",
    "XLC": "Communication Services", "XLB": "Materials",
    "GLD": "Commodities", "SLV": "Commodities", "USO": "Energy",
    "ARKK": "Technology", "SOXX": "Technology", "SMH": "Technology",
    "TLT": "Fixed Income", "AGG": "Fixed Income", "BND": "Fixed Income",
    "HYG": "Fixed Income", "LQD": "Fixed Income",
}

# Static sector map for the most common S&P 500 / NASDAQ stocks (GICS-based).
# Checked against FMP profile data. Used as instant fallback when APIs are
# unavailable so sector auto-detection always works for popular tickers.
_STOCK_SECTOR: dict[str, str] = {
    # ── Technology ────────────────────────────────────────────────────────
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ORCL": "Technology", "QCOM": "Technology",
    "INTC": "Technology", "AMD": "Technology",  "TXN": "Technology",
    "ADI":  "Technology", "MU":   "Technology", "AMAT": "Technology",
    "LRCX": "Technology", "KLAC": "Technology", "MRVL": "Technology",
    "CRM":  "Technology", "NOW":  "Technology", "ADBE": "Technology",
    "INTU": "Technology", "SNPS": "Technology", "CDNS": "Technology",
    "PANW": "Technology", "CRWD": "Technology", "FTNT": "Technology",
    "SNOW": "Technology", "PLTR": "Technology", "DDOG": "Technology",
    "ZS":   "Technology", "NET":  "Technology", "OKTA": "Technology",
    "ANET": "Technology", "HPQ":  "Technology", "HPE":  "Technology",
    "IBM":  "Technology", "ACN":  "Technology", "CTSH": "Technology",
    "ITUB": "Technology", "TSM":  "Technology", "ASML": "Technology",
    "SAP":  "Technology",
    # ── Communication Services ────────────────────────────────────────────
    "GOOGL": "Communication Services", "GOOG": "Communication Services",
    "META":  "Communication Services", "NFLX": "Communication Services",
    "DIS":   "Communication Services", "CMCSA": "Communication Services",
    "T":     "Communication Services", "VZ":   "Communication Services",
    "TMUS":  "Communication Services", "SNAP": "Communication Services",
    "PINS":  "Communication Services", "RBLX": "Communication Services",
    "EA":    "Communication Services", "TTWO": "Communication Services",
    "WBD":   "Communication Services", "PARA": "Communication Services",
    # ── Consumer Discretionary ────────────────────────────────────────────
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD":   "Consumer Discretionary", "MCD":  "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "NKE":  "Consumer Discretionary",
    "LOW":  "Consumer Discretionary", "TGT":  "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "LULU": "Consumer Discretionary",
    "ABNB": "Consumer Discretionary", "LYFT": "Consumer Discretionary",
    "UBER": "Consumer Discretionary", "GM":   "Consumer Discretionary",
    "F":    "Consumer Discretionary", "RIVN": "Consumer Discretionary",
    "LCID": "Consumer Discretionary", "TJX":  "Consumer Discretionary",
    "EBAY": "Consumer Discretionary", "ETSY": "Consumer Discretionary",
    # ── Consumer Staples ──────────────────────────────────────────────────
    "WMT": "Consumer Staples", "COST": "Consumer Staples",
    "PG":  "Consumer Staples", "KO":   "Consumer Staples",
    "PEP": "Consumer Staples", "PM":   "Consumer Staples",
    "MO":  "Consumer Staples", "MDLZ": "Consumer Staples",
    "CL":  "Consumer Staples", "GIS":  "Consumer Staples",
    "KHC": "Consumer Staples", "STZ":  "Consumer Staples",
    # ── Healthcare ────────────────────────────────────────────────────────
    "LLY": "Healthcare", "JNJ":  "Healthcare", "UNH": "Healthcare",
    "MRK": "Healthcare", "ABBV": "Healthcare", "PFE": "Healthcare",
    "TMO": "Healthcare", "ABT":  "Healthcare", "DHR": "Healthcare",
    "BMY": "Healthcare", "AMGN": "Healthcare", "GILD": "Healthcare",
    "ISRG":"Healthcare", "REGN": "Healthcare", "VRTX": "Healthcare",
    "BIIB":"Healthcare", "MRNA": "Healthcare", "CVS":  "Healthcare",
    "CI":  "Healthcare", "HUM":  "Healthcare", "ELV": "Healthcare",
    "MDT": "Healthcare", "SYK":  "Healthcare", "BSX": "Healthcare",
    "ZTS": "Healthcare",
    # ── Financials ────────────────────────────────────────────────────────
    "BRK.B":"Financials", "BRK.A":"Financials",
    "JPM": "Financials", "BAC":  "Financials", "WFC": "Financials",
    "GS":  "Financials", "MS":   "Financials", "C":   "Financials",
    "BLK": "Financials", "SCHW": "Financials", "AXP": "Financials",
    "V":   "Financials", "MA":   "Financials", "PYPL": "Financials",
    "COF": "Financials", "USB":  "Financials", "TFC": "Financials",
    "PNC": "Financials", "CB":   "Financials", "MMC": "Financials",
    "ICE": "Financials", "CME":  "Financials", "SPGI":"Financials",
    "MCO": "Financials", "AFL":  "Financials", "AIG": "Financials",
    "MET": "Financials", "PRU":  "Financials",
    # ── Industrials ───────────────────────────────────────────────────────
    "CAT": "Industrials", "GE":  "Industrials", "HON": "Industrials",
    "UNP": "Industrials", "RTX": "Industrials", "LMT": "Industrials",
    "BA":  "Industrials", "DE":  "Industrials", "MMM": "Industrials",
    "UPS": "Industrials", "FDX": "Industrials", "CSX": "Industrials",
    "NSC": "Industrials", "EMR": "Industrials", "ETN": "Industrials",
    "PH":  "Industrials", "ROK": "Industrials", "NOC": "Industrials",
    "GD":  "Industrials", "LHX": "Industrials",
    # ── Energy ────────────────────────────────────────────────────────────
    "XOM": "Energy", "CVX":  "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG":  "Energy", "PXD": "Energy",
    "MPC": "Energy", "PSX":  "Energy", "VLO": "Energy",
    "OXY": "Energy", "HAL":  "Energy", "DVN": "Energy",
    # ── Materials ─────────────────────────────────────────────────────────
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "FCX": "Materials", "NEM": "Materials", "NUE": "Materials",
    "ALB": "Materials",
    # ── Utilities ─────────────────────────────────────────────────────────
    "NEE": "Utilities", "SO":  "Utilities", "DUK": "Utilities",
    "D":   "Utilities", "AEP": "Utilities", "EXC": "Utilities",
    "XEL": "Utilities", "SRE": "Utilities",
    # ── Real Estate ───────────────────────────────────────────────────────
    "PLD": "Real Estate", "AMT": "Real Estate", "EQIX": "Real Estate",
    "CCI": "Real Estate", "PSA": "Real Estate", "O":   "Real Estate",
    "WY":  "Real Estate", "SPG": "Real Estate", "DLR": "Real Estate",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_live_price(ticker: str) -> float | None:
    """Return the latest quote for `ticker`. Returns None on failure.

    Tries FMP first (faster), falls back to Alpha Vantage.
    Results are cached for 15 minutes.
    """
    t = ticker.upper()
    now = datetime.utcnow()

    # Cache hit
    if t in _price_cache:
        price, fetched_at = _price_cache[t]
        if now - fetched_at < _PRICE_TTL:
            return price

    price = _fmp_quote(t) or _av_quote(t)
    if price is not None:
        _price_cache[t] = (price, now)
    return price


def fetch_live_prices(tickers: list[str]) -> dict[str, float]:
    """Batch-fetch prices for multiple tickers. Returns {ticker: price}."""
    result: dict[str, float] = {}
    uncached = []
    now = datetime.utcnow()

    for t in [tk.upper() for tk in tickers]:
        if t in _price_cache:
            price, fetched_at = _price_cache[t]
            if now - fetched_at < _PRICE_TTL:
                result[t] = price
                continue
        uncached.append(t)

    if uncached:
        # FMP batch quote (one call for all)
        batch = _fmp_batch_quote(uncached)
        for t, p in batch.items():
            result[t] = p
            _price_cache[t] = (p, now)
        # Individual AV fallback for anything still missing
        for t in uncached:
            if t not in result:
                p = _av_quote(t)
                if p is not None:
                    result[t] = p
                    _price_cache[t] = (p, now)

    return result


def fetch_sector(ticker: str) -> str | None:
    """Return the GICS sector name for `ticker`. Returns None on failure.

    Priority:
      1. Static ETF map (instant)
      2. FMP profile endpoint (fast, reliable)
      3. Alpha Vantage OVERVIEW (fallback, rate-limited)
    Results cached 24 hours.
    """
    t = ticker.upper()

    # 1. Static maps (instant, no API call needed)
    if t in _ETF_SECTOR:
        return _ETF_SECTOR[t]
    if t in _STOCK_SECTOR:
        return _STOCK_SECTOR[t]

    now = datetime.utcnow()

    # Cache hit
    if t in _sector_cache:
        sector, fetched_at = _sector_cache[t]
        if now - fetched_at < _SECTOR_TTL:
            return sector

    # 2. FMP profile (primary — fast and reliable)
    sector = _fmp_sector(t)
    # 3. Alpha Vantage fallback
    if not sector:
        sector = _av_sector(t)
    if sector:
        _sector_cache[t] = (sector, now)
    return sector


# ---------------------------------------------------------------------------
# Private — FMP
# ---------------------------------------------------------------------------

def _fmp_quote(ticker: str) -> float | None:
    if not _FMP_KEY:
        return None
    try:
        r = requests.get(
            f"https://financialmodelingprep.com/api/v3/quote/{ticker}",
            params={"apikey": _FMP_KEY},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        if data and isinstance(data, list):
            price = data[0].get("price")
            return float(price) if price is not None else None
    except Exception as exc:
        logger.debug("FMP quote %s failed: %s", ticker, exc)
    return None


def _fmp_batch_quote(tickers: list[str]) -> dict[str, float]:
    if not _FMP_KEY or not tickers:
        return {}
    try:
        symbols = ",".join(tickers)
        r = requests.get(
            f"https://financialmodelingprep.com/api/v3/quote/{symbols}",
            params={"apikey": _FMP_KEY},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        result = {}
        if isinstance(data, list):
            for item in data:
                sym = item.get("symbol", "")
                price = item.get("price")
                if sym and price is not None:
                    result[sym.upper()] = float(price)
        return result
    except Exception as exc:
        logger.debug("FMP batch quote failed: %s", exc)
        return {}


def _fmp_sector(ticker: str) -> str | None:
    """Fetch sector from FMP /profile endpoint. Returns GICS-style name or None."""
    if not _FMP_KEY:
        return None
    try:
        r = requests.get(
            f"https://financialmodelingprep.com/api/v3/profile/{ticker}",
            params={"apikey": _FMP_KEY},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        if data and isinstance(data, list):
            sector = data[0].get("sector", "")
            return sector.strip() if sector else None
    except Exception as exc:
        logger.debug("FMP profile %s failed: %s", ticker, exc)
    return None


# ---------------------------------------------------------------------------
# Private — Alpha Vantage
# ---------------------------------------------------------------------------

def _av_quote(ticker: str) -> float | None:
    if not _AV_KEY:
        return None
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": _AV_KEY},
            timeout=10,
        )
        r.raise_for_status()
        price_str = r.json().get("Global Quote", {}).get("05. price")
        return float(price_str) if price_str else None
    except Exception as exc:
        logger.debug("AV quote %s failed: %s", ticker, exc)
    return None


def _av_sector(ticker: str) -> str | None:
    if not _AV_KEY:
        return None
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "OVERVIEW", "symbol": ticker, "apikey": _AV_KEY},
            timeout=10,
        )
        r.raise_for_status()
        raw_sector = r.json().get("Sector", "")
        mapped = _AV_TO_GICS.get(raw_sector.lower())
        if mapped:
            return mapped
        # Return as-is if we at least got something
        return raw_sector.title() if raw_sector else None
    except Exception as exc:
        logger.debug("AV overview %s failed: %s", ticker, exc)
    return None
