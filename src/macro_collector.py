"""
macro_collector.py
------------------
Fetches key US macroeconomic indicators from the FRED API
(Federal Reserve Bank of St. Louis — free, no rate limits for research use).

Required env var:
    FRED_API_KEY  — free registration at https://fred.stlouisfed.org/docs/api/api_key.html

If FRED_API_KEY is not set, returns realistic mock data covering the past 2 years.

Each indicator is returned as:
    {
        "name":         str,          # display label
        "unit":         str,          # "%" or "$T" etc.
        "color":        str,          # hex color for chart
        "current":      float,        # latest available value
        "prev":         float,        # value ~1 year ago
        "change_1y":    float,        # current - prev
        "observations": [             # newest first, up to 60 data points
            {"date": "YYYY-MM-DD", "value": float}, ...
        ],
    }
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger    = logging.getLogger(__name__)
_API_KEY  = os.getenv("FRED_API_KEY", "").strip()
_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

_FMP_KEY      = os.getenv("FMP_API_KEY", "").strip()
_FMP_BASE_URL = "https://financialmodelingprep.com/api/v4/economic"
_FMP_PMI_MAP  = {
    "mfg_pmi": "ISM_MANUFACTURING",
    "svc_pmi": "ISM_NON_MANUFACTURING",
}

# ── FRED series definitions ────────────────────────────────────────────────────
_SERIES: dict[str, dict] = {
    "fed_rate": {
        "id":    "FEDFUNDS",
        "name":  "Fed Funds Rate",
        "unit":  "%",
        "color": "#FF4757",
        "desc":  "Federal Reserve target interest rate — the most watched macro lever.",
    },
    "cpi": {
        "id":    "CPIAUCSL",
        "name":  "CPI (YoY %)",
        "unit":  "%",
        "color": "#FFB800",
        "desc":  "Consumer Price Index year-over-year change — the primary inflation gauge.",
        "yoy":   True,   # raw index → compute YoY %
    },
    "yield_10y": {
        "id":    "DGS10",
        "name":  "10-Year Treasury Yield",
        "unit":  "%",
        "color": "#4B7BE5",
        "desc":  "Benchmark long-term rate; rising yields pressure equity valuations.",
    },
    "unemployment": {
        "id":    "UNRATE",
        "name":  "Unemployment Rate",
        "unit":  "%",
        "color": "#A78BFA",
        "desc":  "US civilian unemployment rate — key Fed dual-mandate indicator.",
    },
    "gdp_growth": {
        "id":    "A191RL1Q225SBEA",
        "name":  "Real GDP Growth (QoQ %)",
        "unit":  "%",
        "color": "#00D09C",
        "desc":  "Annualised real GDP growth rate — the broadest economic health signal.",
    },
    "mfg_pmi": {
        "id":    "NAPM",
        "name":  "ISM Mfg PMI",
        "unit":  "",
        "color": "#F97316",
        "desc":  "ISM Manufacturing PMI — monthly survey of purchasing managers. Above 50 = expansion, below 50 = contraction.",
        "pmi":   True,   # mark for special 50-line rendering
    },
    "svc_pmi": {
        "id":    "NMFCI",
        "name":  "ISM Services PMI",
        "unit":  "",
        "color": "#06B6D4",
        "desc":  "ISM Non-Manufacturing (Services) PMI — covers ~80% of US GDP. Above 50 = expansion, below 50 = contraction.",
        "pmi":   True,
    },
}

# ── Module-level cache ─────────────────────────────────────────────────────────
_cache: dict[str, Any] | None = None
_cache_ts: datetime | None    = None
_CACHE_TTL_HOURS               = 24


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_macro_indicators() -> dict[str, dict]:
    """Return all 5 macro indicators, using cache if fresh.

    Returns dict keyed by series identifier (fed_rate, cpi, yield_10y,
    unemployment, gdp_growth).
    """
    global _cache, _cache_ts

    now = datetime.utcnow()
    if (_cache is not None and _cache_ts is not None
            and (now - _cache_ts).total_seconds() < _CACHE_TTL_HOURS * 3600):
        return _cache

    if _API_KEY:
        data = _fetch_from_fred()
    else:
        logger.info("FRED_API_KEY not set — using macro mock data")
        data = _mock_macro()

    _cache    = data
    _cache_ts = now
    return data


# ---------------------------------------------------------------------------
# FRED fetcher
# ---------------------------------------------------------------------------

def _fetch_from_fred() -> dict[str, dict]:
    results: dict[str, dict] = {}
    start_date = (datetime.utcnow() - timedelta(days=760)).strftime("%Y-%m-%d")

    for key, meta in _SERIES.items():
        if meta.get("pmi"):
            continue  # PMI not available on FRED — handled by _fetch_pmi_from_fmp below
        try:
            params = {
                "series_id":        meta["id"],
                "api_key":          _API_KEY,
                "file_type":        "json",
                "sort_order":       "desc",
                "limit":            60,
                "observation_start": start_date,
            }
            resp = requests.get(_BASE_URL, params=params, timeout=12)
            resp.raise_for_status()
            raw_obs = resp.json().get("observations", [])

            obs = [
                {"date": o["date"], "value": float(o["value"])}
                for o in raw_obs
                if o.get("value") not in (".", None, "")
            ]

            if meta.get("yoy") and len(obs) >= 13:
                # Convert CPI index to YoY %: (current / 12-months-ago - 1) * 100
                obs = _cpi_to_yoy(obs)

            if not obs:
                continue

            # obs is newest-first from FRED (sort_order=desc)
            current = obs[0]["value"]
            # ~1 year ago: index 11 for monthly, 3 for quarterly
            year_idx = 11 if len(obs) > 11 else len(obs) - 1
            prev     = obs[year_idx]["value"]

            results[key] = {
                "name":         meta["name"],
                "unit":         meta["unit"],
                "color":        meta["color"],
                "desc":         meta.get("desc", ""),
                "pmi":          False,
                "current":      round(current, 2),
                "prev":         round(prev, 2),
                "change_1y":    round(current - prev, 2),
                "observations": obs[:48],   # keep last 48 data points for chart
            }

        except Exception as exc:
            logger.warning("FRED fetch failed for %s (%s): %s", key, meta["id"], exc)

    # ── PMI from FMP ──────────────────────────────────────────────────────────
    pmi_results = _fetch_pmi_from_fmp()
    results.update(pmi_results)

    # Fill any remaining missing series with mock data
    mock = _mock_macro()
    for key in _SERIES:
        if key not in results:
            mock_entry = dict(mock[key])
            if _SERIES[key].get("pmi"):
                # Label as mock so the UI can show "(mock)" badge
                mock_entry["name"] = mock_entry["name"] + " (mock)"
                mock_entry["is_mock"] = True
            results[key] = mock_entry

    return results


def _cpi_to_yoy(obs: list[dict]) -> list[dict]:
    """Convert a CPI level series (newest-first) to YoY % change."""
    # obs[0] = newest, obs[12] = same month last year
    yoy = []
    for i, item in enumerate(obs):
        if i + 12 >= len(obs):
            break
        past = obs[i + 12]["value"]
        if past:
            yoy.append({
                "date":  item["date"],
                "value": round((item["value"] / past - 1) * 100, 2),
            })
    return yoy


# ---------------------------------------------------------------------------
# FMP PMI fetcher
# ---------------------------------------------------------------------------

def _fetch_pmi_from_fmp() -> dict[str, dict]:
    """Fetch ISM PMI data from FMP /v4/economic.

    Returns dict with 'mfg_pmi' and/or 'svc_pmi' keys on success.
    Returns {} if FMP_API_KEY is not set or all requests fail.
    """
    if not _FMP_KEY:
        return {}

    results: dict[str, dict] = {}
    for key, fmp_name in _FMP_PMI_MAP.items():
        meta = _SERIES[key]
        try:
            resp = requests.get(
                _FMP_BASE_URL,
                params={"name": fmp_name, "apikey": _FMP_KEY},
                timeout=12,
            )
            resp.raise_for_status()
            raw = resp.json()
            if not raw or not isinstance(raw, list):
                logger.warning("FMP PMI: empty response for %s", fmp_name)
                continue

            # FMP returns oldest-first; sort newest-first
            obs = sorted(
                [
                    {"date": o["date"], "value": float(o["value"])}
                    for o in raw
                    if o.get("value") is not None
                ],
                key=lambda x: x["date"],
                reverse=True,
            )[:48]

            if not obs:
                continue

            current  = obs[0]["value"]
            year_idx = min(11, len(obs) - 1)
            prev     = obs[year_idx]["value"]

            results[key] = {
                "name":         meta["name"],
                "unit":         meta["unit"],
                "color":        meta["color"],
                "desc":         meta.get("desc", ""),
                "pmi":          True,
                "is_mock":      False,
                "current":      round(current, 1),
                "prev":         round(prev, 1),
                "change_1y":    round(current - prev, 1),
                "observations": obs,
            }
            logger.info("FMP PMI fetched %s (%s): current=%.1f", key, fmp_name, current)

        except Exception as exc:
            logger.warning("FMP PMI fetch failed for %s (%s): %s", key, fmp_name, exc)

    return results


# ---------------------------------------------------------------------------
# Mock data (realistic 2023-2025 values, newest-first)
# ---------------------------------------------------------------------------

def _mock_macro() -> dict[str, dict]:
    """Return realistic static macro data for mock/dev mode."""

    def _series(values_newest_first: list[float], start_month: str) -> list[dict]:
        """Build observation list from a list of values + starting month."""
        obs = []
        from datetime import date  # noqa: PLC0415
        year, month = int(start_month[:4]), int(start_month[5:7])
        for v in values_newest_first:
            obs.append({"date": f"{year:04d}-{month:02d}-01", "value": v})
            month -= 1
            if month == 0:
                month = 12
                year -= 1
        return obs

    # Latest ≈ Jan 2026, 24 monthly points (newest first)
    # Fed cut 3×25bps in Sep/Nov/Dec 2024 → 4.25-4.50%; continued cuts through 2025
    fed_vals = [
        3.83, 3.83, 3.83, 3.83, 3.83, 4.08, 4.08, 4.08,
        4.33, 4.33, 4.33, 4.58, 4.58, 4.58, 4.83, 4.83,
        5.08, 5.33, 5.33, 5.33, 5.33, 5.33, 5.33, 5.33,
    ]
    cpi_vals = [
        2.9, 2.7, 2.6, 2.5, 2.4, 2.5, 2.6, 2.9,
        3.0, 3.2, 3.4, 3.5, 3.7, 3.7, 3.2, 3.1,
        3.0, 2.9, 3.4, 3.7, 4.0, 4.9, 5.5, 6.0,
    ]
    yield_vals = [
        4.57, 4.42, 4.28, 4.17, 4.20, 4.25, 4.38, 4.30,
        4.25, 4.22, 4.20, 4.40, 4.60, 4.70, 4.50, 4.25,
        4.00, 3.95, 4.10, 4.30, 4.50, 4.70, 4.00, 3.80,
    ]
    unemp_vals = [
        4.1, 4.2, 4.2, 4.3, 4.1, 4.0, 3.9, 3.8,
        3.7, 3.7, 3.8, 3.9, 4.0, 4.1, 3.9, 3.8,
        3.7, 3.6, 3.5, 3.4, 3.4, 3.5, 3.6, 3.7,
    ]
    # GDP is quarterly — 8 quarters
    gdp_vals = [3.1, 2.8, 1.6, 3.4, 4.9, 2.1, 2.0, 2.6]
    gdp_dates = [
        "2025-10-01", "2025-07-01", "2025-04-01", "2025-01-01",
        "2024-10-01", "2024-07-01", "2024-04-01", "2024-01-01",
    ]
    gdp_obs = [{"date": d, "value": v} for d, v in zip(gdp_dates, gdp_vals)]

    def _build(key, vals, obs, start="2026-01-01"):
        m = _SERIES[key]
        use_obs = obs if obs else _series(vals, start[:7])
        current = use_obs[0]["value"]
        prev    = use_obs[min(11, len(use_obs) - 1)]["value"]
        return {
            "name":         m["name"],
            "unit":         m["unit"],
            "color":        m["color"],
            "desc":         m.get("desc", ""),
            "pmi":          m.get("pmi", False),
            "current":      current,
            "prev":         prev,
            "change_1y":    round(current - prev, 2),
            "observations": use_obs,
        }

    # PMI — manufacturing mostly below 50 (contraction) in 2023-24, services resilient
    mfg_pmi_vals = [
        49.3, 49.1, 48.7, 47.8, 47.6, 47.8, 48.5, 49.2,
        46.7, 46.4, 47.1, 48.4, 49.0, 49.2, 50.3, 51.2,
        52.8, 53.0, 52.6, 51.4, 50.9, 48.3, 47.4, 46.9,
    ]
    svc_pmi_vals = [
        54.1, 53.5, 52.8, 53.3, 54.9, 55.1, 53.8, 52.7,
        51.4, 51.6, 52.0, 53.4, 54.5, 54.9, 56.7, 57.2,
        55.3, 52.7, 52.9, 53.4, 54.5, 55.1, 56.4, 55.2,
    ]
    return {
        "fed_rate":    _build("fed_rate",    fed_vals,     []),
        "cpi":         _build("cpi",         cpi_vals,     []),
        "yield_10y":   _build("yield_10y",   yield_vals,   []),
        "unemployment":_build("unemployment",unemp_vals,   []),
        "gdp_growth":  _build("gdp_growth",  [],           gdp_obs),
        "mfg_pmi":     _build("mfg_pmi",     mfg_pmi_vals, []),
        "svc_pmi":     _build("svc_pmi",     svc_pmi_vals, []),
    }
