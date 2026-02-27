"""
data_collector.py
-----------------
Fetches 13F-HR institutional filings and detects three Whale signals:
  - NEW_ENTRY:       Position not present in the prior quarter's filing.
  - AGGRESSIVE_BUY:  Share count increased >20% quarter-over-quarter.
  - HIGH_CONCENTRATION: Position is >5% of the Whale's total portfolio value.

Data sources:
  - Financial Modeling Prep (FMP) API  → set FMP_API_KEY in .env
  - SEC EDGAR EDGAR full-text search   → public, no key required
  - Mock data                          → set DATA_MODE=mock in .env
"""

import os
import json
import logging
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FMP_API_KEY: str = os.getenv("FMP_API_KEY", "")
DATA_MODE: str = os.getenv("DATA_MODE", "mock")  # "mock" | "live"
CACHE_DIR: Path = Path(__file__).parent.parent / "data"

# Tracked Whales: {display_name: CIK or FMP identifier}
TRACKED_WHALES: dict[str, str] = {
    "Berkshire Hathaway": "0001067983",
    "Bridgewater Associates": "0001350694",
    "Appaloosa Management": "0001056831",
    "Pershing Square": "0001336528",
    "Tiger Global": "0001167483",
}

# Signal thresholds
AGGRESSIVE_BUY_THRESHOLD = 0.20   # 20% QoQ share increase
HIGH_CONCENTRATION_THRESHOLD = 0.05  # 5% of portfolio value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_all_whale_filings() -> dict[str, list[dict[str, Any]]]:
    """Return the latest 13F holdings for every tracked Whale.

    Returns:
        {whale_name: [holding_dict, ...]}
        Each holding dict contains at minimum:
          ticker, company, shares, value_usd, portfolio_pct, signal
    """
    if DATA_MODE == "mock":
        return _load_mock_data()

    results: dict[str, list[dict]] = {}
    for name, cik in TRACKED_WHALES.items():
        try:
            results[name] = _fetch_fmp_13f(cik)
        except Exception as exc:
            logger.error("Failed to fetch filings for %s: %s", name, exc)
            results[name] = []
    return results


def detect_signals(
    current: list[dict[str, Any]],
    previous: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Annotate each holding in `current` with a Whale signal.

    Args:
        current:  List of current-quarter holdings.
        previous: List of prior-quarter holdings (same Whale).

    Returns:
        `current` with an added `signal` key on each holding.
    """
    prev_map: dict[str, dict] = {h["ticker"]: h for h in previous}
    annotated = []
    for holding in current:
        ticker = holding["ticker"]
        signal = _classify_signal(holding, prev_map.get(ticker))
        annotated.append({**holding, "signal": signal})
    return annotated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_signal(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> str:
    """Return the highest-priority signal for a single holding."""
    if previous is None:
        return "NEW_ENTRY"

    prev_shares = previous.get("shares", 0)
    curr_shares = current.get("shares", 0)
    portfolio_pct = current.get("portfolio_pct", 0.0)

    if prev_shares > 0 and (curr_shares - prev_shares) / prev_shares > AGGRESSIVE_BUY_THRESHOLD:
        return "AGGRESSIVE_BUY"

    if portfolio_pct > HIGH_CONCENTRATION_THRESHOLD:
        return "HIGH_CONCENTRATION"

    return "HOLD"


def _fetch_fmp_13f(cik: str) -> list[dict[str, Any]]:
    """Fetch the latest 13F filing via Financial Modeling Prep."""
    if not FMP_API_KEY:
        raise EnvironmentError("FMP_API_KEY is not set. Add it to your .env file.")

    url = (
        f"https://financialmodelingprep.com/api/v3/form-thirteen-f/{cik}"
        f"?apikey={FMP_API_KEY}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    raw: list[dict] = resp.json()
    return _normalize_fmp_holdings(raw)


def _normalize_fmp_holdings(raw: list[dict]) -> list[dict[str, Any]]:
    """Map FMP response fields to our internal schema."""
    normalized = []
    for item in raw:
        normalized.append({
            "ticker": item.get("symbol", ""),
            "company": item.get("nameOfIssuer", ""),
            "shares": item.get("sharesNumber", 0),
            "value_usd": item.get("marketValue", 0),
            "portfolio_pct": item.get("portfolioPercent", 0.0) / 100,
        })
    return normalized


def _load_mock_data() -> dict[str, list[dict[str, Any]]]:
    """Return hard-coded sample data for development / testing."""
    mock_path = CACHE_DIR / "mock_filings.json"
    if mock_path.exists():
        return json.loads(mock_path.read_text())

    # Inline fallback if the JSON file is absent
    return {
        "Berkshire Hathaway": [
            {"ticker": "AAPL",  "company": "Apple Inc.",       "shares": 905_560_000, "value_usd": 174_300_000_000, "portfolio_pct": 0.48, "signal": "HIGH_CONCENTRATION"},
            {"ticker": "BAC",   "company": "Bank of America",  "shares": 1_032_000_000, "value_usd": 29_500_000_000, "portfolio_pct": 0.09, "signal": "HOLD"},
            {"ticker": "CVX",   "company": "Chevron Corp.",    "shares": 123_120_000, "value_usd": 18_800_000_000, "portfolio_pct": 0.06, "signal": "HIGH_CONCENTRATION"},
            {"ticker": "OXY",   "company": "Occidental Petroleum", "shares": 248_018_128, "value_usd": 14_800_000_000, "portfolio_pct": 0.04, "signal": "AGGRESSIVE_BUY"},
        ],
        "Appaloosa Management": [
            {"ticker": "META",  "company": "Meta Platforms",   "shares": 850_000,  "value_usd": 430_000_000, "portfolio_pct": 0.07, "signal": "NEW_ENTRY"},
            {"ticker": "GOOGL", "company": "Alphabet Inc.",    "shares": 500_000,  "value_usd": 680_000_000, "portfolio_pct": 0.09, "signal": "AGGRESSIVE_BUY"},
            {"ticker": "AMZN",  "company": "Amazon.com Inc.",  "shares": 1_200_000, "value_usd": 204_000_000, "portfolio_pct": 0.03, "signal": "HOLD"},
        ],
    }
