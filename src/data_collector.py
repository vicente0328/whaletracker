"""
data_collector.py
-----------------
Fetches institutional filings and detects signals across four SEC form types:

  13F-HR  (quarterly, 45-day lag)
    - NEW_ENTRY:          Position absent from prior quarter → +3
    - AGGRESSIVE_BUY:     Shares +>20% QoQ → +4
    - HIGH_CONCENTRATION: Position >5% of whale portfolio → +2

  SC 13D/G (5-10 day lag after crossing 5% ownership)
    - ACTIVIST_STAKE:       13D — filer intends to influence management → +5
    - LARGE_PASSIVE_STAKE:  13G — passive >5% owner → +2

  Form 4 (2-day lag, insiders: officers, directors, >10% holders)
    - INSIDER_BUY:   Open-market purchase by insider → +3
    - INSIDER_SELL:  Open-market sale by insider → -2

  N-PORT (monthly, 60-day lag, registered funds / ETFs)
    - FUND_ACCUMULATION: Shares increased vs prior month → +2
    - FUND_LIQUIDATION:  Shares decreased vs prior month → -1

Data sources:
  - Financial Modeling Prep (FMP) API  → set FMP_API_KEY in .env  (13F live)
  - SEC EDGAR full-text search API     → public, no key required   (13D/G, Form 4, N-PORT live stubs)
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

# Tracked Whales: {display_name: CIK}
TRACKED_WHALES: dict[str, str] = {
    "Berkshire Hathaway":    "0001067983",
    "Bridgewater Associates":"0001350694",
    "Appaloosa Management":  "0001056831",
    "Pershing Square":       "0001336528",
    "Tiger Global":          "0001167483",
}

# Tracked N-PORT funds: {display_name: CIK}
TRACKED_NPORT_FUNDS: dict[str, str] = {
    "Vanguard 500 Index":  "0000102909",
    "Fidelity Contrafund": "0000021624",
    "ARK Innovation ETF":  "0001697748",
}

# Signal thresholds
AGGRESSIVE_BUY_THRESHOLD    = 0.20  # 20% QoQ share increase
HIGH_CONCENTRATION_THRESHOLD = 0.05  # 5% of portfolio value
FUND_ACCUMULATION_THRESHOLD  = 0.05  # 5% month-over-month increase
FUND_LIQUIDATION_THRESHOLD   = -0.05  # 5% month-over-month decrease


# ---------------------------------------------------------------------------
# Public API — 13F
# ---------------------------------------------------------------------------

def fetch_all_whale_filings() -> dict[str, list[dict[str, Any]]]:
    """Return the latest 13F holdings for every tracked Whale.

    Returns:
        {whale_name: [holding_dict, ...]}
        Each holding dict: ticker, company, shares, value_usd, portfolio_pct, signal
    """
    if DATA_MODE == "mock":
        return _load_mock_data()

    results: dict[str, list[dict]] = {}
    for name, cik in TRACKED_WHALES.items():
        try:
            results[name] = _fetch_fmp_13f(cik)
        except Exception as exc:
            logger.error("Failed to fetch 13F for %s: %s", name, exc)
            results[name] = []
    return results


def detect_signals(
    current: list[dict[str, Any]],
    previous: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Annotate each holding in `current` with a 13F Whale signal."""
    prev_map: dict[str, dict] = {h["ticker"]: h for h in previous}
    return [{**h, "signal": _classify_signal(h, prev_map.get(h["ticker"]))}
            for h in current]


# ---------------------------------------------------------------------------
# Public API — SC 13D/G
# ---------------------------------------------------------------------------

def fetch_13dg_filings() -> dict[str, dict[str, Any]]:
    """Return latest SC 13D/G activist/passive stake filings per ticker.

    Returns:
        {ticker: filing_dict}
        filing_dict keys: form_type, filer, shares, pct_outstanding,
                          filed_date, signal
    """
    if DATA_MODE == "mock":
        return _mock_13dg()
    return _fetch_edgar_13dg()


# ---------------------------------------------------------------------------
# Public API — Form 4
# ---------------------------------------------------------------------------

def fetch_form4_filings(tickers: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Return recent Form 4 insider transactions per ticker.

    Args:
        tickers: Optional filter list. None = all tracked tickers.

    Returns:
        {ticker: [transaction_dict, ...]}
        transaction_dict keys: insider, role, transaction_type, shares,
                               price, value_usd, filed_date, signal
    """
    if DATA_MODE == "mock":
        return _mock_form4()
    return _fetch_edgar_form4(tickers)


# ---------------------------------------------------------------------------
# Public API — N-PORT
# ---------------------------------------------------------------------------

def fetch_nport_filings() -> dict[str, list[dict[str, Any]]]:
    """Return latest N-PORT monthly holdings for tracked funds.

    Returns:
        {fund_name: [holding_dict, ...]}
        holding_dict keys: ticker, company, shares, value_usd, portfolio_pct,
                           change_pct, signal
    """
    if DATA_MODE == "mock":
        return _mock_nport()
    return _fetch_edgar_nport()


# ---------------------------------------------------------------------------
# 13F internals
# ---------------------------------------------------------------------------

def _classify_signal(current: dict[str, Any], previous: dict[str, Any] | None) -> str:
    if previous is None:
        return "NEW_ENTRY"
    prev_shares  = previous.get("shares", 0)
    curr_shares  = current.get("shares", 0)
    portfolio_pct = current.get("portfolio_pct", 0.0)
    if prev_shares > 0 and (curr_shares - prev_shares) / prev_shares > AGGRESSIVE_BUY_THRESHOLD:
        return "AGGRESSIVE_BUY"
    if portfolio_pct > HIGH_CONCENTRATION_THRESHOLD:
        return "HIGH_CONCENTRATION"
    return "HOLD"


def _fetch_fmp_13f(cik: str) -> list[dict[str, Any]]:
    if not FMP_API_KEY:
        raise EnvironmentError("FMP_API_KEY not set.")
    url = (f"https://financialmodelingprep.com/api/v3/form-thirteen-f/{cik}"
           f"?apikey={FMP_API_KEY}")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return _normalize_fmp_holdings(resp.json())


def _normalize_fmp_holdings(raw: list[dict]) -> list[dict[str, Any]]:
    return [{
        "ticker":        item.get("symbol", ""),
        "company":       item.get("nameOfIssuer", ""),
        "shares":        item.get("sharesNumber", 0),
        "value_usd":     item.get("marketValue", 0),
        "portfolio_pct": item.get("portfolioPercent", 0.0) / 100,
    } for item in raw]


def _load_mock_data() -> dict[str, list[dict[str, Any]]]:
    mock_path = CACHE_DIR / "mock_filings.json"
    if mock_path.exists():
        return json.loads(mock_path.read_text())

    return {
        "Berkshire Hathaway": [
            {"ticker": "AAPL", "company": "Apple Inc.",           "shares": 905_560_000,   "value_usd": 174_300_000_000, "portfolio_pct": 0.48, "signal": "HIGH_CONCENTRATION"},
            {"ticker": "BAC",  "company": "Bank of America",      "shares": 1_032_000_000, "value_usd":  29_500_000_000, "portfolio_pct": 0.09, "signal": "HOLD"},
            {"ticker": "CVX",  "company": "Chevron Corp.",        "shares":   123_120_000, "value_usd":  18_800_000_000, "portfolio_pct": 0.06, "signal": "HIGH_CONCENTRATION"},
            {"ticker": "OXY",  "company": "Occidental Petroleum", "shares":   248_018_128, "value_usd":  14_800_000_000, "portfolio_pct": 0.04, "signal": "AGGRESSIVE_BUY"},
        ],
        "Appaloosa Management": [
            {"ticker": "META",  "company": "Meta Platforms",  "shares":   850_000, "value_usd": 430_000_000, "portfolio_pct": 0.07, "signal": "NEW_ENTRY"},
            {"ticker": "GOOGL", "company": "Alphabet Inc.",   "shares":   500_000, "value_usd": 680_000_000, "portfolio_pct": 0.09, "signal": "AGGRESSIVE_BUY"},
            {"ticker": "AMZN",  "company": "Amazon.com Inc.", "shares": 1_200_000, "value_usd": 204_000_000, "portfolio_pct": 0.03, "signal": "HOLD"},
        ],
    }


# ---------------------------------------------------------------------------
# 13D/G internals
# ---------------------------------------------------------------------------

def _mock_13dg() -> dict[str, dict[str, Any]]:
    return {
        "OXY": {
            "form_type":       "SC 13D",
            "filer":           "Berkshire Hathaway",
            "shares":          248_018_128,
            "pct_outstanding": 0.264,
            "filed_date":      "2024-10-02",
            "signal":          "ACTIVIST_STAKE",
        },
        "META": {
            "form_type":       "SC 13G",
            "filer":           "Appaloosa Management",
            "shares":          850_000,
            "pct_outstanding": 0.033,
            "filed_date":      "2024-11-14",
            "signal":          "LARGE_PASSIVE_STAKE",
        },
        "GOOGL": {
            "form_type":       "SC 13G",
            "filer":           "Tiger Global",
            "shares":          2_100_000,
            "pct_outstanding": 0.017,
            "filed_date":      "2024-11-08",
            "signal":          "LARGE_PASSIVE_STAKE",
        },
    }


def _fetch_edgar_13dg() -> dict[str, dict[str, Any]]:
    """Live: parse SC 13D/G from SEC EDGAR full-text search API.

    Endpoint: https://efts.sec.gov/LATEST/search-index?forms=SC+13D,SC+13G
    (No API key required — public endpoint.)
    """
    logger.warning("Live 13D/G fetch not yet implemented; returning empty dict.")
    return {}


# ---------------------------------------------------------------------------
# Form 4 internals
# ---------------------------------------------------------------------------

def _mock_form4() -> dict[str, list[dict[str, Any]]]:
    return {
        "AAPL": [
            {"insider": "Tim Cook",         "role": "CEO", "transaction_type": "Purchase",
             "shares":  60_000,  "price": 189.30, "value_usd":  11_358_000,
             "filed_date": "2024-11-20", "signal": "INSIDER_BUY"},
        ],
        "OXY": [
            {"insider": "Vicki Hollub",     "role": "CEO", "transaction_type": "Purchase",
             "shares":  20_000,  "price":  59.10, "value_usd":   1_182_000,
             "filed_date": "2024-11-18", "signal": "INSIDER_BUY"},
        ],
        "BAC": [
            {"insider": "Brian Moynihan",   "role": "CEO", "transaction_type": "Sale",
             "shares": 150_000,  "price":  38.50, "value_usd":   5_775_000,
             "filed_date": "2024-11-22", "signal": "INSIDER_SELL"},
        ],
        "GOOGL": [
            {"insider": "Sundar Pichai",    "role": "CEO", "transaction_type": "Purchase",
             "shares":  25_000,  "price": 174.20, "value_usd":   4_355_000,
             "filed_date": "2024-11-19", "signal": "INSIDER_BUY"},
        ],
    }


def _fetch_edgar_form4(tickers: list[str] | None) -> dict[str, list[dict[str, Any]]]:
    """Live: parse Form 4 from SEC EDGAR.

    Endpoint: https://data.sec.gov/submissions/CIK{cik}.json
    then filter form4 filings by insider CIK.
    (No API key required — public endpoint.)
    """
    logger.warning("Live Form 4 fetch not yet implemented; returning empty dict.")
    return {}


# ---------------------------------------------------------------------------
# N-PORT internals
# ---------------------------------------------------------------------------

def _mock_nport() -> dict[str, list[dict[str, Any]]]:
    return {
        "Vanguard 500 Index": [
            {"ticker": "AAPL",  "company": "Apple Inc.",       "shares": 12_500_000, "value_usd": 2_367_500_000, "portfolio_pct": 0.071, "change_pct":  0.08,  "signal": "FUND_ACCUMULATION"},
            {"ticker": "MSFT",  "company": "Microsoft Corp.",  "shares":  9_800_000, "value_usd": 3_704_400_000, "portfolio_pct": 0.069, "change_pct":  0.05,  "signal": "FUND_ACCUMULATION"},
            {"ticker": "NVDA",  "company": "NVIDIA Corp.",     "shares":  8_200_000, "value_usd": 3_690_000_000, "portfolio_pct": 0.061, "change_pct":  0.22,  "signal": "FUND_ACCUMULATION"},
            {"ticker": "AMZN",  "company": "Amazon.com Inc.",  "shares":  6_100_000, "value_usd": 1_037_000_000, "portfolio_pct": 0.037, "change_pct": -0.04,  "signal": "FUND_LIQUIDATION"},
        ],
        "ARK Innovation ETF": [
            {"ticker": "TSLA",  "company": "Tesla Inc.",       "shares":  3_200_000, "value_usd":   819_200_000, "portfolio_pct": 0.112, "change_pct":  0.18,  "signal": "FUND_ACCUMULATION"},
            {"ticker": "COIN",  "company": "Coinbase Global",  "shares":  2_100_000, "value_usd":   504_000_000, "portfolio_pct": 0.073, "change_pct":  0.31,  "signal": "FUND_ACCUMULATION"},
            {"ticker": "GOOGL", "company": "Alphabet Inc.",    "shares":    850_000, "value_usd":   148_100_000, "portfolio_pct": 0.021, "change_pct": -0.12,  "signal": "FUND_LIQUIDATION"},
        ],
    }


def _fetch_edgar_nport() -> dict[str, list[dict[str, Any]]]:
    """Live: parse N-PORT from SEC EDGAR structured data API.

    Endpoint: https://data.sec.gov/api/xbrl/frames/...  or
              https://efts.sec.gov/LATEST/search-index?forms=NPORT-P
    (No API key required — public endpoint.)
    """
    logger.warning("Live N-PORT fetch not yet implemented; returning empty dict.")
    return {}
