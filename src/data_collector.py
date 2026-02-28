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
  - Financial Modeling Prep (FMP) API  → 13F live + Form 4 live  (FMP_API_KEY in .env)
  - SEC EDGAR public APIs              → 13D/G live + N-PORT live (no key required)
  - Mock data                          → DATA_MODE=mock in .env
"""

import os
import re
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
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
DATA_MODE: str  = os.getenv("DATA_MODE", "mock")   # "mock" | "live"
CACHE_DIR: Path = Path(__file__).parent.parent / "data"

# Tracked Whales: {display_name: CIK (zero-padded 10 digits)}
TRACKED_WHALES: dict[str, str] = {
    "Berkshire Hathaway":    "0001067983",
    "Bridgewater Associates":"0001350694",
    "Appaloosa Management":  "0001056831",
    "Pershing Square":       "0001336528",
    "Tiger Global":          "0001167483",
}

# Tracked N-PORT funds: {display_name: CIK}
# CIKs verified via data.sec.gov/submissions/CIK{cik}.json
TRACKED_NPORT_FUNDS: dict[str, str] = {
    "Vanguard 500 Index": "0000102909",
    "ARK Innovation ETF": "0001697748",
    "SPDR S&P 500 ETF":   "0000884394",   # SPY trust (State Street)
}

# ---------------------------------------------------------------------------
# Hardcoded company-name → ticker map for 13F XML parsing
# (13F documents carry CUSIP + company name, not ticker symbols)
# ---------------------------------------------------------------------------
_13F_NAME_MAP: dict[str, str] = {
    # Technology
    "apple inc": "AAPL",                "microsoft corp": "MSFT",
    "microsoft corporation": "MSFT",    "amazon.com inc": "AMZN",
    "amazon com inc": "AMZN",           "alphabet inc": "GOOGL",
    "alphabet inc cl a": "GOOGL",       "alphabet inc cl c": "GOOG",
    "meta platforms inc": "META",       "meta platforms": "META",
    "tesla inc": "TSLA",               "nvidia corp": "NVDA",
    "nvidia corporation": "NVDA",       "broadcom inc": "AVGO",
    "oracle corp": "ORCL",             "salesforce inc": "CRM",
    "adobe inc": "ADBE",               "qualcomm inc": "QCOM",
    "advanced micro devices": "AMD",    "intel corp": "INTC",
    "applied materials inc": "AMAT",    "lam research corp": "LRCX",
    "servicenow inc": "NOW",           "snowflake inc": "SNOW",
    "palantir technologies": "PLTR",   "uber technologies": "UBER",
    "airbnb inc": "ABNB",             "datadog inc": "DDOG",
    # Finance
    "berkshire hathaway inc": "BRK-B", "berkshire hathaway": "BRK-B",
    "jpmorgan chase & co": "JPM",      "jpmorgan chase": "JPM",
    "bank of america corp": "BAC",     "bank of america": "BAC",
    "wells fargo & co": "WFC",         "wells fargo": "WFC",
    "citigroup inc": "C",             "goldman sachs group inc": "GS",
    "morgan stanley": "MS",           "american express co": "AXP",
    "visa inc": "V",                  "mastercard inc": "MA",
    "blackrock inc": "BLK",           "charles schwab corp": "SCHW",
    "s&p global inc": "SPGI",
    # Energy
    "exxon mobil corp": "XOM",         "exxon mobil": "XOM",
    "chevron corp": "CVX",             "occidental petroleum corp": "OXY",
    "occidental petroleum": "OXY",     "conocophillips": "COP",
    "pioneer natural resources": "PXD","marathon oil corp": "MRO",
    "hess corp": "HES",               "coterra energy inc": "CTRA",
    # Healthcare
    "unitedhealth group inc": "UNH",   "unitedhealth": "UNH",
    "johnson & johnson": "JNJ",        "abbvie inc": "ABBV",
    "eli lilly & co": "LLY",          "eli lilly and co": "LLY",
    "pfizer inc": "PFE",              "merck & co inc": "MRK",
    "thermo fisher scientific": "TMO", "abbott laboratories": "ABT",
    "danaher corp": "DHR",            "intuitive surgical inc": "ISRG",
    "cigna group": "CI",              "elevance health inc": "ELV",
    # Consumer
    "amazon com": "AMZN",             "home depot inc": "HD",
    "walmart inc": "WMT",             "costco wholesale corp": "COST",
    "nike inc": "NKE",                "mcdonalds corp": "MCD",
    "starbucks corp": "SBUX",         "procter & gamble co": "PG",
    "coca-cola co": "KO",             "coca cola co": "KO",
    "pepsico inc": "PEP",             "colgate-palmolive co": "CL",
    "target corp": "TGT",             "lowes companies inc": "LOW",
    # Industrial / Other
    "caterpillar inc": "CAT",          "deere & co": "DE",
    "union pacific corp": "UNP",       "united parcel service": "UPS",
    "boeing co": "BA",                "lockheed martin corp": "LMT",
    "3m co": "MMM",                   "honeywell international": "HON",
    "emerson electric co": "EMR",      "ge healthcare": "GEHC",
    "s&p 500 etf tr": "SPY",          "spdr s&p 500 etf tr": "SPY",
}

# Signal thresholds
AGGRESSIVE_BUY_THRESHOLD     = 0.20    # 20% QoQ share increase
HIGH_CONCENTRATION_THRESHOLD = 0.05    # 5% of portfolio value
FUND_ACCUMULATION_THRESHOLD  = 0.05    # +5% MoM share increase
FUND_LIQUIDATION_THRESHOLD   = -0.05   # -5% MoM share decrease

# SEC requires a descriptive User-Agent for all EDGAR API calls
_EDGAR_HEADERS = {
    "User-Agent": "WhaleTracker research@whaletracker.ai",
    "Accept-Encoding": "gzip, deflate",
}


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
            logger.error("13F fetch failed for %s: %s", name, exc)
            results[name] = []
    return results


def detect_signals(
    current:  list[dict[str, Any]],
    previous: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Annotate each holding in `current` with a 13F Whale signal."""
    prev_map = {h["ticker"]: h for h in previous}
    return [{**h, "signal": _classify_signal(h, prev_map.get(h["ticker"]))}
            for h in current]


# ---------------------------------------------------------------------------
# Public API — SC 13D/G
# ---------------------------------------------------------------------------

def fetch_13dg_filings() -> dict[str, dict[str, Any]]:
    """Return latest SC 13D/G activist/passive stake filings per ticker.

    Returns:
        {ticker: {form_type, filer, shares, pct_outstanding, filed_date, signal}}
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
        tickers: Tickers to query (from live 13F holdings). None → mock.

    Returns:
        {ticker: [{insider, role, transaction_type, shares, price,
                   value_usd, filed_date, signal}]}
    """
    if DATA_MODE == "mock":
        return _mock_form4()
    return _fetch_fmp_form4(tickers or [])


# ---------------------------------------------------------------------------
# Public API — N-PORT
# ---------------------------------------------------------------------------

def fetch_nport_filings() -> dict[str, list[dict[str, Any]]]:
    """Return latest N-PORT monthly holdings for tracked funds.

    Returns:
        {fund_name: [{ticker, company, shares, value_usd,
                      portfolio_pct, change_pct, signal}]}
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
    prev_shares   = previous.get("shares", 0)
    curr_shares   = current.get("shares", 0)
    portfolio_pct = current.get("portfolio_pct", 0.0)
    if prev_shares > 0 and (curr_shares - prev_shares) / prev_shares > AGGRESSIVE_BUY_THRESHOLD:
        return "AGGRESSIVE_BUY"
    if portfolio_pct > HIGH_CONCENTRATION_THRESHOLD:
        return "HIGH_CONCENTRATION"
    return "HOLD"


def _fetch_fmp_13f(cik: str) -> list[dict[str, Any]]:
    """Try FMP first; fall back to EDGAR if the plan doesn't include 13F."""
    if FMP_API_KEY:
        url = (f"https://financialmodelingprep.com/api/v3/form-thirteen-f/{cik}"
               f"?apikey={FMP_API_KEY}")
        resp = requests.get(url, timeout=15)
        if resp.status_code in (401, 403):
            logger.info("FMP plan does not include 13F (HTTP %s) — falling back to EDGAR.", resp.status_code)
        else:
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                return _normalize_fmp_holdings(data)

    # EDGAR fallback (free, no key required)
    return _fetch_edgar_13f(cik)


def _fetch_edgar_13f(cik: str) -> list[dict[str, Any]]:
    """Fetch 13F-HR holdings directly from SEC EDGAR."""
    resp = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=_EDGAR_HEADERS, timeout=20,
    )
    resp.raise_for_status()
    data   = resp.json()
    recent = data.get("filings", {}).get("recent", {})

    forms    = recent.get("form", [])
    acc_nums = recent.get("accessionNumber", [])

    for i, form_type in enumerate(forms):
        if form_type != "13F-HR":
            continue
        acc = acc_nums[i] if i < len(acc_nums) else ""
        if not acc:
            continue

        cik_int   = int(cik)
        holdings_url = _find_13f_holdings_doc(cik_int, acc)
        if holdings_url:
            result = _parse_13f_xml(holdings_url)
            if result:
                return result
        break   # Only try the most recent 13F-HR

    return []


def _find_13f_holdings_doc(cik_int: int, acc: str) -> str | None:
    """Return the URL of the 13F holdings XML within a given filing."""
    acc_clean = acc.replace("-", "")
    index_url = (f"https://www.sec.gov/Archives/edgar/data/"
                 f"{cik_int}/{acc_clean}/{acc}-index.htm")
    try:
        resp = requests.get(index_url, headers=_EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        # Grab every .xml link from the index page
        xml_links = re.findall(
            r'href="(/Archives/edgar/data/\d+/\d+/[^"]+\.xml)"',
            resp.text, re.IGNORECASE,
        )
        # Prefer files with "table" / "info" / "holdings" in the name
        for link in xml_links:
            fname = link.split("/")[-1].lower()
            if any(kw in fname for kw in ("table", "info", "holdings")):
                return f"https://www.sec.gov{link}"
        if xml_links:
            return f"https://www.sec.gov{xml_links[0]}"
    except Exception as exc:
        logger.debug("13F index error for %s: %s", acc, exc)
    return None


def _parse_13f_xml(xml_url: str) -> list[dict[str, Any]]:
    """Download and parse a 13F-HR holdings XML into our internal schema."""
    resp = requests.get(xml_url, headers=_EDGAR_HEADERS, timeout=30)
    resp.raise_for_status()
    content = resp.text

    # Strip XML namespaces for simple tag access
    content = re.sub(r'\s+xmlns(?::[^=]+)?="[^"]*"', "", content)
    content = re.sub(r"<([a-zA-Z]+:)",  "<",  content)
    content = re.sub(r"</([a-zA-Z]+:)", "</", content)

    root        = ET.fromstring(content)
    raw_entries = []
    total_value = 0

    for info in root.iter("infoTable"):
        name_el   = info.find("nameOfIssuer")
        value_el  = info.find("value")           # USD thousands
        shares_el = info.find(".//sshPrnamt")

        if name_el is None or value_el is None:
            continue

        raw_name = (name_el.text or "").strip()
        ticker   = _resolve_ticker(raw_name)
        if not ticker:
            continue

        try:
            value_usd = int(value_el.text or 0) * 1_000
            shares    = int(shares_el.text or 0) if shares_el is not None else 0
        except (ValueError, TypeError):
            continue

        total_value += value_usd
        raw_entries.append({
            "ticker":    ticker,
            "company":   raw_name.title(),
            "shares":    shares,
            "value_usd": value_usd,
        })

    if not raw_entries:
        return []

    # Sort by value, compute portfolio_pct
    raw_entries.sort(key=lambda x: x["value_usd"], reverse=True)
    return [
        {**e, "portfolio_pct": e["value_usd"] / total_value if total_value else 0}
        for e in raw_entries
    ]


def _resolve_ticker(company_name: str) -> str:
    """Map a 13F issuer name to a ticker symbol."""
    key = company_name.lower().strip()

    # 1. Direct match
    if key in _13F_NAME_MAP:
        return _13F_NAME_MAP[key]

    # 2. Strip common suffixes and retry
    for suffix in (" inc", " corp", " co", " ltd", " llc", " plc", " & co"):
        stripped = key.removesuffix(suffix).strip()
        if stripped in _13F_NAME_MAP:
            return _13F_NAME_MAP[stripped]

    # 3. Substring match (first matching key that appears inside company_name)
    for map_key, ticker in _13F_NAME_MAP.items():
        if map_key in key:
            return ticker

    return ""   # Unknown — skip this holding


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
# SC 13D/G — live via EDGAR submissions API
# ---------------------------------------------------------------------------

def _fetch_edgar_13dg() -> dict[str, dict[str, Any]]:
    """Scan each whale's EDGAR submissions for SC 13D/G filings."""
    results: dict[str, dict] = {}
    for whale_name, cik in TRACKED_WHALES.items():
        try:
            filing = _latest_13dg_for_cik(cik, whale_name)
            if filing:
                ticker = filing.pop("ticker", "")
                if ticker and ticker not in results:
                    results[ticker] = filing
        except Exception as exc:
            logger.warning("13D/G skip %s: %s", whale_name, exc)
    return results


def _latest_13dg_for_cik(cik: str, whale_name: str) -> dict[str, Any] | None:
    """Return the most recent SC 13D or 13G filing for a given whale CIK."""
    resp = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=_EDGAR_HEADERS, timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    recent       = data.get("filings", {}).get("recent", {})
    forms        = recent.get("form", [])
    acc_nums     = recent.get("accessionNumber", [])
    dates        = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form_type in enumerate(forms):
        base_form = form_type.replace("/A", "").strip()
        if base_form not in ("SC 13D", "SC 13G"):
            continue

        acc       = acc_nums[i]     if i < len(acc_nums)     else ""
        date      = dates[i]        if i < len(dates)        else ""
        prime_doc = primary_docs[i] if i < len(primary_docs) else ""
        if not acc or not prime_doc:
            continue

        cik_int   = int(cik)
        acc_clean = acc.replace("-", "")
        doc_url   = (f"https://www.sec.gov/Archives/edgar/data/"
                     f"{cik_int}/{acc_clean}/{prime_doc}")

        parsed = _parse_13dg_document(doc_url, form_type, date, whale_name)
        if parsed:
            return parsed

        # Only attempt the most recent 13D/G per whale
        break

    return None


def _parse_13dg_document(
    doc_url: str, form_type: str, date: str, filer: str
) -> dict[str, Any] | None:
    """Fetch an SC 13D/G document and extract subject ticker + ownership %."""
    try:
        resp = requests.get(doc_url, headers=_EDGAR_HEADERS, timeout=25)
        resp.raise_for_status()
        text = resp.text

        # ── Ticker ──────────────────────────────────────────────────────────
        # Cover page: "Issuer's Ticker Symbol" / "Ticker Symbol" / "Trading Symbol"
        ticker_m = re.search(
            r"(?:issuer'?s?\s+)?ticker(?:\s+or\s+trading)?\s+symbol[^A-Z\n]{0,30}([A-Z]{1,5})\b",
            text, re.IGNORECASE,
        )
        # Fallback: look for CUSIP row then ticker on same line
        if not ticker_m:
            ticker_m = re.search(
                r"\bCUSIP\b[^\n]{0,60}\n[^\n]{0,30}\(([A-Z]{1,5})\)",
                text, re.IGNORECASE,
            )

        # ── Percent of class ────────────────────────────────────────────────
        # Cover page item 11 (13G) or 13 (13D): "Percent of Class"
        pct_m = re.search(
            r"[Pp]ercent\s+of\s+[Cc]lass[^0-9]{0,60}(\d{1,3}(?:\.\d+)?)\s*%",
            text,
        )

        if not ticker_m or not pct_m:
            logger.debug("Could not extract ticker/pct from %s", doc_url)
            return None

        ticker = ticker_m.group(1).upper()
        pct    = float(pct_m.group(1)) / 100
        signal = "ACTIVIST_STAKE" if "13D" in form_type else "LARGE_PASSIVE_STAKE"

        return {
            "ticker":          ticker,
            "form_type":       form_type,
            "filer":           filer,
            "shares":          0,           # full share count needs additional parsing
            "pct_outstanding": pct,
            "filed_date":      date,
            "signal":          signal,
        }

    except Exception as exc:
        logger.debug("13D/G parse error %s: %s", doc_url, exc)
        return None


# ---------------------------------------------------------------------------
# Form 4 — live via FMP insider-trading endpoint
# ---------------------------------------------------------------------------

def _fetch_fmp_form4(tickers: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Fetch Form 4 insider transactions via FMP for each ticker."""
    if not FMP_API_KEY:
        logger.warning("FMP_API_KEY not set — Form 4 skipped.")
        return {}

    results: dict[str, list[dict]] = {}
    cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    for ticker in tickers:
        try:
            url = (
                f"https://financialmodelingprep.com/api/v4/insider-trading"
                f"?symbol={ticker}&page=0&apikey={FMP_API_KEY}"
            )
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
            if not isinstance(raw, list):
                continue

            txs: list[dict] = []
            for tx in raw:
                filing_date = tx.get("filingDate", "")
                if filing_date < cutoff:
                    break   # Results are date-sorted newest-first

                tx_type = tx.get("transactionType", "")
                if "P-Purchase" in tx_type or tx_type == "P":
                    signal = "INSIDER_BUY"
                elif "S-Sale" in tx_type or tx_type == "S":
                    signal = "INSIDER_SELL"
                else:
                    continue    # Skip options exercises, gifts, etc.

                shares = abs(int(tx.get("securitiesTransacted", 0) or 0))
                price  = float(tx.get("price", 0) or 0)

                # Clean up role string from FMP format "officer: Chief Executive Officer"
                raw_role = tx.get("typeOfOwner", "") or ""
                role = re.sub(r"^(officer|director):\s*", "", raw_role, flags=re.I).strip()
                role = role.title() if role else "Insider"

                txs.append({
                    "insider":          tx.get("reportingName", "Unknown"),
                    "role":             role,
                    "transaction_type": "Purchase" if signal == "INSIDER_BUY" else "Sale",
                    "shares":           shares,
                    "price":            price,
                    "value_usd":        shares * price,
                    "filed_date":       filing_date,
                    "signal":           signal,
                })

                if len(txs) >= 5:   # Cap at 5 per ticker to avoid noise
                    break

            if txs:
                results[ticker] = txs

        except Exception as exc:
            logger.warning("Form 4 skip %s: %s", ticker, exc)

    return results


# ---------------------------------------------------------------------------
# N-PORT — live via EDGAR submissions + XML parsing
# ---------------------------------------------------------------------------

def _fetch_edgar_nport() -> dict[str, list[dict[str, Any]]]:
    """Fetch latest N-PORT-P holdings for each tracked fund."""
    results: dict[str, list[dict]] = {}
    for fund_name, cik in TRACKED_NPORT_FUNDS.items():
        try:
            holdings = _latest_nport_for_cik(cik)
            if holdings:
                results[fund_name] = holdings
        except Exception as exc:
            logger.warning("N-PORT skip %s: %s", fund_name, exc)
    return results


def _latest_nport_for_cik(cik: str) -> list[dict[str, Any]]:
    """Fetch, parse, and annotate the two most recent NPORT-P filings."""
    resp = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=_EDGAR_HEADERS, timeout=20,
    )
    resp.raise_for_status()
    data   = resp.json()
    recent = data.get("filings", {}).get("recent", {})

    forms        = recent.get("form", [])
    acc_nums     = recent.get("accessionNumber", [])
    dates        = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    # Collect two most recent NPORT-P filings for MoM comparison
    nport_meta: list[dict] = []
    for i, form_type in enumerate(forms):
        if form_type == "NPORT-P":
            nport_meta.append({
                "acc":  acc_nums[i]     if i < len(acc_nums)     else "",
                "date": dates[i]        if i < len(dates)        else "",
                "doc":  primary_docs[i] if i < len(primary_docs) else "",
            })
        if len(nport_meta) >= 2:
            break

    if not nport_meta:
        return []

    # Parse current and (optionally) prior month
    current_holdings = _parse_nport_xml(cik, nport_meta[0])
    if not current_holdings:
        return []

    prev_map: dict[str, int] = {}
    if len(nport_meta) >= 2:
        prev = _parse_nport_xml(cik, nport_meta[1])
        prev_map = {h["ticker"]: h["shares"] for h in prev if h.get("ticker")}

    # Annotate with MoM change + signal
    result: list[dict] = []
    for h in current_holdings:
        ticker      = h.get("ticker", "")
        curr_shares = h.get("shares", 0)
        prev_shares = prev_map.get(ticker, 0)

        if prev_shares > 0:
            change_pct = (curr_shares - prev_shares) / prev_shares
        elif curr_shares > 0:
            change_pct = 1.0    # brand-new position
        else:
            change_pct = 0.0

        if change_pct >= FUND_ACCUMULATION_THRESHOLD:
            signal = "FUND_ACCUMULATION"
        elif change_pct <= FUND_LIQUIDATION_THRESHOLD:
            signal = "FUND_LIQUIDATION"
        else:
            signal = "HOLD"

        result.append({**h, "change_pct": change_pct, "signal": signal})

    # Return top 20 by portfolio weight
    result.sort(key=lambda x: x.get("portfolio_pct", 0), reverse=True)
    return result[:20]


def _parse_nport_xml(cik: str, filing: dict) -> list[dict[str, Any]]:
    """Download and parse an NPORT-P XML document into a holdings list."""
    acc = filing.get("acc", "")
    doc = filing.get("doc", "")
    if not acc or not doc:
        return []

    cik_int   = int(cik)
    acc_clean = acc.replace("-", "")
    url = (f"https://www.sec.gov/Archives/edgar/data/"
           f"{cik_int}/{acc_clean}/{doc}")

    try:
        resp = requests.get(url, headers=_EDGAR_HEADERS, timeout=45)
        resp.raise_for_status()
        content = resp.text

        # Strip XML namespaces so ElementTree can use simple tag names
        content = re.sub(r'\s+xmlns(?::[^=]+)?="[^"]*"', "", content)
        content = re.sub(r"<([a-zA-Z]+:)",  "<",  content)
        content = re.sub(r"</([a-zA-Z]+:)", "</", content)

        root = ET.fromstring(content)

        holdings: list[dict] = []
        for sec in root.iter("invstOrSec"):
            # Ticker: lives in <identifiers><ticker> or direct <ticker>
            ticker_el = sec.find(".//ticker") or sec.find("ticker")
            if ticker_el is None or not (ticker_el.text or "").strip():
                continue
            ticker = ticker_el.text.strip().upper()

            name_el    = sec.find("name")
            balance_el = sec.find("balance")
            val_el     = sec.find("valUSD")
            pct_el     = sec.find("pctVal")

            try:
                shares      = int(float(balance_el.text)) if balance_el is not None else 0
                value_usd   = float(val_el.text)          if val_el     is not None else 0.0
                portfolio_pct = float(pct_el.text) / 100  if pct_el     is not None else 0.0
            except (ValueError, TypeError):
                continue

            holdings.append({
                "ticker":        ticker,
                "company":       (name_el.text or "").strip() if name_el is not None else "",
                "shares":        shares,
                "value_usd":     value_usd,
                "portfolio_pct": portfolio_pct,
            })

        return holdings

    except Exception as exc:
        logger.warning("N-PORT XML parse error CIK %s: %s", cik, exc)
        return []


# ---------------------------------------------------------------------------
# Mock data (mock mode fallback)
# ---------------------------------------------------------------------------

def _mock_13dg() -> dict[str, dict[str, Any]]:
    return {
        "OXY": {
            "form_type": "SC 13D", "filer": "Berkshire Hathaway",
            "shares": 248_018_128, "pct_outstanding": 0.264,
            "filed_date": "2024-10-02", "signal": "ACTIVIST_STAKE",
        },
        "META": {
            "form_type": "SC 13G", "filer": "Appaloosa Management",
            "shares": 850_000, "pct_outstanding": 0.033,
            "filed_date": "2024-11-14", "signal": "LARGE_PASSIVE_STAKE",
        },
        "GOOGL": {
            "form_type": "SC 13G", "filer": "Tiger Global",
            "shares": 2_100_000, "pct_outstanding": 0.017,
            "filed_date": "2024-11-08", "signal": "LARGE_PASSIVE_STAKE",
        },
    }


def _mock_form4() -> dict[str, list[dict[str, Any]]]:
    return {
        "AAPL":  [{"insider": "Tim Cook",       "role": "CEO", "transaction_type": "Purchase",
                   "shares":  60_000, "price": 189.30, "value_usd":  11_358_000,
                   "filed_date": "2024-11-20", "signal": "INSIDER_BUY"}],
        "OXY":   [{"insider": "Vicki Hollub",   "role": "CEO", "transaction_type": "Purchase",
                   "shares":  20_000, "price":  59.10, "value_usd":   1_182_000,
                   "filed_date": "2024-11-18", "signal": "INSIDER_BUY"}],
        "BAC":   [{"insider": "Brian Moynihan", "role": "CEO", "transaction_type": "Sale",
                   "shares": 150_000, "price":  38.50, "value_usd":   5_775_000,
                   "filed_date": "2024-11-22", "signal": "INSIDER_SELL"}],
        "GOOGL": [{"insider": "Sundar Pichai",  "role": "CEO", "transaction_type": "Purchase",
                   "shares":  25_000, "price": 174.20, "value_usd":   4_355_000,
                   "filed_date": "2024-11-19", "signal": "INSIDER_BUY"}],
    }


def _mock_nport() -> dict[str, list[dict[str, Any]]]:
    return {
        "Vanguard 500 Index": [
            {"ticker": "AAPL", "company": "Apple Inc.",      "shares": 12_500_000, "value_usd": 2_367_500_000, "portfolio_pct": 0.071, "change_pct":  0.08, "signal": "FUND_ACCUMULATION"},
            {"ticker": "MSFT", "company": "Microsoft Corp.", "shares":  9_800_000, "value_usd": 3_704_400_000, "portfolio_pct": 0.069, "change_pct":  0.05, "signal": "FUND_ACCUMULATION"},
            {"ticker": "NVDA", "company": "NVIDIA Corp.",    "shares":  8_200_000, "value_usd": 3_690_000_000, "portfolio_pct": 0.061, "change_pct":  0.22, "signal": "FUND_ACCUMULATION"},
            {"ticker": "AMZN", "company": "Amazon.com Inc.", "shares":  6_100_000, "value_usd": 1_037_000_000, "portfolio_pct": 0.037, "change_pct": -0.04, "signal": "FUND_LIQUIDATION"},
        ],
        "ARK Innovation ETF": [
            {"ticker": "TSLA", "company": "Tesla Inc.",     "shares": 3_200_000, "value_usd":  819_200_000, "portfolio_pct": 0.112, "change_pct":  0.18, "signal": "FUND_ACCUMULATION"},
            {"ticker": "COIN", "company": "Coinbase Global","shares": 2_100_000, "value_usd":  504_000_000, "portfolio_pct": 0.073, "change_pct":  0.31, "signal": "FUND_ACCUMULATION"},
            {"ticker": "GOOGL","company": "Alphabet Inc.",  "shares":   850_000, "value_usd":  148_100_000, "portfolio_pct": 0.021, "change_pct": -0.12, "signal": "FUND_LIQUIDATION"},
        ],
    }
