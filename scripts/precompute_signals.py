#!/usr/bin/env python3
"""
scripts/precompute_signals.py
-----------------------------
Pre-computes historical WhaleTracker STRONG BUY / BUY signals for every
tracked whale across the past N years and writes the results to:

    data/historical_signals.json

This file is the single source of truth for the Backtest tab.  Instead of
hitting EDGAR + FMP in real-time when the user clicks "Run Backtest" (which
would take 3–8 minutes and burn API quotas), the backtester reads this
pre-computed file and only needs to fetch price data from FMP (~30 s).

── How signals are computed ────────────────────────────────────────────────
For each calendar quarter (identified by report_date = quarter-end date):
  1. Collect every 13F filing that any tracked whale submitted for that quarter.
  2. For each whale in the group, parse its holdings XML from EDGAR and
     compare with its own PRIOR quarter holdings to detect:
       NEW_ENTRY          → ticker absent from prior quarter          score 3
       AGGRESSIVE_BUY     → shares +>20 % QoQ                        score 4
       HIGH_CONCENTRATION → position >5 % of portfolio value          score 2
  3. Scale each per-whale signal score by that whale's tier multiplier
     (from WHALE_TIERS: T1 = 1.3–1.5, T2 = 1.1–1.3, T3 = 0.9–1.0).
  4. Sum scaled scores across ALL whales for each ticker.
  5. Classify:
       STRONG BUY  if total_score ≥ 6  OR  (score ≥ 4 AND whale_count ≥ 2)
       BUY         if total_score ≥ 3
       HOLD        otherwise

── Signal date vs. report date ─────────────────────────────────────────────
The "signal_date" stored in the output is the MAX(filed_date) across all
whales for that quarter — i.e., the earliest date at which an investor
would have had COMPLETE information from all whales.  The backtester
executes trades on signal_date, not on report_date.

── Caching ──────────────────────────────────────────────────────────────────
Parsed holdings (one JSON file per accession number) are cached in
data/edgar_cache/ so that re-running this script is fast.  Delete the cache
directory to force a full re-fetch.

── Usage ────────────────────────────────────────────────────────────────────
  cd /path/to/whaletracker
  python scripts/precompute_signals.py [--years 5] [--no-cache]

Requirements: FMP_API_KEY is NOT needed here (only EDGAR is used).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# ── Bootstrap — add project root to sys.path ─────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.data_collector import (  # noqa: E402
    TRACKED_WHALES,
    WHALE_TIERS,
    detect_signals,
    _find_13f_holdings_doc,
    _parse_13f_xml,
)

load_dotenv(_ROOT / ".env")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = _ROOT / "data"
CACHE_DIR  = DATA_DIR / "edgar_cache"
OUTPUT_FILE = DATA_DIR / "historical_signals.json"

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ── EDGAR constants ───────────────────────────────────────────────────────────
_EDGAR_HEADERS = {
    "User-Agent":      "WhaleTracker research@whaletracker.ai",
    "Accept-Encoding": "gzip, deflate",
}
_EDGAR_RATE_S = 0.35   # pause between EDGAR requests (< 10 req/s policy)

# ── Filing date cap ──────────────────────────────────────────────────────────
# 13F-HR/A amendments can be filed months or years late, which would otherwise
# push a quarter's signal_date far past when the original data became public.
# We cap signal_date to report_date + CAP_DAYS so amended filings don't distort
# the simulation timeline.
_CAP_DAYS = 90   # 45-day legal deadline + 45 days tolerance

def _cap_filed_date(report_date: str, filed_date: str) -> str:
    """Return filed_date capped at report_date + CAP_DAYS.

    This prevents late 13F-HR/A amendments from pushing a quarter's
    signal_date months into the future and colliding with other quarters.
    """
    try:
        rdt = datetime.strptime(report_date, "%Y-%m-%d")
        fdt = datetime.strptime(filed_date,  "%Y-%m-%d")
        cap = rdt + timedelta(days=_CAP_DAYS)
        return min(fdt, cap).strftime("%Y-%m-%d")
    except Exception:
        return filed_date


# ── Signal scoring (mirrors analysis_engine.py) ───────────────────────────────
_SIG_SCORE = {
    "NEW_ENTRY":          3,
    "AGGRESSIVE_BUY":     4,
    "HIGH_CONCENTRATION": 2,
}

STRONG_BUY_THRESHOLD = 6      # OR score ≥ 4 with 2+ whales
BUY_THRESHOLD        = 3


# ═══════════════════════════════════════════════════════════════════════════════
# EDGAR helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _edgar_get(url: str, *, retries: int = 3) -> dict | None:
    """GET an EDGAR JSON endpoint with retry + rate-limit pause."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_EDGAR_HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            wait = _EDGAR_RATE_S * (attempt + 1) * 2
            logger.debug("EDGAR GET failed (%s) attempt %d — retry in %.1fs: %s",
                         url, attempt + 1, wait, exc)
            time.sleep(wait)
    return None


def fetch_accessions(cik: str, n: int = 24) -> list[dict]:
    """
    Return up to `n` most-recent 13F-HR filings for `cik` (newest first).

    Each entry: {accession, filed_date, report_date}

    Handles EDGAR pagination: the submissions JSON lists the most recent
    filings inline; older ones are referenced in `filings.files`.
    """
    cik_padded = cik.lstrip("0").zfill(10)
    url        = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    data       = _edgar_get(url)
    if not data:
        return []

    def _extract(recent: dict) -> list[dict]:
        forms  = recent.get("form", [])
        dates  = recent.get("filingDate", [])
        perds  = recent.get("reportDate", [])
        accs   = recent.get("accessionNumber", [])
        out    = []
        for i, form in enumerate(forms):
            if form.strip().upper() not in {"13F-HR", "13F-HR/A"}:
                continue
            out.append({
                "accession":   accs[i],
                "filed_date":  dates[i],
                "report_date": perds[i],
            })
        return out

    results = _extract(data.get("filings", {}).get("recent", {}))

    # Fetch paginated older filings if we don't have enough yet
    if len(results) < n:
        for page_ref in data.get("filings", {}).get("files", []):
            if len(results) >= n:
                break
            page_url  = f"https://data.sec.gov/submissions/{page_ref['name']}"
            page_data = _edgar_get(page_url)
            if page_data:
                results.extend(_extract(page_data))
            time.sleep(_EDGAR_RATE_S)

    # Sort newest-first, deduplicate by accession
    seen = set()
    unique = []
    for r in results:
        if r["accession"] not in seen:
            seen.add(r["accession"])
            unique.append(r)
    unique.sort(key=lambda x: x["filed_date"], reverse=True)
    return unique[:n]


# ═══════════════════════════════════════════════════════════════════════════════
# Holdings cache (persist parsed XML → JSON to avoid re-fetching)
# ═══════════════════════════════════════════════════════════════════════════════

def _cache_path(accession: str) -> Path:
    safe = accession.replace("-", "").replace("/", "_")
    return CACHE_DIR / f"{safe}.json"


def _deduplicate_holdings(holdings: list[dict]) -> list[dict]:
    """Merge duplicate ticker entries into one (keep highest value_usd entry).

    13F XMLs commonly list the same company multiple times under different
    security types (e.g., common stock, call options, put options) or share
    classes that resolve to the same ticker symbol.  Without deduplication
    the signal scores are artificially inflated (e.g., AAPL scoring 85 from
    a single whale instead of a realistic 6–9).

    We keep only the entry with the highest value_usd per ticker (most likely
    the underlying equity position rather than an option overlay), then
    recompute portfolio_pct based on the deduplicated total.
    """
    best: dict[str, dict] = {}
    for h in holdings:
        t = h.get("ticker", "")
        if not t:
            continue
        if t not in best or h.get("value_usd", 0) > best[t].get("value_usd", 0):
            best[t] = h
    total = sum(h.get("value_usd", 0) for h in best.values())
    return [
        {**h, "portfolio_pct": h["value_usd"] / total if total else 0}
        for h in sorted(best.values(), key=lambda x: x.get("value_usd", 0), reverse=True)
    ]


def load_holdings_cached(
    cik: str,
    acc_info: dict,
    use_cache: bool = True,
) -> list[dict]:
    """
    Return parsed + deduplicated holdings for `acc_info`.

    Uses local JSON cache (data/edgar_cache/) when available.
    On cache miss: fetches the XML from EDGAR, deduplicates, writes cache.
    Returns [] on any error.
    """
    path = _cache_path(acc_info["accession"])

    if use_cache and path.exists():
        try:
            raw = json.loads(path.read_text())
            # Apply dedup even on cache hits — old cache files may be undeduped
            return _deduplicate_holdings(raw)
        except Exception:
            pass   # corrupt cache — re-fetch

    cik_int = int(cik.lstrip("0") or "0")
    try:
        xml_url = _find_13f_holdings_doc(cik_int, acc_info["accession"])
        time.sleep(_EDGAR_RATE_S)
        if not xml_url:
            logger.debug("No holdings doc URL for %s", acc_info["accession"])
            return []
        raw_holdings = _parse_13f_xml(xml_url)
        time.sleep(_EDGAR_RATE_S)
    except Exception as exc:
        logger.warning("Holdings fetch failed for %s: %s", acc_info["accession"], exc)
        return []

    holdings = _deduplicate_holdings(raw_holdings)

    if use_cache:
        try:
            path.write_text(json.dumps(holdings))
        except Exception:
            pass   # non-fatal

    return holdings


# ═══════════════════════════════════════════════════════════════════════════════
# Signal computation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _score_holdings(holdings_with_signals: list[dict], multiplier: float) -> dict[str, dict]:
    """
    Convert a single whale's detect_signals() output into a per-ticker score map.

    Returns: {ticker: {"score": float, "company": str, "signals": [str]}}
    """
    out: dict[str, dict] = {}
    for h in holdings_with_signals:
        sig  = h.get("signal", "HOLD")
        base = _SIG_SCORE.get(sig, 0)
        if base <= 0:
            continue
        t = h["ticker"]
        if t not in out:
            out[t] = {"score": 0.0, "company": h.get("company", t), "signals": []}
        out[t]["score"] += base * multiplier
        if sig not in out[t]["signals"]:
            out[t]["signals"].append(sig)
    return out


def aggregate_quarter_signals(
    whale_scored: dict[str, dict[str, dict]],   # {whale_name: {ticker: scored}}
) -> dict[str, dict]:
    """
    Merge per-whale scores into a final {ticker: {...}} dict with STRONG BUY/BUY.

    whale_scored = {
        "Berkshire Hathaway": {"AAPL": {"score": 6.0, "company": "...", "signals": [...]}},
        "Tiger Global":        {"NVDA": {"score": 4.0, ...}},
    }
    """
    agg: dict[str, dict] = {}
    for whale, ticker_map in whale_scored.items():
        for ticker, info in ticker_map.items():
            if ticker not in agg:
                agg[ticker] = {
                    "score":       0.0,
                    "company":     info["company"],
                    "whale_count": 0,
                    "whales":      [],
                    "signals":     [],
                }
            agg[ticker]["score"]       += info["score"]
            agg[ticker]["whale_count"] += 1
            agg[ticker]["whales"].append(whale)
            for sig in info["signals"]:
                if sig not in agg[ticker]["signals"]:
                    agg[ticker]["signals"].append(sig)

    # Assign recommendation
    for info in agg.values():
        sc = info["score"]
        wc = info["whale_count"]
        if sc >= STRONG_BUY_THRESHOLD or (sc >= 4 and wc >= 2):
            info["recommendation"] = "STRONG BUY"
        elif sc >= BUY_THRESHOLD:
            info["recommendation"] = "BUY"
        else:
            info["recommendation"] = "HOLD"

    return agg


def quarter_label(report_date: str) -> str:
    """'2024-09-30'  →  'Q3 2024'"""
    try:
        d = datetime.strptime(report_date, "%Y-%m-%d")
        q = (d.month - 1) // 3 + 1
        return f"Q{q} {d.year}"
    except Exception:
        return report_date


# ═══════════════════════════════════════════════════════════════════════════════
# Main pre-computation logic
# ═══════════════════════════════════════════════════════════════════════════════

def build_quarter_map(
    whale_accessions: dict[str, list[dict]],
    cutoff_date: str,
) -> dict[str, dict]:
    """
    Group all whale accessions by report_date to form a quarter map.

    Returns:
        { report_date: {
            "filed_dates": [str, ...],         # one per whale
            "members": [(whale, cik, acc_info), ...]
          }
        }

    Only includes quarters whose signal_date (max filed_date) >= cutoff_date.
    """
    quarter_map: dict[str, dict] = {}

    for whale, cik in TRACKED_WHALES.items():
        for acc_info in whale_accessions.get(whale, []):
            rdate = acc_info["report_date"]
            fdate = acc_info["filed_date"]
            # Cap filed_date to prevent late amendments from distorting signal_date
            capped_fdate = _cap_filed_date(rdate, fdate)
            if rdate not in quarter_map:
                quarter_map[rdate] = {"filed_dates": [], "members": []}
            quarter_map[rdate]["filed_dates"].append(capped_fdate)
            quarter_map[rdate]["members"].append((whale, cik, acc_info))

    # Remove quarters where we never get a signal_date >= cutoff
    # (i.e., the entire quarter was before the backtest window)
    to_remove = []
    for rdate, qdata in quarter_map.items():
        signal_date = max(qdata["filed_dates"])
        if signal_date < cutoff_date:
            to_remove.append(rdate)
    for r in to_remove:
        del quarter_map[r]

    return quarter_map


def compute_signals_for_quarter(
    whale_accs: dict[str, list[dict]],   # full accession list per whale (for prior lookup)
    members:    list[tuple],              # [(whale, cik, acc_info), ...] for THIS quarter
    use_cache:  bool,
) -> dict[str, dict]:
    """
    Parse holdings + prior-quarter holdings for each whale in `members`,
    run detect_signals(), score, and aggregate.

    Returns: {ticker: {score, recommendation, company, whale_count, whales, signals}}
    """
    whale_scored: dict[str, dict[str, dict]] = {}

    for whale, cik, acc_info in members:
        # ── Current quarter holdings ─────────────────────────────────────────
        current = load_holdings_cached(cik, acc_info, use_cache)
        if not current:
            logger.debug("  ↳ %s: no holdings parsed — skipping", whale)
            continue

        # ── Prior quarter holdings (for QoQ signal detection) ───────────────
        whale_list = whale_accs.get(whale, [])
        # Find index of this accession in the whale's list (sorted newest-first)
        this_idx = next(
            (i for i, a in enumerate(whale_list)
             if a["accession"] == acc_info["accession"]),
            None,
        )
        prior: list[dict] = []
        if this_idx is not None and this_idx + 1 < len(whale_list):
            prior_info = whale_list[this_idx + 1]
            prior = load_holdings_cached(cik, prior_info, use_cache)

        # ── Signal detection ─────────────────────────────────────────────────
        holdings_with_signals = detect_signals(current, prior)

        # ── Score using per-whale multiplier ──────────────────────────────────
        mult = WHALE_TIERS.get(whale, {}).get("multiplier", 1.0)
        scored = _score_holdings(holdings_with_signals, mult)
        if scored:
            whale_scored[whale] = scored
            sb = sum(1 for v in scored.values() if v["score"] >= STRONG_BUY_THRESHOLD)
            logger.debug("  ↳ %-26s  mult=%.1f  tickers=%d  strong_buy=%d",
                         whale, mult, len(scored), sb)

    return aggregate_quarter_signals(whale_scored)


def run(years: int = 5, use_cache: bool = True) -> None:
    """
    Full pre-computation pipeline.

    Args:
        years:     How many years of history to compute (1, 3, or 5).
        use_cache: Whether to use cached XML holdings from data/edgar_cache/.
    """
    n_quarters_needed = years * 4 + 2   # +2 gives one extra quarter for prior comparison
    cutoff_date = datetime.utcnow().date().replace(
        year=datetime.utcnow().year - years
    ).strftime("%Y-%m-%d")

    logger.info("━" * 60)
    logger.info("WhaleTracker Signal Pre-computation")
    logger.info("  years=%d  cutoff=%s  use_cache=%s", years, cutoff_date, use_cache)
    logger.info("  tracked whales: %d", len(TRACKED_WHALES))
    logger.info("━" * 60)

    # ── Step 1: Fetch accession lists from EDGAR ──────────────────────────────
    logger.info("")
    logger.info("STEP 1 — Fetching 13F accession lists from EDGAR …")

    whale_accs: dict[str, list[dict]] = {}
    for i, (whale, cik) in enumerate(TRACKED_WHALES.items(), 1):
        logger.info("  [%2d/%d] %-30s (CIK %s)", i, len(TRACKED_WHALES), whale, cik)
        accs = fetch_accessions(cik, n=n_quarters_needed)
        whale_accs[whale] = accs
        logger.info("         → %d filings found", len(accs))
        time.sleep(_EDGAR_RATE_S)

    # ── Step 2: Build quarter map ─────────────────────────────────────────────
    logger.info("")
    logger.info("STEP 2 — Grouping filings into quarters …")

    quarter_map = build_quarter_map(whale_accs, cutoff_date)
    report_dates = sorted(quarter_map.keys())   # oldest → newest
    logger.info("  → %d quarters in backtest window", len(report_dates))

    if not report_dates:
        logger.error("No quarters found for the given time window.  Aborting.")
        sys.exit(1)

    for rdate in report_dates:
        qdata = quarter_map[rdate]
        signal_date = max(qdata["filed_dates"])
        coverage    = len(qdata["members"])
        logger.info("  %-12s  signal_date=%-12s  whales=%d/%d",
                    quarter_label(rdate), signal_date, coverage, len(TRACKED_WHALES))

    # ── Step 3: Parse holdings + compute signals per quarter ─────────────────
    logger.info("")
    logger.info("STEP 3 — Computing signals quarter-by-quarter …")

    quarters_output: list[dict] = []
    total_q = len(report_dates)

    for q_idx, rdate in enumerate(report_dates, 1):
        qdata       = quarter_map[rdate]
        signal_date = max(qdata["filed_dates"])
        first_filed = min(qdata["filed_dates"])
        label       = quarter_label(rdate)
        members     = qdata["members"]

        logger.info("")
        logger.info("  [%2d/%d] %s  (signal_date=%s, %d whales)",
                    q_idx, total_q, label, signal_date, len(members))

        tickers = compute_signals_for_quarter(whale_accs, members, use_cache)

        # Summary
        n_strong = sum(1 for v in tickers.values() if v["recommendation"] == "STRONG BUY")
        n_buy    = sum(1 for v in tickers.values() if v["recommendation"] == "BUY")
        n_total  = len(tickers)
        logger.info("  → %d tickers scored  |  STRONG BUY: %d  |  BUY: %d",
                    n_total, n_strong, n_buy)
        if n_strong:
            sb_list = sorted(
                [t for t, v in tickers.items() if v["recommendation"] == "STRONG BUY"],
                key=lambda t: tickers[t]["score"],
                reverse=True,
            )[:8]
            logger.info("    STRONG BUY: %s", "  ".join(sb_list))

        quarters_output.append({
            "report_date":    rdate,
            "signal_date":    signal_date,
            "first_filed":    first_filed,
            "label":          label,
            "whale_coverage": len(members),
            "tickers":        tickers,
        })

    # ── Step 4: Write output ──────────────────────────────────────────────────
    logger.info("")
    logger.info("STEP 4 — Writing output …")

    output = {
        "meta": {
            "computed_at":       datetime.now(timezone.utc).isoformat(),
            "years_covered":     years,
            "cutoff_date":       cutoff_date,
            "quarters_computed": len(quarters_output),
            "whales":            list(TRACKED_WHALES.keys()),
            "strong_buy_threshold": STRONG_BUY_THRESHOLD,
            "buy_threshold":        BUY_THRESHOLD,
            "schema_version":    "1.0",
        },
        "quarters": quarters_output,
    }

    OUTPUT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    total_strong_buys = sum(
        sum(1 for v in q["tickers"].values() if v["recommendation"] == "STRONG BUY")
        for q in quarters_output
    )
    cache_files = len(list(CACHE_DIR.glob("*.json")))

    logger.info("  ✓ Written to %s", OUTPUT_FILE)
    logger.info("  ✓ %d quarters  |  %d total STRONG BUY signals  |  %d cached XMLs",
                len(quarters_output), total_strong_buys, cache_files)
    logger.info("")
    logger.info("━" * 60)
    logger.info("Pre-computation complete.")
    logger.info("━" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute historical WhaleTracker signals for backtest.",
    )
    parser.add_argument(
        "--years", type=int, default=5,
        choices=[1, 3, 5],
        help="How many years of history to compute (default: 5)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore existing edgar_cache and re-fetch all XMLs from EDGAR",
    )
    args = parser.parse_args()

    run(years=args.years, use_cache=not args.no_cache)
