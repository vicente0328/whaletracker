"""
analysis_engine.py
------------------
Aggregates signals from all four SEC filing types into per-ticker conviction
scores and Buy / Hold / Sell recommendations.

Signal scoring:
  13F signals      NEW_ENTRY +3 | AGGRESSIVE_BUY +4 | HIGH_CONCENTRATION +2
  13D/G signals    ACTIVIST_STAKE +5 | LARGE_PASSIVE_STAKE +2
  Form 4 signals   INSIDER_BUY +3 | INSIDER_SELL -2
  N-PORT signals   FUND_ACCUMULATION +2 | FUND_LIQUIDATION -1

Recommendation thresholds (after macro adjustment):
  STRONG BUY  score ≥ 6, or score ≥ 4 with 2+ whale sources
  BUY         score ≥ 3
  HOLD        score ≥ 1
  SELL        score = 0 or negative
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conviction scoring weights
# ---------------------------------------------------------------------------

SIGNAL_SCORES: dict[str, float] = {
    # 13F signals
    "NEW_ENTRY":           3,
    "AGGRESSIVE_BUY":      4,
    "HIGH_CONCENTRATION":  2,
    "HOLD":                0,
    # SC 13D/G signals
    "ACTIVIST_STAKE":      5,
    "LARGE_PASSIVE_STAKE": 2,
    # Form 4 signals
    "INSIDER_BUY":         3,
    "INSIDER_SELL":       -2,
    "PLANNED_SELL":       -0.5,  # Rule 10b5-1 pre-planned sale — minimal bearish weight
    # N-PORT signals
    "FUND_ACCUMULATION":   2,
    "FUND_LIQUIDATION":   -1,
}

HIGH_CONVICTION_THRESHOLD = 2   # min whale sources for STRONG BUY fast-track
MAX_INSIDER_BONUS = 2           # cap on Form 4 entries counted per ticker


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_recommendations(
    whale_filings:    dict[str, list[dict[str, Any]]],
    macro_context:    dict[str, Any] | None = None,
    activist_filings: dict[str, dict[str, Any]] | None = None,
    insider_filings:  dict[str, list[dict[str, Any]]] | None = None,
    nport_filings:    dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate all filing signals into per-ticker recommendations.

    Args:
        whale_filings:     {whale_name: [holding_dict]}  ← 13F
        macro_context:     Optional {"rate_regime": "rising"|"neutral", ...}
        activist_filings:  {ticker: filing_dict}         ← SC 13D/G
        insider_filings:   {ticker: [tx_dict]}           ← Form 4
        nport_filings:     {fund_name: [holding_dict]}   ← N-PORT

    Returns:
        List of recommendation dicts sorted by conviction_score descending.
    """
    # Import tier data lazily to avoid circular imports
    try:
        from src.data_collector import WHALE_TIERS  # noqa: PLC0415
    except ImportError:
        WHALE_TIERS = {}

    ticker_map: dict[str, dict] = {}

    # ── 1. 13F whale signals (tier-weighted) ─────────────────────────────────
    for whale_name, holdings in whale_filings.items():
        tier_info   = WHALE_TIERS.get(whale_name, {"tier": 3, "multiplier": 1.0})
        multiplier  = tier_info["multiplier"]
        for holding in holdings:
            ticker = holding.get("ticker", "")
            if not ticker:
                continue
            entry  = _get_or_create(ticker_map, ticker, holding)
            signal = holding.get("signal", "HOLD")
            raw    = SIGNAL_SCORES.get(signal, 0)
            # Apply tier multiplier to positive signals only; negatives stay unweighted
            weighted = round(raw * multiplier) if raw > 0 else raw
            entry["signals"].append(signal)
            entry["conviction_score"] += weighted
            entry["supporting_whales"].append(whale_name)
            # Store tier label for UI display
            entry.setdefault("whale_tiers", {})[whale_name] = tier_info.get("label", "")

    # ── 2. SC 13D/G activist / passive stake signals ─────────────────────────
    for ticker, filing in (activist_filings or {}).items():
        entry = _get_or_create(ticker_map, ticker, {})
        signal = filing.get("signal", "LARGE_PASSIVE_STAKE")
        entry["signals"].append(signal)
        entry["conviction_score"] += SIGNAL_SCORES.get(signal, 0)
        entry["activist_filing"] = filing
        pct = filing.get("pct_outstanding", 0)
        entry["sources"].add(
            f"{'🔴 13D' if filing.get('form_type') == 'SC 13D' else '🟡 13G'} "
            f"{filing.get('filer','?')} ({pct:.1%})"
        )

    # ── 3. Form 4 insider transaction signals ────────────────────────────────
    for ticker, transactions in (insider_filings or {}).items():
        entry = _get_or_create(ticker_map, ticker, {})
        # Cap contribution to avoid over-weighting noisy insider sells
        for tx in transactions[:MAX_INSIDER_BONUS]:
            signal = tx.get("signal", "INSIDER_BUY")
            entry["signals"].append(signal)
            entry["conviction_score"] += SIGNAL_SCORES.get(signal, 0)
            role = tx.get("role", "")
            name = tx.get("insider", "?")
            entry["sources"].add(f"👤 {name} ({role})")

    # ── 4. N-PORT fund signals ───────────────────────────────────────────────
    for fund_name, holdings in (nport_filings or {}).items():
        for holding in holdings:
            ticker = holding.get("ticker", "")
            if not ticker:
                continue
            entry = _get_or_create(ticker_map, ticker, holding)
            signal = holding.get("signal", "FUND_ACCUMULATION")
            entry["signals"].append(signal)
            entry["conviction_score"] += SIGNAL_SCORES.get(signal, 0)
            chg = holding.get("change_pct", 0)
            entry["sources"].add(
                f"📦 {fund_name} ({'+' if chg >= 0 else ''}{chg:.0%})"
            )

    # ── 5. Build final list ──────────────────────────────────────────────────
    recommendations = []
    for ticker, data in ticker_map.items():
        rec = _score_to_recommendation(
            data["conviction_score"],
            len(set(data["supporting_whales"])),
            macro_context,
        )
        recommendations.append({
            "ticker":            ticker,
            "company":           data["company"],
            "signal_summary":    ", ".join(dict.fromkeys(data["signals"])),  # dedup, preserve order
            "whale_count":       len(set(data["supporting_whales"])),
            "conviction_score":  round(data["conviction_score"]),
            "recommendation":    rec,
            "supporting_whales": list(set(data["supporting_whales"])),
            "whale_tiers":       data.get("whale_tiers", {}),
            "sources":           sorted(data["sources"]),
            "macro_note":        _build_macro_note(macro_context),
        })

    recommendations.sort(key=lambda r: r["conviction_score"], reverse=True)
    return recommendations


def get_sector_rotation_signals(
    whale_filings: dict[str, list[dict[str, Any]]],
    sector_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Calculate net institutional flow score per sector from 13F filings."""
    _sector_map = _DEFAULT_SECTOR_MAP.copy()
    if sector_map:
        _sector_map.update(sector_map)

    sector_scores: dict[str, float] = {}
    for _whale, holdings in whale_filings.items():
        for holding in holdings:
            sector = _sector_map.get(holding["ticker"], "Unknown")
            score  = SIGNAL_SCORES.get(holding.get("signal", "HOLD"), 0)
            sector_scores[sector] = sector_scores.get(sector, 0.0) + score

    return dict(sorted(sector_scores.items(), key=lambda x: x[1], reverse=True))


def get_insider_sentiment(
    insider_filings: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Summarise insider buy/sell activity per ticker.

    Returns:
        {ticker: {"buy_count": int, "sell_count": int, "net_value_usd": float}}
    """
    summary: dict[str, dict] = {}
    for ticker, transactions in insider_filings.items():
        buys  = [t for t in transactions if t.get("signal") == "INSIDER_BUY"]
        sells = [t for t in transactions if t.get("signal") == "INSIDER_SELL"]
        net   = sum(t.get("value_usd", 0) for t in buys) \
              - sum(t.get("value_usd", 0) for t in sells)
        summary[ticker] = {
            "buy_count":    len(buys),
            "sell_count":   len(sells),
            "net_value_usd": net,
        }
    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_or_create(ticker_map: dict, ticker: str, holding: dict) -> dict:
    if ticker not in ticker_map:
        ticker_map[ticker] = {
            "ticker":            ticker,
            "company":           holding.get("company", ""),
            "signals":           [],
            "conviction_score":  0,
            "supporting_whales": [],
            "sources":           set(),
        }
    return ticker_map[ticker]


def _score_to_recommendation(score: int, whale_count: int, macro_context: dict | None) -> str:
    rate_penalty = 1 if (macro_context or {}).get("rate_regime") == "rising" else 0
    adjusted = score - rate_penalty

    if adjusted >= 6 or (adjusted >= 4 and whale_count >= HIGH_CONVICTION_THRESHOLD):
        return "STRONG BUY"
    if adjusted >= 3:
        return "BUY"
    if adjusted >= 1:
        return "HOLD"
    return "SELL"


def _build_macro_note(macro_context: dict | None) -> str:
    if not macro_context:
        return ""
    notes = []
    if macro_context.get("rate_regime") == "rising":
        notes.append("Rising rates may compress valuations.")
    if macro_context.get("cpi_trend") == "cooling":
        notes.append("Cooling CPI supports risk assets.")
    return " ".join(notes)


_DEFAULT_SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology",          "MSFT": "Technology",
    "NVDA": "Technology",          "GOOGL": "Technology",
    "META": "Technology",          "TSLA": "Consumer Discretionary",
    "COIN": "Financials",          "AMZN": "Consumer Discretionary",
    "XLE":  "Energy",              "CVX":  "Energy",
    "OXY":  "Energy",              "BAC":  "Financials",
    "JPM":  "Financials",          "BRK-B":"Financials",
    "UNH":  "Healthcare",          "JNJ":  "Healthcare",
}
