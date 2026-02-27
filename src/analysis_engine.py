"""
analysis_engine.py
------------------
Correlates Whale signal data with macroeconomic context and recent price
action to produce Buy / Hold / Sell recommendations with a conviction score.

Key considerations:
  - 13F filings have a ~45-day reporting lag; price action is used to
    confirm or discount stale positions.
  - Macro context (interest rate regime, CPI trend) shifts conviction
    thresholds.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conviction scoring weights
# ---------------------------------------------------------------------------

SIGNAL_SCORES: dict[str, int] = {
    "NEW_ENTRY":          3,
    "AGGRESSIVE_BUY":     4,
    "HIGH_CONCENTRATION": 2,
    "HOLD":               0,
}

# How many Whales must agree before conviction is "HIGH"
HIGH_CONVICTION_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_recommendations(
    whale_filings: dict[str, list[dict[str, Any]]],
    macro_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate Whale signals into per-ticker recommendations.

    Args:
        whale_filings:  Output of data_collector.fetch_all_whale_filings().
        macro_context:  Optional macro snapshot, e.g.:
                        {"rate_regime": "rising", "cpi_trend": "cooling"}

    Returns:
        List of recommendation dicts, sorted by conviction score descending.
        Each dict contains:
          ticker, signal_summary, whale_count, conviction_score,
          recommendation, supporting_whales, macro_note
    """
    ticker_map: dict[str, dict] = {}

    for whale_name, holdings in whale_filings.items():
        for holding in holdings:
            ticker = holding.get("ticker", "")
            if not ticker:
                continue

            if ticker not in ticker_map:
                ticker_map[ticker] = {
                    "ticker": ticker,
                    "company": holding.get("company", ""),
                    "signals": [],
                    "conviction_score": 0,
                    "supporting_whales": [],
                }

            signal = holding.get("signal", "HOLD")
            ticker_map[ticker]["signals"].append(signal)
            ticker_map[ticker]["conviction_score"] += SIGNAL_SCORES.get(signal, 0)
            ticker_map[ticker]["supporting_whales"].append(whale_name)

    recommendations = []
    for ticker, data in ticker_map.items():
        recommendation = _score_to_recommendation(
            data["conviction_score"],
            len(data["supporting_whales"]),
            macro_context,
        )
        macro_note = _build_macro_note(ticker, macro_context)

        recommendations.append({
            "ticker": ticker,
            "company": data["company"],
            "signal_summary": ", ".join(set(data["signals"])),
            "whale_count": len(set(data["supporting_whales"])),
            "conviction_score": data["conviction_score"],
            "recommendation": recommendation,
            "supporting_whales": list(set(data["supporting_whales"])),
            "macro_note": macro_note,
        })

    recommendations.sort(key=lambda r: r["conviction_score"], reverse=True)
    return recommendations


def get_sector_rotation_signals(
    whale_filings: dict[str, list[dict[str, Any]]],
    sector_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Calculate net institutional flow score per sector.

    Args:
        whale_filings: Output of fetch_all_whale_filings().
        sector_map:    Optional {ticker: sector} override. Falls back to a
                       built-in map for common tickers.

    Returns:
        {sector: net_score} sorted descending — positive = inflows.
    """
    _sector_map = _DEFAULT_SECTOR_MAP.copy()
    if sector_map:
        _sector_map.update(sector_map)

    sector_scores: dict[str, float] = {}

    for _whale, holdings in whale_filings.items():
        for holding in holdings:
            sector = _sector_map.get(holding["ticker"], "Unknown")
            score = SIGNAL_SCORES.get(holding.get("signal", "HOLD"), 0)
            sector_scores[sector] = sector_scores.get(sector, 0.0) + score

    return dict(sorted(sector_scores.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_to_recommendation(
    score: int,
    whale_count: int,
    macro_context: dict | None,
) -> str:
    rate_regime = (macro_context or {}).get("rate_regime", "neutral")

    # Rising rates penalise growth / tech slightly
    rate_penalty = 1 if rate_regime == "rising" else 0

    adjusted = score - rate_penalty

    if adjusted >= 6 or (adjusted >= 4 and whale_count >= HIGH_CONVICTION_THRESHOLD):
        return "STRONG BUY"
    if adjusted >= 3:
        return "BUY"
    if adjusted >= 1:
        return "HOLD"
    return "SELL"


def _build_macro_note(ticker: str, macro_context: dict | None) -> str:
    if not macro_context:
        return ""
    notes = []
    if macro_context.get("rate_regime") == "rising":
        notes.append("Rising rates may compress valuations.")
    if macro_context.get("cpi_trend") == "cooling":
        notes.append("Cooling CPI supports risk assets.")
    return " ".join(notes)


# Lightweight sector lookup for common tickers (extend as needed)
_DEFAULT_SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology", "AMZN": "Consumer Discretionary",
    "XLE": "Energy", "CVX": "Energy", "OXY": "Energy",
    "BAC": "Financials", "JPM": "Financials", "BRK-B": "Financials",
    "UNH": "Healthcare", "JNJ": "Healthcare",
}
