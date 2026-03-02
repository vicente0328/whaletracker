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
    # ── Technology ────────────────────────────────────────────────────────
    "AAPL":  "Technology",   "MSFT":  "Technology",   "NVDA":  "Technology",
    "GOOGL": "Technology",   "GOOG":  "Technology",   "META":  "Technology",
    "AMD":   "Technology",   "INTC":  "Technology",   "QCOM":  "Technology",
    "AVGO":  "Technology",   "TXN":   "Technology",   "MU":    "Technology",
    "AMAT":  "Technology",   "LRCX":  "Technology",   "KLAC":  "Technology",
    "ADI":   "Technology",   "MRVL":  "Technology",   "NXPI":  "Technology",
    "HPE":   "Technology",   "WDC":   "Technology",   "STX":   "Technology",
    "IBM":   "Technology",   "ORCL":  "Technology",   "CRM":   "Technology",
    "SAP":   "Technology",   "NOW":   "Technology",   "SNOW":  "Technology",
    "DDOG":  "Technology",   "PLTR":  "Technology",   "NET":   "Technology",
    "CRWD":  "Technology",   "ZS":    "Technology",   "PANW":  "Technology",
    "FTNT":  "Technology",   "OKTA":  "Technology",   "MDB":   "Technology",
    "TEAM":  "Technology",   "HUBS":  "Technology",   "ZM":    "Technology",
    "ADBE":  "Technology",   "INTU":  "Technology",   "ANSS":  "Technology",
    "CDNS":  "Technology",   "SNPS":  "Technology",   "UBER":  "Technology",
    # ── Consumer Discretionary ────────────────────────────────────────────
    "AMZN":  "Consumer Discretionary",   "TSLA":  "Consumer Discretionary",
    "HD":    "Consumer Discretionary",   "MCD":   "Consumer Discretionary",
    "NKE":   "Consumer Discretionary",   "SBUX":  "Consumer Discretionary",
    "CMG":   "Consumer Discretionary",   "YUM":   "Consumer Discretionary",
    "DPZ":   "Consumer Discretionary",   "BKNG":  "Consumer Discretionary",
    "ABNB":  "Consumer Discretionary",   "MAR":   "Consumer Discretionary",
    "HLT":   "Consumer Discretionary",   "LVS":   "Consumer Discretionary",
    "WYNN":  "Consumer Discretionary",   "RCL":   "Consumer Discretionary",
    "CCL":   "Consumer Discretionary",   "F":     "Consumer Discretionary",
    "GM":    "Consumer Discretionary",   "TGT":   "Consumer Discretionary",
    "LOW":   "Consumer Discretionary",   "ROST":  "Consumer Discretionary",
    "TJX":   "Consumer Discretionary",   "BBY":   "Consumer Discretionary",
    # ── Consumer Staples ──────────────────────────────────────────────────
    "WMT":   "Consumer Staples",   "KO":    "Consumer Staples",
    "PEP":   "Consumer Staples",   "PG":    "Consumer Staples",
    "COST":  "Consumer Staples",   "MDLZ":  "Consumer Staples",
    "KHC":   "Consumer Staples",   "GIS":   "Consumer Staples",
    "KMB":   "Consumer Staples",   "CL":    "Consumer Staples",
    "EL":    "Consumer Staples",   "CLX":   "Consumer Staples",
    "MO":    "Consumer Staples",   "PM":    "Consumer Staples",
    # ── Healthcare ────────────────────────────────────────────────────────
    "UNH":   "Healthcare",   "JNJ":   "Healthcare",   "LLY":   "Healthcare",
    "ABT":   "Healthcare",   "MRK":   "Healthcare",   "PFE":   "Healthcare",
    "ABBV":  "Healthcare",   "BMY":   "Healthcare",   "AMGN":  "Healthcare",
    "GILD":  "Healthcare",   "BIIB":  "Healthcare",   "VRTX":  "Healthcare",
    "REGN":  "Healthcare",   "ISRG":  "Healthcare",   "SYK":   "Healthcare",
    "MDT":   "Healthcare",   "BSX":   "Healthcare",   "DHR":   "Healthcare",
    "TMO":   "Healthcare",   "IQV":   "Healthcare",   "CVS":   "Healthcare",
    # ── Financials ────────────────────────────────────────────────────────
    "BRK-B": "Financials",   "JPM":   "Financials",   "BAC":   "Financials",
    "WFC":   "Financials",   "MS":    "Financials",   "GS":    "Financials",
    "C":     "Financials",   "AXP":   "Financials",   "V":     "Financials",
    "MA":    "Financials",   "PYPL":  "Financials",   "COIN":  "Financials",
    "BLK":   "Financials",   "SCHW":  "Financials",   "CB":    "Financials",
    "PGR":   "Financials",   "MET":   "Financials",   "AIG":   "Financials",
    "SPGI":  "Financials",   "MCO":   "Financials",   "ICE":   "Financials",
    "CME":   "Financials",   "KKR":   "Financials",   "APO":   "Financials",
    # ── Energy ────────────────────────────────────────────────────────────
    "XOM":   "Energy",   "CVX":   "Energy",   "OXY":   "Energy",
    "COP":   "Energy",   "EOG":   "Energy",   "SLB":   "Energy",
    "MPC":   "Energy",   "VLO":   "Energy",   "PSX":   "Energy",
    "HES":   "Energy",   "DVN":   "Energy",   "FANG":  "Energy",
    "HAL":   "Energy",   "BKR":   "Energy",   "XLE":   "Energy",
    "CTRA":  "Energy",   "APA":   "Energy",
    # ── Industrials ───────────────────────────────────────────────────────
    "GE":    "Industrials",   "CAT":   "Industrials",   "HON":   "Industrials",
    "RTX":   "Industrials",   "LMT":   "Industrials",   "NOC":   "Industrials",
    "BA":    "Industrials",   "DE":    "Industrials",   "MMM":   "Industrials",
    "EMR":   "Industrials",   "ETN":   "Industrials",   "PH":    "Industrials",
    "UPS":   "Industrials",   "FDX":   "Industrials",   "CSX":   "Industrials",
    "NSC":   "Industrials",   "WAB":   "Industrials",   "GD":    "Industrials",
    # ── Real Estate ───────────────────────────────────────────────────────
    "AMT":   "Real Estate",   "PLD":   "Real Estate",   "EQIX":  "Real Estate",
    "SPG":   "Real Estate",   "PSA":   "Real Estate",   "O":     "Real Estate",
    "WELL":  "Real Estate",   "VTR":   "Real Estate",   "EQR":   "Real Estate",
    # ── Communication Services ────────────────────────────────────────────
    "NFLX":  "Communication Services",   "DIS":   "Communication Services",
    "CMCSA": "Communication Services",   "T":     "Communication Services",
    "VZ":    "Communication Services",   "CHTR":  "Communication Services",
    "TMUS":  "Communication Services",   "EA":    "Communication Services",
    "TTWO":  "Communication Services",   "WBD":   "Communication Services",
    # ── Materials ─────────────────────────────────────────────────────────
    "LIN":   "Materials",   "APD":   "Materials",   "SHW":   "Materials",
    "NEM":   "Materials",   "FCX":   "Materials",   "NUE":   "Materials",
    "ALB":   "Materials",   "CF":    "Materials",   "MOS":   "Materials",
    # ── Utilities ─────────────────────────────────────────────────────────
    "NEE":   "Utilities",   "SO":    "Utilities",   "DUK":   "Utilities",
    "AEP":   "Utilities",   "EXC":   "Utilities",   "XEL":   "Utilities",
    "PCG":   "Utilities",   "SRE":   "Utilities",   "WEC":   "Utilities",
}


# ---------------------------------------------------------------------------
# Whale-Retail Alignment (Reddit ↔ 13F 교차 분석)
# ---------------------------------------------------------------------------

# 개미 감성 분류 임계값
_RETAIL_BULL = +0.2
_RETAIL_BEAR = -0.2

# 고래 강세 추천 레이블
_WHALE_BULL_RECS = frozenset({"STRONG BUY", "BUY"})


def get_whale_retail_alignment(
    posts: list[dict],
    signals_data: dict | None,
) -> dict:
    """
    Reddit DD 포스트 감성과 최신 고래 13F 시그널을 교차 분석합니다.

    분류:
      aligned_bull  — 고래 강세 + 개미 강세  (합의 매수)
      divergence    — 고래 강세 + 개미 약세  (역발상 신호 — 컨트라리안 기회)
      hype_trap     — 고래 미포지션 + 개미 강세  (하이프 트랩 — 주의)
      aligned_bear  — 고래 미포지션/약세 + 개미 약세  (합의 약세)

    Args:
        posts:        fetch_dd_posts()가 반환한 엔리치먼트 게시글 목록.
        signals_data: load_historical_signals()가 반환한 dict. None이면 고래 데이터 없음으로 처리.

    Returns:
        {
            "ticker_sentiments": {TICKER: {"avg_sentiment", "mention_count", "total_upvotes"}},
            "whale_map":         {TICKER: {"recommendation", "score", "whale_count"}},
            "aligned_bull":  [item, ...],
            "divergence":    [item, ...],
            "hype_trap":     [item, ...],
            "aligned_bear":  [item, ...],
        }
        item = {"ticker", "avg_sentiment", "total_upvotes", "mention_count",
                "whale_rec", "whale_score", "whale_count", "category"}
    """
    # ── 1. 티커별 감성 집계 ──────────────────────────────────────────────────
    ticker_sents: dict[str, dict] = {}
    for post in posts:
        tickers   = post.get("tickers") or []
        sentiment = float(post.get("sentiment", 0.0))
        upvotes   = int(post.get("upvotes", 0))

        for ticker in tickers[:3]:   # 포스트당 최대 3개 티커
            if ticker not in ticker_sents:
                ticker_sents[ticker] = {
                    "sentiment_sum":  0.0,
                    "mention_count":  0,
                    "total_upvotes":  0,
                    "avg_sentiment":  0.0,
                }
            s = ticker_sents[ticker]
            # 업보트 가중 감성 합산
            s["sentiment_sum"] += sentiment * (upvotes + 1)
            s["mention_count"] += 1
            s["total_upvotes"] += upvotes

    # 가중 평균 감성 계산 후 임시 필드 제거
    for ticker, s in ticker_sents.items():
        denom = s["total_upvotes"] + s["mention_count"]
        s["avg_sentiment"] = s["sentiment_sum"] / denom if denom else 0.0
        del s["sentiment_sum"]

    # ── 2. 고래 시그널 맵 구성 ───────────────────────────────────────────────
    whale_map: dict[str, dict] = {}
    if signals_data:
        quarters = signals_data.get("quarters", [])
        if quarters:
            latest = quarters[-1].get("tickers", {})
            for ticker, info in latest.items():
                whale_map[ticker] = {
                    "recommendation": info.get("recommendation"),
                    "score":          float(info.get("score", 0)),
                    "whale_count":    int(info.get("whale_count", 0)),
                }

    # ── 3. 분류 ──────────────────────────────────────────────────────────────
    result: dict[str, list] = {
        "aligned_bull": [],
        "divergence":   [],
        "hype_trap":    [],
        "aligned_bear": [],
    }

    for ticker, s in ticker_sents.items():
        avg_sent  = s["avg_sentiment"]
        whale_inf = whale_map.get(ticker, {})
        whale_rec = whale_inf.get("recommendation")

        is_retail_bull = avg_sent >  _RETAIL_BULL
        is_retail_bear = avg_sent <  _RETAIL_BEAR
        is_whale_bull  = whale_rec in _WHALE_BULL_RECS

        if   is_whale_bull and is_retail_bull:
            cat = "aligned_bull"
        elif is_whale_bull and is_retail_bear:
            cat = "divergence"
        elif (not is_whale_bull) and is_retail_bull:
            cat = "hype_trap"
        else:
            cat = "aligned_bear"

        item = {
            "ticker":        ticker,
            "avg_sentiment": round(avg_sent, 3),
            "total_upvotes": s["total_upvotes"],
            "mention_count": s["mention_count"],
            "whale_rec":     whale_rec,
            "whale_score":   whale_inf.get("score"),
            "whale_count":   whale_inf.get("whale_count"),
            "category":      cat,
        }
        result[cat].append(item)

    # 각 카테고리 내 업보트 기준 정렬
    for cat in result:
        result[cat].sort(key=lambda x: x["total_upvotes"], reverse=True)

    result["ticker_sentiments"] = ticker_sents   # type: ignore[assignment]
    result["whale_map"]         = whale_map       # type: ignore[assignment]
    return result
