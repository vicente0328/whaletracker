"""
backtester.py
-------------
Backtests WhaleTracker's institutional signal strategy against historical data.

Strategy:
  At each 13F filing date (information becomes public ~45 days after quarter end):
    1. Compute per-ticker conviction scores from all tracked whales
    2. Enter equal-weight positions in tickers meeting the chosen threshold
       (STRONG BUY: score ≥ 6,  BUY: score ≥ 3)
    3. Rebalance at the next quarterly filing event
    4. Benchmark: buy-and-hold SPY with the same initial capital

Data sources:
  - SEC EDGAR (free): historical 13F filing dates + holdings XML
  - FMP API:          historical daily close prices for all tickers + SPY

Reuses from existing modules:
  data_collector: TRACKED_WHALES, WHALE_TIERS, detect_signals,
                  _find_13f_holdings_doc, _parse_13f_xml
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_FMP_KEY  = os.getenv("FMP_API_KEY", "")
_FMP_BASE = "https://financialmodelingprep.com/api/v3"
_EDGAR_HEADERS = {
    "User-Agent":      "WhaleTracker research@whaletracker.ai",
    "Accept-Encoding": "gzip, deflate",
}

# Minimum conviction score thresholds (mirrors analysis_engine.py)
SCORE_THRESHOLDS = {"STRONG BUY": 6, "BUY": 3}

# Signal base scores (mirrors analysis_engine.py)
_SIG_SCORES = {
    "NEW_ENTRY":          3,
    "AGGRESSIVE_BUY":     4,
    "HIGH_CONCENTRATION": 2,
}
# Tier multipliers (mirrors analysis_engine.py)
_TIER_MULT = {1: 1.5, 2: 1.2, 3: 1.0}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticker:  str
    company: str
    action:  str    # "BUY" | "SELL"
    date:    str
    price:   float
    shares:  float
    value:   float
    signal:  str   = ""
    score:   float = 0.0


@dataclass
class BacktestResult:
    portfolio_series:  pd.Series              # date str → USD value
    benchmark_series:  pd.Series              # date str → SPY USD value
    trades:            list[Trade] = field(default_factory=list)
    metrics:           dict        = field(default_factory=dict)
    quarterly_log:     list[dict]  = field(default_factory=list)


# ── EDGAR historical 13F helpers ──────────────────────────────────────────────

def _edgar_submissions(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    try:
        r = requests.get(url, headers=_EDGAR_HEADERS, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("EDGAR submissions failed CIK %s: %s", cik, exc)
        return {}


def fetch_13f_accessions(cik: str, n: int = 14) -> list[dict]:
    """
    Return up to `n` most-recent 13F-HR filings (newest first).
    Each entry: {accession, filed_date, report_date}
    """
    data   = _edgar_submissions(cik)
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return []

    forms      = recent.get("form", [])
    dates      = recent.get("filingDate", [])
    periods    = recent.get("reportDate", [])
    accessions = recent.get("accessionNumber", [])

    out = []
    for i, form in enumerate(forms):
        if form.strip().upper() not in {"13F-HR", "13F-HR/A"}:
            continue
        out.append({
            "accession":   accessions[i],
            "filed_date":  dates[i],
            "report_date": periods[i],
        })
        if len(out) >= n:
            break
    return out  # newest first


# ── FMP historical price helpers ──────────────────────────────────────────────

def fetch_historical_prices(
    tickers:   list[str],
    from_date: str,
    to_date:   str,
) -> dict[str, pd.Series]:
    """
    Fetch daily EOD close prices for multiple tickers from FMP.
    Returns {TICKER: pd.Series(date_str → close)}, sorted ascending.
    """
    if not tickers or not _FMP_KEY:
        return {}

    result: dict[str, pd.Series] = {}

    def _parse_response(data: dict, fallback_sym: str) -> None:
        if "historicalStockList" in data:
            for item in data["historicalStockList"]:
                sym  = item.get("symbol", "").upper()
                hist = item.get("historical", [])
                if hist:
                    result[sym] = pd.Series(
                        {row["date"]: row["close"] for row in hist},
                        dtype=float,
                    ).sort_index()
        elif "historical" in data:
            hist = data["historical"]
            if hist:
                result[fallback_sym.upper()] = pd.Series(
                    {row["date"]: row["close"] for row in hist},
                    dtype=float,
                ).sort_index()

    # FMP batch: up to 5 symbols per call (conservative)
    batch_size = 5
    tickers_upper = [t.upper() for t in tickers]
    for i in range(0, len(tickers_upper), batch_size):
        batch   = tickers_upper[i:i + batch_size]
        symbols = ",".join(batch)
        try:
            r = requests.get(
                f"{_FMP_BASE}/historical-price-full/{symbols}",
                params={"from": from_date, "to": to_date, "apikey": _FMP_KEY},
                timeout=30,
            )
            r.raise_for_status()
            _parse_response(r.json(), batch[0])
        except Exception as exc:
            logger.warning("FMP historical prices failed (%s): %s", symbols, exc)
        if i + batch_size < len(tickers_upper):
            time.sleep(0.4)

    return result


def _price_on(prices: dict[str, pd.Series], ticker: str, date_str: str) -> float | None:
    """Closest available close price for `ticker` at or before `date_str`."""
    s = prices.get(ticker.upper())
    if s is None or s.empty:
        return None
    sub = s[s.index <= date_str]
    if sub.empty:
        sub = s
    return float(sub.iloc[-1])


# ── Signal aggregation ────────────────────────────────────────────────────────

def _score_holdings(holdings: list[dict], mult: float) -> dict[str, dict]:
    """holdings with signals → {ticker: {score, company}}"""
    out: dict[str, dict] = {}
    for h in holdings:
        sc = _SIG_SCORES.get(h.get("signal", ""), 0)
        if sc <= 0:
            continue
        t = h["ticker"]
        if t not in out:
            out[t] = {"score": 0.0, "company": h.get("company", t)}
        out[t]["score"] += sc * mult
    return out


def _aggregate_scores(
    whale_holdings: dict[str, list[dict]],
    whale_tiers:    dict,
) -> dict[str, dict]:
    """
    Combine per-whale scored holdings into a single {ticker: {score, company,
    whale_count, whales, recommendation}} dict.
    """
    agg: dict[str, dict] = {}
    for whale, holdings in whale_holdings.items():
        tier = whale_tiers.get(whale, {}).get("tier", 3)
        mult = _TIER_MULT.get(tier, 1.0)
        for ticker, info in _score_holdings(holdings, mult).items():
            if ticker not in agg:
                agg[ticker] = {"score": 0.0, "company": info["company"],
                               "whale_count": 0, "whales": []}
            agg[ticker]["score"]       += info["score"]
            agg[ticker]["whale_count"] += 1
            agg[ticker]["whales"].append(whale)

    for info in agg.values():
        sc = info["score"]
        wc = info["whale_count"]
        if sc >= 6 or (sc >= 4 and wc >= 2):
            info["recommendation"] = "STRONG BUY"
        elif sc >= 3:
            info["recommendation"] = "BUY"
        else:
            info["recommendation"] = "HOLD"
    return agg


# ── Main backtest engine ──────────────────────────────────────────────────────

def run_backtest(
    years:           int   = 3,
    initial_capital: float = 100_000.0,
    min_signal:      str   = "STRONG BUY",
    max_positions:   int   = 15,
) -> BacktestResult | None:
    """
    Simulate following WhaleTracker signals over `years` years.

    Returns BacktestResult on success, None on fatal error (no EDGAR data,
    no FMP price data).  All network errors are logged and skipped gracefully.
    """
    from src.data_collector import (  # noqa: PLC0415
        TRACKED_WHALES, WHALE_TIERS,
        _find_13f_holdings_doc, _parse_13f_xml, detect_signals,
    )

    logger.info("[backtest] %d-year sim · capital=%.0f · signal=%s",
                years, initial_capital, min_signal)

    n_q       = years * 4 + 2          # extra quarters for QoQ comparison
    end_date  = datetime.utcnow().date()
    start_dt  = end_date.replace(year=end_date.year - years)
    from_str  = start_dt.strftime("%Y-%m-%d")
    to_str    = end_date.strftime("%Y-%m-%d")
    min_score = SCORE_THRESHOLDS.get(min_signal, 3)

    # ── 1. Fetch 13F filing lists from EDGAR ─────────────────────────────────
    whale_acc: dict[str, list[dict]] = {}
    for whale, cik in TRACKED_WHALES.items():
        logger.info("[backtest]  → %s", whale)
        whale_acc[whale] = fetch_13f_accessions(cik, n=n_q)
        time.sleep(0.25)

    # ── 2. Build rebalancing events (one per unique quarter date) ─────────────
    # Collect all (filed_date, whale, cik, acc_info) for dates >= from_str
    events: list[tuple[str, str, str, dict]] = []
    for whale, accs in whale_acc.items():
        cik = TRACKED_WHALES[whale]
        for a in accs:
            if a["filed_date"] >= from_str:
                events.append((a["filed_date"], whale, cik, a))
    events.sort(key=lambda x: x[0])

    # Group into quarters: entries within 45 days of the first entry in group
    quarters: list[dict] = []  # [{date, members: [(whale, cik, acc_info)]}]
    for fd, whale, cik, acc_info in events:
        placed = False
        for q in quarters:
            d0 = datetime.strptime(q["date"], "%Y-%m-%d")
            d1 = datetime.strptime(fd, "%Y-%m-%d")
            if abs((d1 - d0).days) <= 45:
                q["members"].append((whale, cik, acc_info))
                placed = True
                break
        if not placed:
            quarters.append({"date": fd, "members": [(whale, cik, acc_info)]})
    quarters.sort(key=lambda q: q["date"])
    logger.info("[backtest] %d rebalancing quarters", len(quarters))
    if len(quarters) < 2:
        logger.warning("[backtest] Too few quarters — aborting.")
        return None

    # ── 3. Parse holdings for each quarter ───────────────────────────────────
    holdings_cache: dict[str, list[dict]] = {}

    def _get_holdings(cik: str, acc_info: dict) -> list[dict]:
        key = acc_info["accession"]
        if key in holdings_cache:
            return holdings_cache[key]
        cik_int = int(cik)
        url     = _find_13f_holdings_doc(cik_int, acc_info["accession"])
        raw     = _parse_13f_xml(url) if url else []
        time.sleep(0.3)
        holdings_cache[key] = raw
        return raw

    quarterly_scores: list[dict] = []   # [{date, scores}]

    for q in quarters:
        q_date   = q["date"]
        wh_hold: dict[str, list[dict]] = {}

        for whale, cik, acc_info in q["members"]:
            current = _get_holdings(cik, acc_info)
            if not current:
                continue
            # Find prior quarter for this whale
            whale_list = whale_acc.get(whale, [])
            idx = next(
                (i for i, a in enumerate(whale_list)
                 if a["accession"] == acc_info["accession"]),
                None,
            )
            prior: list[dict] = []
            if idx is not None and idx + 1 < len(whale_list):
                prior = _get_holdings(cik, whale_list[idx + 1])

            wh_hold[whale] = detect_signals(current, prior)

        if wh_hold:
            quarterly_scores.append({
                "date":   q_date,
                "scores": _aggregate_scores(wh_hold, WHALE_TIERS),
            })

    if not quarterly_scores:
        logger.error("[backtest] No signal data.")
        return None

    # ── 4. Collect all tickers that were ever a target ────────────────────────
    all_tickers: set[str] = {"SPY"}
    for qs in quarterly_scores:
        for ticker, info in qs["scores"].items():
            if info["score"] >= min_score:
                all_tickers.add(ticker)
    logger.info("[backtest] Fetching prices for %d tickers…", len(all_tickers))

    # ── 5. Fetch historical prices ────────────────────────────────────────────
    ticker_list = sorted(all_tickers)
    all_prices: dict[str, pd.Series] = {}
    for i in range(0, len(ticker_list), 5):
        chunk = ticker_list[i:i + 5]
        all_prices.update(fetch_historical_prices(chunk, from_str, to_str))
        if i + 5 < len(ticker_list):
            time.sleep(0.5)

    spy_series = all_prices.get("SPY", pd.Series(dtype=float))
    if spy_series.empty:
        logger.error("[backtest] SPY prices unavailable.")
        return None

    # ── 6. Simulate portfolio day by day ─────────────────────────────────────
    trading_days = spy_series.index.tolist()    # sorted date strings
    spy_start    = float(spy_series.iloc[0])
    spy_shares   = initial_capital / spy_start

    cash         = initial_capital
    positions:   dict[str, float] = {}   # ticker → shares
    cost_basis:  dict[str, float] = {}   # ticker → avg_cost
    trades:      list[Trade]      = []
    q_log:       list[dict]       = []

    port_vals:  dict[str, float] = {}
    bench_vals: dict[str, float] = {}

    q_idx = 0   # next quarter to process

    for day in trading_days:
        # ── Rebalance if a new filing date has arrived ────────────────────────
        while q_idx < len(quarterly_scores) and quarterly_scores[q_idx]["date"] <= day:
            qs     = quarterly_scores[q_idx]
            scores = qs["scores"]

            # Tickers meeting threshold, sorted by score desc, capped
            targets = sorted(
                [t for t, info in scores.items() if info["score"] >= min_score],
                key=lambda t: scores[t]["score"],
                reverse=True,
            )[:max_positions]
            target_set  = set(targets)
            current_set = set(positions.keys())

            # Compute total portfolio value before rebalancing
            port_now = cash + sum(
                positions[t] * (_price_on(all_prices, t, day) or cost_basis.get(t, 0))
                for t in positions
            )

            # Sell positions leaving the target set
            for ticker in current_set - target_set:
                px = _price_on(all_prices, ticker, day)
                if px and positions.get(ticker, 0) > 0:
                    shares = positions[ticker]
                    val    = shares * px
                    cash  += val
                    trades.append(Trade(
                        ticker  = ticker,
                        company = scores.get(ticker, {}).get("company", ticker),
                        action  = "SELL",
                        date    = day,
                        price   = px,
                        shares  = shares,
                        value   = val,
                    ))
                    del positions[ticker]
                    del cost_basis[ticker]

            # Determine equal-weight target per position
            n_targets = len(target_set)
            if n_targets == 0:
                q_idx += 1
                continue
            port_now  = cash + sum(
                positions[t] * (_price_on(all_prices, t, day) or cost_basis.get(t, 0))
                for t in positions
            )
            target_val = port_now / n_targets

            # Rebalance existing / enter new positions
            for ticker in targets:
                px = _price_on(all_prices, ticker, day)
                if not px or px <= 0:
                    continue
                if ticker in positions:
                    # Trim or add to existing position
                    current_val = positions[ticker] * px
                    diff        = target_val - current_val
                    if abs(diff) < 50:    # $50 minimum rebalance band
                        continue
                    adj_shares  = diff / px
                    positions[ticker] += adj_shares
                    cash              -= diff
                else:
                    # New position
                    shares = target_val / px
                    if cash < target_val * 0.95:
                        # Not enough cash — use what we have, split evenly
                        shares = max(cash * 0.9 / px / max(len(target_set - current_set), 1), 0)
                    if shares <= 0:
                        continue
                    actual_cost        = shares * px
                    positions[ticker]  = shares
                    cost_basis[ticker] = px
                    cash              -= actual_cost
                    info               = scores.get(ticker, {})
                    trades.append(Trade(
                        ticker  = ticker,
                        company = info.get("company", ticker),
                        action  = "BUY",
                        date    = day,
                        price   = px,
                        shares  = shares,
                        value   = actual_cost,
                        signal  = info.get("recommendation", ""),
                        score   = info.get("score", 0.0),
                    ))

            q_log.append({
                "date":     qs["date"],
                "holdings": targets,
                "n":        len(targets),
            })
            q_idx += 1

        # ── Mark-to-market ────────────────────────────────────────────────────
        port_val = max(cash, 0) + sum(
            positions.get(t, 0) * (_price_on(all_prices, t, day) or 0)
            for t in positions
        )
        spy_px   = spy_series.get(day, spy_start)
        port_vals[day]  = port_val
        bench_vals[day] = spy_px / spy_start * initial_capital

    # ── 7. Calculate metrics ──────────────────────────────────────────────────
    port_s  = pd.Series(port_vals,  dtype=float).sort_index()
    bench_s = pd.Series(bench_vals, dtype=float).sort_index()
    metrics = _calc_metrics(port_s, bench_s, initial_capital)
    metrics["n_trades"] = len(trades)
    logger.info("[backtest] Done — return %.1f%%, vs SPY %.1f%%",
                metrics.get("total_return_pct", 0),
                metrics.get("benchmark_return_pct", 0))

    return BacktestResult(
        portfolio_series = port_s,
        benchmark_series = bench_s,
        trades           = trades,
        metrics          = metrics,
        quarterly_log    = q_log,
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def _calc_metrics(
    port:    pd.Series,
    bench:   pd.Series,
    capital: float,
    rf:      float = 0.05,
) -> dict:
    if port.empty or capital == 0:
        return {}

    total_ret  = (port.iloc[-1] - capital) / capital
    bench_ret  = (bench.iloc[-1] - capital) / capital

    try:
        d0    = datetime.strptime(str(port.index[0]),  "%Y-%m-%d")
        d1    = datetime.strptime(str(port.index[-1]), "%Y-%m-%d")
        days  = max((d1 - d0).days, 1)
    except Exception:
        days  = 365
    years     = days / 365.25

    ann_ret   = (1 + total_ret) ** (1 / years) - 1 if years > 0 else total_ret
    daily_ret = port.pct_change().dropna()
    ann_std   = daily_ret.std() * (252 ** 0.5) if not daily_ret.empty else 0.01
    sharpe    = (ann_ret - rf) / ann_std if ann_std > 0 else 0.0

    # Max drawdown
    roll_max  = port.cummax()
    drawdown  = (port - roll_max) / roll_max
    max_dd    = float(drawdown.min())

    # Monthly win rate
    try:
        port_dt = port.copy()
        port_dt.index = pd.to_datetime(port_dt.index)
        monthly   = port_dt.resample("ME").last().pct_change().dropna()
        win_rate  = float((monthly > 0).sum() / len(monthly)) if len(monthly) > 0 else 0.5
    except Exception:
        win_rate = 0.5

    return {
        "total_return_pct":      round(total_ret  * 100, 2),
        "benchmark_return_pct":  round(bench_ret  * 100, 2),
        "alpha_pct":             round((total_ret - bench_ret) * 100, 2),
        "annualized_return_pct": round(ann_ret    * 100, 2),
        "max_drawdown_pct":      round(max_dd     * 100, 2),
        "sharpe_ratio":          round(sharpe,           2),
        "win_rate_pct":          round(win_rate   * 100, 1),
        "final_value":           round(float(port.iloc[-1]), 2),
        "n_trades":              0,
    }
