"""
scheduler.py
------------
Background APScheduler job that periodically refreshes SEC filing data
and dispatches Slack alerts when meaningful signals change.

Environment variables (all optional):
  REFRESH_INTERVAL_HOURS  — how often to re-fetch data (default: 4)
  DAILY_DIGEST_HOUR       — UTC hour to send morning digest (default: 8)
  ALERT_WATCHLIST         — comma-separated tickers to watch, e.g. "AAPL,NVDA"
  ALERT_MIN_SCORE         — minimum score delta to trigger watchlist alert (default: 2)
  FORM4_WATCH_TICKERS     — comma-separated tickers for real-time Form 4 polling
                            (falls back to ALERT_WATCHLIST if not set)
  FORM4_REFRESH_MINUTES   — Form 4 polling interval in minutes (default: 30)
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

REFRESH_INTERVAL_HOURS  = int(os.getenv("REFRESH_INTERVAL_HOURS", "4"))
DAILY_DIGEST_HOUR       = int(os.getenv("DAILY_DIGEST_HOUR", "8"))
DAILY_NEWS_HOUR         = int(os.getenv("DAILY_NEWS_HOUR", "8"))   # UTC hour for morning news alert
EVENT_CHECK_HOUR        = int(os.getenv("EVENT_CHECK_HOUR", "7"))  # UTC hour for market event pre-alerts
ALERT_MIN_SCORE         = int(os.getenv("ALERT_MIN_SCORE", "2"))
FORM4_REFRESH_MINUTES   = int(os.getenv("FORM4_REFRESH_MINUTES", "30"))

_WATCHLIST_RAW = os.getenv("ALERT_WATCHLIST", "")
ALERT_WATCHLIST: set[str] = {
    t.strip().upper() for t in _WATCHLIST_RAW.split(",") if t.strip()
}

# Form 4 watch list: separate env var, falls back to ALERT_WATCHLIST
_F4_RAW = os.getenv("FORM4_WATCH_TICKERS", "") or _WATCHLIST_RAW
FORM4_WATCH_TICKERS: list[str] = [
    t.strip().upper() for t in _F4_RAW.split(",") if t.strip()
]

# In-memory snapshot: {ticker: {"recommendation": str, "conviction_score": int}}
_snapshot: dict[str, dict[str, Any]] = {}
_last_insider_alert: dict[str, datetime] = {}   # ticker → last alerted time
_form4_seen_accessions: set[str] = set()         # dedup set for real-time Form 4 alerts
_activist_seen_tickers: set[str] = set()         # dedup set for 13D/G alerts
_INSIDER_CLUSTER_MIN  = 3       # insiders that must sell within window
_INSIDER_CLUSTER_DAYS = 30

# ── Batch buffer for low-urgency signals ─────────────────────────────────────
# Items: {"type": "watchlist"|"13g", "data": dict, ...}
# Flushed once per day by _flush_batch_job (10:00 KST = 01:00 UTC).
_batch_buffer: list[dict[str, Any]] = []
_batch_seen: set[str] = set()   # dedup key per item

_scheduler = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start(app_data: dict[str, Any]) -> None:
    """Initialise and start the APScheduler BackgroundScheduler.

    Args:
        app_data: dict with keys:
            filings, activist, insiders, nport,
            recommendations (initial list), rebalancing (initial list)
    """
    global _scheduler
    if _scheduler is not None:
        logger.warning("Scheduler already running — skipping start()")
        return

    try:
        from apscheduler.schedulers.background import BackgroundScheduler  # noqa: PLC0415
    except ImportError:
        logger.error(
            "apscheduler is not installed. "
            "Run: pip install 'apscheduler==3.10.4'"
        )
        return

    # Seed snapshot with initial recommendations so we only alert on *changes*
    _seed_snapshot(app_data.get("recommendations", []))

    _scheduler = BackgroundScheduler(timezone="UTC")

    # ── Periodic refresh job ────────────────────────────────────────────────
    _scheduler.add_job(
        _refresh_and_alert,
        trigger="interval",
        hours=REFRESH_INTERVAL_HOURS,
        id="whale_refresh",
        replace_existing=True,
        kwargs={"app_data": app_data},
    )

    # ── Daily digest job ────────────────────────────────────────────────────
    _scheduler.add_job(
        _daily_digest_job,
        trigger="cron",
        hour=DAILY_DIGEST_HOUR,
        minute=0,
        id="daily_digest",
        replace_existing=True,
        kwargs={"app_data": app_data},
    )

    # ── Daily news alert job ────────────────────────────────────────────────
    # Runs every hour at :10; the job itself checks the user's preferred hour
    # from daily_news_sub.json so no restart is needed when settings change.
    _scheduler.add_job(
        _daily_news_job,
        trigger="cron",
        hour="*",
        minute=10,
        id="daily_news",
        replace_existing=True,
    )

    # ── Market event pre-alert job ──────────────────────────────────────────
    _scheduler.add_job(
        _market_event_job,
        trigger="cron",
        hour=EVENT_CHECK_HOUR,
        minute=20,
        id="market_events",
        replace_existing=True,
    )

    # ── Real-time Form 4 polling (runs more frequently than full refresh) ───
    # Always registered — uses FORM4_WATCH_TICKERS env var if set,
    # otherwise falls back to all tickers in current whale holdings.
    _scheduler.add_job(
        _realtime_form4_job,
        trigger="interval",
        minutes=FORM4_REFRESH_MINUTES,
        id="form4_realtime",
        replace_existing=True,
        kwargs={"app_data": app_data},
    )
    logger.info(
        "Form 4 real-time polling — every %d min (watch list: %s)",
        FORM4_REFRESH_MINUTES,
        ", ".join(FORM4_WATCH_TICKERS[:5]) + ("…" if len(FORM4_WATCH_TICKERS) > 5 else "")
        if FORM4_WATCH_TICKERS else "whale holdings (auto)",
    )

    # ── Batch digest job (10:00 KST = 01:00 UTC) ─────────────────────────────
    # Flushes _batch_buffer: watchlist score changes, 13G filings.
    _scheduler.add_job(
        _flush_batch_job,
        trigger="cron",
        hour=1,
        minute=0,
        id="batch_digest",
        replace_existing=True,
    )

    # ── Pre-market briefing (06:00 KST = 21:00 UTC previous day) ─────────────
    # Combines top signals + upcoming events + institutional news headlines.
    _scheduler.add_job(
        _premarket_briefing_job,
        trigger="cron",
        hour=21,
        minute=0,
        id="premarket_briefing",
        replace_existing=True,
        kwargs={"app_data": app_data},
    )

    _scheduler.start()
    logger.info(
        "Scheduler started — refresh every %dh, digest at %02d:00 UTC",
        REFRESH_INTERVAL_HOURS, DAILY_DIGEST_HOUR,
    )


def stop() -> None:
    """Gracefully shut down the scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler stopped")


def is_running() -> bool:
    return _scheduler is not None and _scheduler.running


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def _refresh_and_alert(app_data: dict[str, Any]) -> None:
    """Fetch fresh data and fire alerts for any meaningful changes."""
    logger.info("[scheduler] Running data refresh at %s UTC",
                datetime.now(timezone.utc).strftime("%H:%M"))

    try:
        from src.data_collector import (  # noqa: PLC0415
            fetch_all_whale_filings,
            fetch_13dg_filings,
            fetch_form4_filings,
            fetch_nport_filings,
            WHALE_TIERS,
        )
        from src.analysis_engine import (  # noqa: PLC0415
            build_recommendations,
            get_sector_rotation_signals,
        )
        from src.portfolio_manager import suggest_rebalancing  # noqa: PLC0415
    except Exception as exc:
        logger.error("[scheduler] Import error: %s", exc)
        return

    # ── Fetch ────────────────────────────────────────────────────────────────
    try:
        filings  = fetch_all_whale_filings()
        tickers  = list({h["ticker"] for holds in filings.values() for h in holds})
        activist = fetch_13dg_filings()
        insiders = fetch_form4_filings(tickers)
        nport    = fetch_nport_filings()
    except Exception as exc:
        logger.error("[scheduler] Data fetch error: %s", exc)
        return

    # ── Analyse ──────────────────────────────────────────────────────────────
    try:
        new_recs = build_recommendations(
            filings,
            activist_filings=activist,
            insider_filings=insiders,
            nport_filings=nport,
        )
        rotation = get_sector_rotation_signals(filings)
    except Exception as exc:
        logger.error("[scheduler] Analysis error: %s", exc)
        return

    # ── Update app_data in-place so the dashboard picks up fresh results ─────
    app_data["filings"]   = filings
    app_data["activist"]  = activist
    app_data["insiders"]  = insiders
    app_data["nport"]     = nport
    app_data["recommendations"] = new_recs
    app_data["rebalancing"] = suggest_rebalancing(
        app_data.get("portfolio", {}), rotation
    )

    # ── Alert checks ─────────────────────────────────────────────────────────
    _check_strong_buy_alerts(new_recs)
    _check_tier1_alerts(filings, WHALE_TIERS, new_recs)
    _check_watchlist_alerts(new_recs)          # → batch buffer
    _check_activist_alerts(activist)           # 13D → real-time, 13G → batch
    _check_insider_cluster_alerts(insiders)

    # Refresh snapshot
    _seed_snapshot(new_recs)
    logger.info("[scheduler] Refresh complete — %d recommendations", len(new_recs))


_MIN_VALUE_BUY  = 100_000   # alert on insider buys ≥ $100K
_MIN_VALUE_SELL = 500_000   # alert on insider sells ≥ $500K (noise filter)


def _realtime_form4_job(app_data: dict[str, Any] | None = None) -> None:
    """Poll EDGAR for new Form 4 filings and fire per-transaction Slack alerts.

    Watch-ticker priority:
      1. FORM4_WATCH_TICKERS env var (explicit list)
      2. ALERT_WATCHLIST env var
      3. All tickers currently held by tracked whales (auto, capped at 60)

    Signal filters applied:
      • PLANNED_SELL (10b5-1 pre-arranged) → skipped (low informational value)
      • INSIDER_BUY  → alert if value ≥ $100K
      • INSIDER_SELL → alert if value ≥ $500K
    """
    # Resolve effective watch list
    effective: list[str] = FORM4_WATCH_TICKERS
    if not effective and app_data:
        effective = sorted({
            h.get("ticker", "")
            for holds in app_data.get("filings", {}).values()
            for h in holds
            if h.get("ticker")
        })[:60]   # cap to avoid excessive EDGAR requests

    if not effective:
        logger.debug("[scheduler] Form 4 poll skipped — no tickers to watch")
        return

    logger.info("[scheduler] Form 4 poll — %d tickers", len(effective))
    try:
        from src.data_collector import fetch_recent_form4_filings  # noqa: PLC0415
        from src.slack_notifier import send_form4_realtime_alert   # noqa: PLC0415
    except Exception as exc:
        logger.error("[scheduler] Form 4 import error: %s", exc)
        return

    try:
        txs = fetch_recent_form4_filings(effective, hours_back=2)
    except Exception as exc:
        logger.error("[scheduler] Form 4 fetch error: %s", exc)
        return

    new_alerts = 0
    for tx in txs:
        acc = tx.get("accession", "")
        if not acc or acc in _form4_seen_accessions:
            continue
        _form4_seen_accessions.add(acc)

        signal    = tx.get("signal", "")
        value_usd = tx.get("value_usd", 0.0)
        is_10b51  = tx.get("is_10b51", False)

        # Skip pre-planned 10b5-1 sales — lower informational value
        if signal == "PLANNED_SELL" or is_10b51:
            continue
        if signal == "INSIDER_BUY" and value_usd < _MIN_VALUE_BUY:
            continue
        if signal == "INSIDER_SELL" and value_usd < _MIN_VALUE_SELL:
            continue
        if signal not in {"INSIDER_BUY", "INSIDER_SELL"}:
            continue

        ticker = tx.get("ticker", "")
        news   = _fetch_ticker_news(ticker, 3) if signal == "INSIDER_BUY" else []
        try:
            send_form4_realtime_alert(
                ticker    = ticker,
                company   = tx.get("company", ""),
                insider   = tx.get("insider", "Insider"),
                role      = tx.get("role", ""),
                signal    = signal,
                shares    = tx.get("shares", 0),
                value_usd = value_usd,
                is_10b51  = False,
                news_items = news or None,
            )
            new_alerts += 1
        except Exception as exc:
            logger.error("[scheduler] Form 4 alert send failed (%s): %s", ticker, exc)

    logger.info("[scheduler] Form 4 poll complete — %d new alerts", new_alerts)


def _read_news_settings() -> dict:
    """Return the full daily-news settings dict.

    Prefers the shared in-memory state (updated from browser localStorage on
    every page load) over direct file reads so Railway redeploys don't silently
    disable alerts.  Falls back to the JSON file then defaults.
    """
    try:
        from src.news_settings import get as _ns_get  # noqa: PLC0415
        settings = _ns_get()
        if settings:
            return settings
    except Exception as exc:
        logger.debug("[scheduler] news_settings import failed: %s", exc)

    # File fallback (may not exist after a Railway redeploy)
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415
    sub_file = Path(__file__).parent.parent / "daily_news_sub.json"
    defaults = {"enabled": False, "hour": DAILY_NEWS_HOUR, "topics": []}
    try:
        if sub_file.exists():
            data = json.loads(sub_file.read_text())
            return {**defaults, **data}
    except Exception:
        pass
    return defaults


def _summarize_articles_ko(articles: list[dict]) -> list[dict]:
    """Add 'summary_ko' and 'sector_impact' fields to each article using Claude Sonnet.

    Each article gets:
      summary_ko    — 2-3 sentence Korean summary focused on institutional implication
      sector_impact — dict with keys 'sector' (str) and 'direction' ('positive'|'negative'|'neutral')

    Falls back gracefully — articles without summaries are returned as-is.
    All articles are summarized in a single API call to minimize latency.
    """
    if not articles:
        return articles

    headlines = [a.get("headline", "") for a in articles if a.get("headline")]
    if not headlines:
        return articles

    try:
        import anthropic  # noqa: PLC0415
        import re          # noqa: PLC0415
        import json as _json  # noqa: PLC0415
        client = anthropic.Anthropic()

        numbered = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines))
        prompt = (
            "당신은 기관투자자 동향을 분석하는 전문 애널리스트입니다.\n"
            "아래 영문 뉴스 헤드라인들을 분석해서 JSON 배열로만 응답해줘. "
            "다른 텍스트나 마크다운 없이 JSON만 출력해줘.\n\n"
            "각 항목 형식:\n"
            "{\n"
            '  "idx": 1,\n'
            '  "summary_ko": "2~3문장 한국어 요약. 기관투자자 움직임의 배경과 시장 시사점 포함.",\n'
            '  "sector": "영향받는 섹터 (Technology/Healthcare/Financials/Energy/Consumer Discretionary/'
            'Consumer Staples/Industrials/Materials/Real Estate/Communication Services/Utilities/Macro 중 하나)",\n'
            '  "direction": "positive 또는 negative 또는 neutral"\n'
            "}\n\n"
            f"헤드라인:\n{numbered}"
        )

        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip() if msg.content else ""

        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

        parsed: list[dict] = _json.loads(raw)
        impact_map: dict[int, dict] = {int(item["idx"]): item for item in parsed}

        result = []
        headline_idx = 0
        for article in articles:
            if article.get("headline"):
                headline_idx += 1
                item = impact_map.get(headline_idx, {})
                enriched = dict(article)
                enriched["summary_ko"]    = item.get("summary_ko", "")
                enriched["sector_impact"] = {
                    "sector":    item.get("sector", ""),
                    "direction": item.get("direction", "neutral"),
                }
                result.append(enriched)
            else:
                result.append(article)
        return result

    except ImportError:
        logger.debug("[scheduler] anthropic SDK not installed — skipping Korean summaries")
    except Exception as exc:
        logger.warning("[scheduler] Korean summary generation failed: %s", exc)

    return articles


def _daily_news_job() -> None:
    """Send institutional investor news to Slack if subscription is ON
    and the current UTC hour matches the user-configured delivery hour.

    Fetches articles focused on institutional investor activity (hedge funds,
    13F filings, activist investors, etc.) and adds Korean summaries via Claude.
    """
    settings = _read_news_settings()

    if not settings.get("enabled"):
        logger.info("[scheduler] Daily news subscription disabled — skipping")
        return

    # Check if current UTC hour matches the user's chosen delivery hour.
    # Settings may store "hour_utc" (new) or legacy "hour" (old, assumed UTC).
    current_hour = datetime.now(timezone.utc).hour
    target_hour  = int(settings.get("hour_utc", settings.get("hour", DAILY_NEWS_HOUR)))
    if current_hour != target_hour:
        logger.debug(
            "[scheduler] Daily news: current hour %d ≠ target %d — skipping",
            current_hour, target_hour,
        )
        return

    try:
        from src.news_collector import fetch_institutional_news, item_matches_topics  # noqa: PLC0415
        from src.slack_notifier import send_daily_news_alert                          # noqa: PLC0415
    except Exception as exc:
        logger.error("[scheduler] Daily news import error: %s", exc)
        return

    try:
        topics = settings.get("topics") or []
        # Fetch institutional investor news pool
        candidates = fetch_institutional_news(20)
        if topics:
            candidates = [n for n in candidates if item_matches_topics(n["headline"], topics)]

        if not candidates:
            logger.info("[scheduler] No institutional news for topics %s — skipping", topics)
            return

        # Take top 5 articles and add Korean summaries
        top = candidates[:5]
        top = _summarize_articles_ko(top)

        send_daily_news_alert(top)
        logger.info(
            "[scheduler] Institutional news alert sent — %d articles (topics=%s)",
            len(top), topics or "all",
        )
    except Exception as exc:
        logger.error("[scheduler] Daily news job error: %s", exc)


def _market_event_job() -> None:
    """Check whether any scheduled market event needs a pre-alert today.

    Fires send_market_event_alert() for every event whose date is exactly
    EVENT_ALERT_DAYS_BEFORE days away (default: 7 days and 1 day before).
    """
    try:
        from src.market_events import get_events_due_for_alert        # noqa: PLC0415
        from src.slack_notifier import send_market_event_alert        # noqa: PLC0415
    except Exception as exc:
        logger.error("[scheduler] Market events import error: %s", exc)
        return

    try:
        due = get_events_due_for_alert()
    except Exception as exc:
        logger.error("[scheduler] Market events check error: %s", exc)
        return

    if not due:
        logger.info("[scheduler] No market event alerts due today")
        return

    for event, days_before in due:
        try:
            send_market_event_alert(event, days_before)
            logger.info(
                "[scheduler] Market event alert sent: %s (%d days before)",
                event.get("title", ""), days_before,
            )
        except Exception as exc:
            logger.error(
                "[scheduler] Market event alert failed (%s): %s",
                event.get("title", ""), exc,
            )


def _daily_digest_job(app_data: dict[str, Any]) -> None:
    """Send the morning digest Slack message, including market news headlines."""
    from src.slack_notifier import send_daily_digest  # noqa: PLC0415

    recs        = app_data.get("recommendations", [])
    rebalancing = app_data.get("rebalancing", [])
    top_recs    = [r for r in recs
                   if r.get("recommendation") in {"STRONG BUY", "BUY"}][:5]

    # Fetch fresh market news headlines for the digest
    news: list[dict] | None = None
    try:
        from src.news_collector import fetch_market_news  # noqa: PLC0415
        news = fetch_market_news(3)
    except Exception:
        pass  # news is optional — digest still sends without it

    if top_recs:
        send_daily_digest(top_recs, rebalancing or None, news=news)
        logger.info("[scheduler] Daily digest sent (%d signals)", len(top_recs))
    else:
        logger.info("[scheduler] No BUY/STRONG BUY signals — digest skipped")


# ---------------------------------------------------------------------------
# Alert helpers
# ---------------------------------------------------------------------------

def _fetch_ticker_news(ticker: str, n: int = 3) -> list[dict]:
    """Fetch up to `n` recent news items for `ticker`. Never raises."""
    try:
        from src.news_collector import search_ticker_news  # noqa: PLC0415
        return search_ticker_news(ticker, n)
    except Exception as exc:
        logger.debug("[scheduler] Ticker news fetch failed (%s): %s", ticker, exc)
        return []


def _generate_ai_context(
    ticker: str,
    company: str,
    whale_name: str,
    whale_style: str,
    filing_type: str,
) -> str:
    """Generate 2-3 sentence Korean investment context using Claude Sonnet.

    Used only for high-value alerts: Tier 1 new entries and 13D filings.
    Returns empty string if API unavailable or on any error.
    """
    try:
        import anthropic  # noqa: PLC0415
        client = anthropic.Anthropic()

        prompt = (
            f"당신은 기관투자자 동향 전문 애널리스트입니다.\n"
            f"다음 내용을 바탕으로 한국어로 2~3문장의 투자 시사점을 작성해줘.\n"
            f"- 기관투자자: {whale_name} ({whale_style} 스타일)\n"
            f"- 종목: {ticker} ({company})\n"
            f"- 공시 유형: {filing_type}\n\n"
            "① 이 펀드가 이 종목을 매수할 가능성 있는 투자 thesis, "
            "② 주목할 리스크 또는 기회, "
            "③ 단기 주가에 미칠 영향을 각 1문장씩 포함해줘. "
            "과도한 수식어 없이 간결하게. 번호나 머리말 없이 바로 본문만 출력해줘."
        )
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip() if msg.content else ""
    except ImportError:
        logger.debug("[scheduler] anthropic SDK not installed — no AI context")
    except Exception as exc:
        logger.warning("[scheduler] AI context generation failed (%s): %s", ticker, exc)
    return ""


def _check_strong_buy_alerts(new_recs: list[dict]) -> None:
    """Fire send_strong_buy_alert() for any ticker newly reaching STRONG BUY."""
    from src.slack_notifier import send_strong_buy_alert  # noqa: PLC0415

    for rec in new_recs:
        ticker = rec["ticker"]
        if rec.get("recommendation") != "STRONG BUY":
            continue
        prev = _snapshot.get(ticker, {})
        if prev.get("recommendation") == "STRONG BUY":
            continue  # Already alerted this cycle

        logger.info("[scheduler] New STRONG BUY: %s", ticker)
        news = _fetch_ticker_news(ticker, 3)
        try:
            send_strong_buy_alert(rec, news_items=news or None)
        except Exception as exc:
            logger.error("[scheduler] Alert send failed (%s): %s", ticker, exc)


def _check_tier1_alerts(
    filings: dict,
    whale_tiers: dict,
    new_recs: list[dict],
) -> None:
    """Fire send_tier1_entry_alert() for any new Tier 1 NEW_ENTRY signal.

    Enriched with:
      - 3 recent news items for the ticker
      - AI-generated investment context (Task 4)
    """
    from src.slack_notifier import send_tier1_entry_alert  # noqa: PLC0415

    tier1_whales = {
        name for name, info in whale_tiers.items() if info.get("tier") == 1
    }
    rec_map = {r["ticker"]: r for r in new_recs}

    for whale_name in tier1_whales:
        holdings = filings.get(whale_name, [])
        tier_info = whale_tiers.get(whale_name, {})
        tier_label = tier_info.get("label", "T1")
        whale_style = tier_info.get("style", "")

        for holding in holdings:
            if holding.get("signal") != "NEW_ENTRY":
                continue
            ticker = holding.get("ticker", "")
            if not ticker:
                continue

            # Only alert if this is genuinely new vs snapshot
            prev = _snapshot.get(ticker, {})
            if whale_name in prev.get("whales_seen", set()):
                continue

            rec = rec_map.get(ticker, {})
            company = rec.get("company", "") or holding.get("company", "")
            score   = rec.get("conviction_score", 0)

            # Fetch ticker news + generate AI context (Task 4)
            news       = _fetch_ticker_news(ticker, 3)
            ai_context = _generate_ai_context(
                ticker=ticker,
                company=company,
                whale_name=whale_name,
                whale_style=whale_style,
                filing_type="13F NEW_ENTRY",
            )

            logger.info("[scheduler] Tier 1 entry: %s → %s", whale_name, ticker)
            try:
                send_tier1_entry_alert(
                    whale_name=whale_name,
                    tier_label=tier_label,
                    ticker=ticker,
                    company=company,
                    signal="NEW_ENTRY",
                    score=score,
                    news_items=news or None,
                    ai_context=ai_context or None,
                )
            except Exception as exc:
                logger.error("[scheduler] T1 alert failed (%s): %s", ticker, exc)


def _check_watchlist_alerts(new_recs: list[dict]) -> None:
    """Queue watchlist score changes into the batch buffer (low urgency).

    Previously sent immediately; now batched and flushed once per day at 10:00 KST.
    """
    if not ALERT_WATCHLIST:
        return

    for rec in new_recs:
        ticker = rec["ticker"]
        if ticker not in ALERT_WATCHLIST:
            continue
        prev_score = _snapshot.get(ticker, {}).get("conviction_score", 0)
        curr_score = rec.get("conviction_score", 0)
        if abs(curr_score - prev_score) < ALERT_MIN_SCORE:
            continue

        dedup_key = f"watchlist:{ticker}:{prev_score}:{curr_score}"
        if dedup_key in _batch_seen:
            continue
        _batch_seen.add(dedup_key)
        _batch_buffer.append({"type": "watchlist", "data": rec, "old_score": prev_score})
        logger.info(
            "[scheduler] Watchlist queued (batch): %s score %d→%d",
            ticker, prev_score, curr_score,
        )


def _check_activist_alerts(activist: dict) -> None:
    """Route SC 13D/G filings: 13D → real-time alert, 13G → batch buffer.

    13D = activist intent (경영 참여 의도) → high urgency, send immediately
    13G = passive stake (단순 투자)       → low urgency, batch daily
    """
    if not activist:
        return

    from src.slack_notifier import send_activist_13d_alert  # noqa: PLC0415

    for ticker, filings_list in activist.items():
        if isinstance(filings_list, dict):
            filings_list = [filings_list]
        for filing in (filings_list or []):
            signal    = filing.get("signal", "")
            form_type = filing.get("form_type", "")
            filer     = filing.get("filer", filing.get("whale_name", ""))
            ownership = filing.get("ownership_pct", 0.0)
            company   = filing.get("company", "")

            dedup_key = f"{form_type}:{ticker}:{filer}"
            if dedup_key in _activist_seen_tickers:
                continue

            if signal == "ACTIVIST_STAKE" or "13D" in form_type:
                # ── Real-time: 13D activist alert ─────────────────────────
                _activist_seen_tickers.add(dedup_key)
                news       = _fetch_ticker_news(ticker, 3)
                ai_context = _generate_ai_context(
                    ticker=ticker,
                    company=company,
                    whale_name=filer,
                    whale_style="Activist",
                    filing_type="SC 13D",
                )
                logger.info("[scheduler] 13D activist alert: %s → %s", filer, ticker)
                try:
                    send_activist_13d_alert(
                        ticker=ticker,
                        company=company,
                        filer=filer,
                        ownership_pct=ownership,
                        ai_context=ai_context or None,
                        news_items=news or None,
                    )
                except Exception as exc:
                    logger.error("[scheduler] 13D alert failed (%s): %s", ticker, exc)

            elif signal == "LARGE_PASSIVE_STAKE" or "13G" in form_type:
                # ── Batch: 13G passive stake ───────────────────────────────
                _activist_seen_tickers.add(dedup_key)
                _batch_buffer.append({"type": "13g", "data": filing})
                logger.info("[scheduler] 13G passive queued (batch): %s → %s", filer, ticker)


def _flush_batch_job() -> None:
    """Flush _batch_buffer → send_batch_digest(). Runs daily at 10:00 KST (01:00 UTC)."""
    if not _batch_buffer:
        logger.info("[scheduler] Batch buffer empty — skipping digest")
        return

    items = list(_batch_buffer)
    _batch_buffer.clear()

    logger.info("[scheduler] Flushing batch digest — %d items", len(items))
    try:
        from src.slack_notifier import send_batch_digest  # noqa: PLC0415
        send_batch_digest(items)
    except Exception as exc:
        logger.error("[scheduler] Batch digest failed: %s", exc)


def _premarket_briefing_job(app_data: dict[str, Any]) -> None:
    """Send pre-market briefing at 06:00 KST (21:00 UTC previous evening).

    Combines:
      - Top BUY / STRONG BUY institutional signals
      - Market events in the next 7 days
      - Institutional investor news headlines (with Korean summaries)
    """
    try:
        from src.slack_notifier import send_premarket_briefing  # noqa: PLC0415
    except Exception as exc:
        logger.error("[scheduler] Premarket briefing import error: %s", exc)
        return

    # ── Top signals ───────────────────────────────────────────────────────────
    recs     = app_data.get("recommendations", [])
    top_recs = [r for r in recs if r.get("recommendation") in {"STRONG BUY", "BUY"}][:5]

    # ── Upcoming events (next 7 days) ─────────────────────────────────────────
    events: list[dict] = []
    try:
        from src.market_events import get_upcoming_events  # noqa: PLC0415
        events = get_upcoming_events(days_ahead=7)
    except Exception as exc:
        logger.debug("[scheduler] Premarket events fetch failed: %s", exc)

    # ── Institutional news with Korean summaries ──────────────────────────────
    news: list[dict] = []
    try:
        from src.news_collector import fetch_institutional_news  # noqa: PLC0415
        raw_news = fetch_institutional_news(5)
        news     = _summarize_articles_ko(raw_news[:3])
    except Exception as exc:
        logger.debug("[scheduler] Premarket news fetch failed: %s", exc)

    if not top_recs and not events and not news:
        logger.info("[scheduler] Premarket briefing: no content to send — skipping")
        return

    try:
        send_premarket_briefing(top_recs=top_recs, events=events, news_items=news)
        logger.info(
            "[scheduler] Premarket briefing sent — %d signals, %d events, %d news",
            len(top_recs), len(events), len(news),
        )
    except Exception as exc:
        logger.error("[scheduler] Premarket briefing send failed: %s", exc)


def _check_insider_cluster_alerts(insiders: dict[str, list[dict]]) -> None:
    """Fire send_insider_cluster_alert() when 3+ insiders sell the same ticker
    within INSIDER_CLUSTER_DAYS days, and we haven't already alerted this ticker
    in the current cycle."""
    from src.slack_notifier import send_insider_cluster_alert  # noqa: PLC0415

    now = datetime.now(timezone.utc)

    for ticker, transactions in insiders.items():
        # Filter to sell transactions within the window
        sells = [
            t for t in transactions
            if t.get("signal") in {"INSIDER_SELL", "PLANNED_SELL"}
            and _days_ago(t.get("filed_date", ""), now) <= _INSIDER_CLUSTER_DAYS
        ]
        if len(sells) < _INSIDER_CLUSTER_MIN:
            continue

        # Cooldown: don't re-alert the same ticker within CLUSTER_DAYS
        last = _last_insider_alert.get(ticker)
        if last and (now - last).days < _INSIDER_CLUSTER_DAYS:
            continue

        total_value = sum(t.get("value_usd", 0) for t in sells)
        company = sells[0].get("company", "") if sells else ""

        logger.info(
            "[scheduler] Insider cluster: %s — %d sellers, $%.1fM",
            ticker, len(sells), total_value / 1e6,
        )
        try:
            send_insider_cluster_alert(
                ticker=ticker,
                company=company,
                insider_count=len(sells),
                total_value_usd=total_value,
            )
            _last_insider_alert[ticker] = now
        except Exception as exc:
            logger.error("[scheduler] Insider cluster alert failed (%s): %s", ticker, exc)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _seed_snapshot(recs: list[dict]) -> None:
    """Update _snapshot with current recommendation state."""
    for rec in recs:
        ticker = rec["ticker"]
        existing = _snapshot.get(ticker, {})
        _snapshot[ticker] = {
            "recommendation":  rec.get("recommendation", "HOLD"),
            "conviction_score": rec.get("conviction_score", 0),
            # Preserve accumulated whale history for Tier 1 dedup
            "whales_seen": (
                existing.get("whales_seen", set())
                | set(rec.get("supporting_whales", []))
            ),
        }


def _days_ago(date_str: str, now: datetime) -> float:
    """Return how many days ago date_str (YYYY-MM-DD) was, or 999 if unparseable."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return (now - d).days
    except (ValueError, TypeError):
        return 999
