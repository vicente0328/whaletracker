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
_INSIDER_CLUSTER_MIN  = 3       # insiders that must sell within window
_INSIDER_CLUSTER_DAYS = 30

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
    if FORM4_WATCH_TICKERS:
        _scheduler.add_job(
            _realtime_form4_job,
            trigger="interval",
            minutes=FORM4_REFRESH_MINUTES,
            id="form4_realtime",
            replace_existing=True,
        )
        logger.info(
            "Form 4 real-time polling — %d tickers, every %d min",
            len(FORM4_WATCH_TICKERS), FORM4_REFRESH_MINUTES,
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
    _check_watchlist_alerts(new_recs)
    _check_insider_cluster_alerts(insiders)

    # Refresh snapshot
    _seed_snapshot(new_recs)
    logger.info("[scheduler] Refresh complete — %d recommendations", len(new_recs))


def _realtime_form4_job() -> None:
    """Poll EDGAR for new Form 4 filings on watched tickers and fire per-transaction alerts."""
    if not FORM4_WATCH_TICKERS:
        return

    logger.info("[scheduler] Form 4 poll — %d tickers", len(FORM4_WATCH_TICKERS))
    try:
        from src.data_collector import fetch_recent_form4_filings  # noqa: PLC0415
        from src.slack_notifier import send_form4_realtime_alert   # noqa: PLC0415
    except Exception as exc:
        logger.error("[scheduler] Form 4 import error: %s", exc)
        return

    try:
        txs = fetch_recent_form4_filings(FORM4_WATCH_TICKERS, hours_back=2)
    except Exception as exc:
        logger.error("[scheduler] Form 4 fetch error: %s", exc)
        return

    new_alerts = 0
    for tx in txs:
        acc = tx.get("accession", "")
        if not acc or acc in _form4_seen_accessions:
            continue
        _form4_seen_accessions.add(acc)

        signal = tx.get("signal", "")
        # Alert on buys always; alert on sells only if not pre-planned
        if signal not in {"INSIDER_BUY", "INSIDER_SELL", "PLANNED_SELL"}:
            continue

        ticker = tx.get("ticker", "")
        try:
            send_form4_realtime_alert(
                ticker    = ticker,
                company   = tx.get("company", ""),
                insider   = tx.get("insider", "Insider"),
                role      = tx.get("role", ""),
                signal    = signal,
                shares    = tx.get("shares", 0),
                value_usd = tx.get("value_usd", 0.0),
                is_10b51  = tx.get("is_10b51", False),
            )
            new_alerts += 1
        except Exception as exc:
            logger.error("[scheduler] Form 4 alert send failed (%s): %s", ticker, exc)

    logger.info("[scheduler] Form 4 poll complete — %d new alerts", new_alerts)


def _read_news_settings() -> dict:
    """Return the full daily-news settings dict from daily_news_sub.json."""
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


def _daily_news_job() -> None:
    """Send the day's top financial news headline to Slack if subscription is ON
    and the current UTC hour matches the user-configured delivery hour."""
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
        from src.news_collector import fetch_market_news, item_matches_topics  # noqa: PLC0415
        from src.slack_notifier import send_daily_news_alert                   # noqa: PLC0415
    except Exception as exc:
        logger.error("[scheduler] Daily news import error: %s", exc)
        return

    try:
        topics = settings.get("topics") or []
        # Fetch a larger pool to find a topic-matching article
        candidates = fetch_market_news(20)
        if topics:
            candidates = [n for n in candidates if item_matches_topics(n["headline"], topics)]

        if candidates:
            send_daily_news_alert(candidates[0])
            logger.info(
                "[scheduler] Daily news sent (topics=%s): %s",
                topics or "all", candidates[0].get("headline", "")[:60],
            )
        else:
            logger.info("[scheduler] No matching news for topics %s — skipping", topics)
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
        try:
            send_strong_buy_alert(rec)
        except Exception as exc:
            logger.error("[scheduler] Alert send failed (%s): %s", ticker, exc)


def _check_tier1_alerts(
    filings: dict,
    whale_tiers: dict,
    new_recs: list[dict],
) -> None:
    """Fire send_tier1_entry_alert() for any new Tier 1 NEW_ENTRY signal."""
    from src.slack_notifier import send_tier1_entry_alert  # noqa: PLC0415

    tier1_whales = {
        name for name, info in whale_tiers.items() if info.get("tier") == 1
    }
    rec_map = {r["ticker"]: r for r in new_recs}

    for whale_name in tier1_whales:
        holdings = filings.get(whale_name, [])
        tier_info = whale_tiers.get(whale_name, {})
        tier_label = tier_info.get("label", "T1")

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

            logger.info("[scheduler] Tier 1 entry: %s → %s", whale_name, ticker)
            try:
                send_tier1_entry_alert(
                    whale_name=whale_name,
                    tier_label=tier_label,
                    ticker=ticker,
                    company=company,
                    signal="NEW_ENTRY",
                    score=score,
                )
            except Exception as exc:
                logger.error("[scheduler] T1 alert failed (%s): %s", ticker, exc)


def _check_watchlist_alerts(new_recs: list[dict]) -> None:
    """Fire send_watchlist_alert() when a watched ticker changes score by ≥ ALERT_MIN_SCORE."""
    if not ALERT_WATCHLIST:
        return

    from src.slack_notifier import send_watchlist_alert  # noqa: PLC0415

    for rec in new_recs:
        ticker = rec["ticker"]
        if ticker not in ALERT_WATCHLIST:
            continue
        prev_score = _snapshot.get(ticker, {}).get("conviction_score", 0)
        curr_score = rec.get("conviction_score", 0)
        if abs(curr_score - prev_score) < ALERT_MIN_SCORE:
            continue

        logger.info(
            "[scheduler] Watchlist alert: %s score %d→%d",
            ticker, prev_score, curr_score,
        )
        try:
            send_watchlist_alert(rec, old_score=prev_score)
        except Exception as exc:
            logger.error("[scheduler] Watchlist alert failed (%s): %s", ticker, exc)


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
