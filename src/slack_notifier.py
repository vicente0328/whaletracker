"""
slack_notifier.py
-----------------
Sends Whale alert messages to Slack using Block Kit for rich formatting.

Required environment variables:
  SLACK_BOT_TOKEN     — xoxb-... (Bot User OAuth Token)
  SLACK_ALERT_CHANNEL — e.g. #whale_tracker_alerts

Alert types:
  send_strong_buy_alert()     → New STRONG BUY signal (score ≥ threshold)
  send_tier1_entry_alert()    → Tier 1 whale takes a new position
  send_watchlist_alert()      → Watched ticker gets a significant signal change
  send_insider_cluster_alert()→ Multiple insiders selling the same stock
  send_daily_digest()         → Morning summary of top recommendations
  send_rebalancing_digest()   → Portfolio rebalancing summary
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SLACK_BOT_TOKEN:    str = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_ALERT_CHANNEL: str = os.getenv("SLACK_ALERT_CHANNEL", "#whale_tracker_alerts")

_client = None


def _get_client():
    global _client
    if _client is None:
        from slack_sdk import WebClient  # noqa: PLC0415
        if not SLACK_BOT_TOKEN:
            raise EnvironmentError(
                "SLACK_BOT_TOKEN is not set. Add it to your .env or Railway environment."
            )
        _client = WebClient(token=SLACK_BOT_TOKEN)
    return _client


def _post(blocks: list, text: str, channel: str | None = None) -> bool:
    """Send a Block Kit message. Falls back to plain text if blocks fail."""
    ch = channel or SLACK_ALERT_CHANNEL
    try:
        _get_client().chat_postMessage(
            channel=ch, text=text, blocks=blocks, mrkdwn=True
        )
        logger.info("Slack alert sent to %s", ch)
        return True
    except Exception as exc:
        logger.error("Slack send failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Public alert functions
# ---------------------------------------------------------------------------

def send_strong_buy_alert(
    rec: dict[str, Any],
    channel: str | None = None,
) -> bool:
    """Fire when a ticker reaches STRONG BUY for the first time this cycle."""
    ticker  = rec["ticker"]
    company = rec.get("company", "")
    score   = rec.get("conviction_score", 0)
    whales  = ", ".join(rec.get("supporting_whales", []))
    signals = rec.get("signal_summary", "")
    tiers   = rec.get("whale_tiers", {})

    tier_line = ""
    if tiers:
        tier_parts = [f"{w}: {t}" for w, t in tiers.items()]
        tier_line = "  ·  ".join(tier_parts[:3])

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": f"🚀 STRONG BUY — {ticker}", "emoji": True}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Company*\n{company or '—'}"},
            {"type": "mrkdwn", "text": f"*Conviction Score*\n{score} / 12"},
            {"type": "mrkdwn", "text": f"*Signals*\n{signals or '—'}"},
            {"type": "mrkdwn", "text": f"*Whales*\n{whales or '—'}"},
        ]},
    ]
    if tier_line:
        blocks.append({"type": "context", "elements": [
            {"type": "mrkdwn", "text": f"🏷️ *Tier:* {tier_line}"}
        ]})
    blocks.append({"type": "divider"})

    text = f"🚀 STRONG BUY — {ticker} ({company}) | Score: {score}/12 | {signals}"
    return _post(blocks, text, channel)


def send_tier1_entry_alert(
    whale_name: str,
    tier_label: str,
    ticker: str,
    company: str,
    signal: str,
    score: int,
    channel: str | None = None,
) -> bool:
    """Fire when a Tier 1 whale takes a brand-new position (NEW_ENTRY)."""
    signal_labels = {
        "NEW_ENTRY":      "NEW ENTRY",
        "AGGRESSIVE_BUY": "AGGRESSIVE BUY",
        "HIGH_CONCENTRATION": "HIGH CONCENTRATION",
    }
    sig_label = signal_labels.get(signal, signal)

    blocks = [
        {"type": "header", "text": {"type": "plain_text",
            "text": f"🐋 Tier 1 Whale Alert — {ticker}", "emoji": True}},
        {"type": "section", "text": {"type": "mrkdwn",
            "text": (
                f"*{whale_name}* `{tier_label}` just filed a "
                f"*{sig_label}* for *{ticker}*"
                + (f" ({company})" if company else "")
            )}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Signal*\n{sig_label}"},
            {"type": "mrkdwn", "text": f"*Current Score*\n{score} / 12"},
        ]},
        {"type": "context", "elements": [
            {"type": "mrkdwn",
             "text": "ℹ️ Tier 1 = long-term value / macro investors with verified long-term returns."}
        ]},
        {"type": "divider"},
    ]
    text = f"🐋 Tier 1 Alert: {whale_name} ({tier_label}) → {sig_label} {ticker}"
    return _post(blocks, text, channel)


def send_watchlist_alert(
    rec: dict[str, Any],
    old_score: int = 0,
    channel: str | None = None,
) -> bool:
    """Fire when a watched ticker has a meaningful score change."""
    ticker  = rec["ticker"]
    score   = rec.get("conviction_score", 0)
    rec_str = rec.get("recommendation", "HOLD")
    signals = rec.get("signal_summary", "")
    delta   = score - old_score
    arrow   = "⬆️" if delta > 0 else "⬇️"

    rec_icon = {"STRONG BUY": "🚀", "BUY": "✅", "HOLD": "⏸️", "SELL": "🔻"}.get(rec_str, "")

    blocks = [
        {"type": "header", "text": {"type": "plain_text",
            "text": f"⭐ Watchlist Update — {ticker}", "emoji": True}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Recommendation*\n{rec_icon} {rec_str}"},
            {"type": "mrkdwn", "text": f"*Score Change*\n{arrow} {old_score} → {score}"},
            {"type": "mrkdwn", "text": f"*Signals*\n{signals or '—'}"},
        ]},
        {"type": "divider"},
    ]
    text = f"⭐ Watchlist [{ticker}]: {rec_str} | Score {old_score}→{score} | {signals}"
    return _post(blocks, text, channel)


def send_insider_cluster_alert(
    ticker: str,
    company: str,
    insider_count: int,
    total_value_usd: float,
    channel: str | None = None,
) -> bool:
    """Fire when 3+ insiders at the same company sell within 30 days."""
    val_str = f"${total_value_usd / 1e6:.1f}M" if total_value_usd >= 1e6 else f"${total_value_usd:,.0f}"
    blocks = [
        {"type": "header", "text": {"type": "plain_text",
            "text": f"⚠️ Insider Sell Cluster — {ticker}", "emoji": True}},
        {"type": "section", "text": {"type": "mrkdwn",
            "text": (
                f"*{insider_count} insiders* at *{company or ticker}* have sold "
                f"in the past 30 days — total value: *{val_str}*.\n"
                f"_Cluster selling is a stronger bearish signal than single-insider moves._"
            )}},
        {"type": "divider"},
    ]
    text = f"⚠️ Insider Cluster [{ticker}]: {insider_count} insiders sold {val_str} in 30d"
    return _post(blocks, text, channel)


def send_daily_digest(
    top_recs: list[dict[str, Any]],
    rebalancing: list[dict[str, Any]] | None = None,
    channel: str | None = None,
) -> bool:
    """Morning digest: top 5 STRONG BUY / BUY recommendations + rebalancing actions."""
    if not top_recs:
        return False

    from datetime import datetime  # noqa: PLC0415
    today = datetime.utcnow().strftime("%b %d, %Y")

    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text",
            "text": f"🐋 WhaleTracker Daily Digest — {today}", "emoji": True}},
        {"type": "section", "text": {"type": "mrkdwn",
            "text": "*Top institutional signals for today:*"}},
    ]

    icons = {"STRONG BUY": "🚀", "BUY": "✅", "HOLD": "⏸️", "SELL": "🔻"}
    for rec in top_recs[:5]:
        ticker = rec["ticker"]
        score  = rec.get("conviction_score", 0)
        rstr   = rec.get("recommendation", "HOLD")
        sigs   = rec.get("signal_summary", "")
        icon   = icons.get(rstr, "")
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": f"{icon} *{ticker}* — {rstr}  `{score}/12`\n_{sigs}_"}})

    if rebalancing:
        blocks.append({"type": "divider"})
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": "*Portfolio rebalancing actions:*"}})
        for s in rebalancing[:4]:
            arrow = "↑" if s["action"] == "INCREASE" else "↓"
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": f"{arrow} *{s['sector']}*  {s['current_weight']:.0%} → {s['target_weight']:.0%}  |  {s['rationale']}"}})

    blocks.append({"type": "divider"})
    blocks.append({"type": "context", "elements": [
        {"type": "mrkdwn", "text": "📊 WhaleTracker AI  ·  SEC 13F / 13D/G / Form 4 / N-PORT"}
    ]})

    text = f"🐋 WhaleTracker Daily Digest — {today} | {len(top_recs)} signals"
    return _post(blocks, text, channel)


def send_rebalancing_digest(
    suggestions: list[dict[str, Any]],
    channel: str | None = None,
) -> bool:
    """Post a portfolio rebalancing digest."""
    if not suggestions:
        return False

    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text",
            "text": "📊 Portfolio Rebalancing Digest", "emoji": True}},
    ]
    for s in suggestions:
        arrow = "↑" if s["action"] == "INCREASE" else "↓"
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": f"{arrow} *{s['sector']}*  {s['current_weight']:.0%} → {s['target_weight']:.0%}  |  {s['rationale']}"}})
    blocks.append({"type": "divider"})

    text = " | ".join(
        f"{'↑' if s['action']=='INCREASE' else '↓'} {s['sector']}"
        for s in suggestions
    )
    return _post(blocks, f"📊 Rebalancing: {text}", channel)


# Legacy alias kept for backward compatibility
def send_whale_alert(recommendation: dict[str, Any], channel: str | None = None) -> bool:
    return send_strong_buy_alert(recommendation, channel)
