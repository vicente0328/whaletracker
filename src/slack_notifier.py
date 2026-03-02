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
    news_items: list[dict[str, Any]] | None = None,
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

    # ── Related news ──────────────────────────────────────────────────────────
    if news_items:
        blocks.append({"type": "divider"})
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": f"*📰 {ticker} 관련 최신 뉴스:*"}})
        for item in news_items[:3]:
            url  = item.get("url", "")
            head = item.get("headline", "")
            src  = item.get("source", "")
            line = f"• <{url}|{head}>" if url else f"• {head}"
            if src:
                line += f"  _{src}_"
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": line}})

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
    news_items: list[dict[str, Any]] | None = None,
    ai_context: str | None = None,
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
    ]

    # ── AI investment context ─────────────────────────────────────────────────
    if ai_context:
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": f"🤖 *AI 투자 시사점*\n{ai_context}"}})

    blocks.append({"type": "context", "elements": [
        {"type": "mrkdwn",
         "text": "ℹ️ Tier 1 = long-term value / macro investors with verified long-term returns."}
    ]})

    # ── Related news ──────────────────────────────────────────────────────────
    if news_items:
        blocks.append({"type": "divider"})
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": f"*📰 {ticker} 관련 최신 뉴스:*"}})
        for item in news_items[:3]:
            url  = item.get("url", "")
            head = item.get("headline", "")
            src  = item.get("source", "")
            line = f"• <{url}|{head}>" if url else f"• {head}"
            if src:
                line += f"  _{src}_"
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": line}})

    blocks.append({"type": "divider"})
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


def send_form4_realtime_alert(
    ticker: str,
    company: str,
    insider: str,
    role: str,
    signal: str,
    shares: int,
    value_usd: float,
    is_10b51: bool = False,
    news_items: list[dict[str, Any]] | None = None,
    channel: str | None = None,
) -> bool:
    """Near real-time Form 4 alert for a single insider transaction on a watched ticker."""
    sig_icon = "🟢" if signal == "INSIDER_BUY" else ("⚪" if is_10b51 else "🔴")
    sig_label = (
        "Insider BUY" if signal == "INSIDER_BUY"
        else ("10b5-1 Pre-planned Sale" if is_10b51 else "Insider SELL")
    )
    val_str  = f"${value_usd / 1e6:.2f}M" if value_usd >= 1_000_000 else f"${value_usd:,.0f}"
    shr_str  = f"{shares:,}"

    blocks = [
        {"type": "header", "text": {"type": "plain_text",
            "text": f"⚡ Real-Time Insider Alert — {ticker}", "emoji": True}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Company*\n{company or ticker}"},
            {"type": "mrkdwn", "text": f"*Signal*\n{sig_icon} {sig_label}"},
            {"type": "mrkdwn", "text": f"*Insider*\n{insider}"},
            {"type": "mrkdwn", "text": f"*Role*\n{role}"},
            {"type": "mrkdwn", "text": f"*Shares*\n{shr_str}"},
            {"type": "mrkdwn", "text": f"*Value*\n{val_str}"},
        ]},
        {"type": "context", "elements": [
            {"type": "mrkdwn",
             "text": ("_Pre-planned 10b5-1 sale — lower bearish weight._"
                      if is_10b51 else
                      "_Form 4 filing detected within 2 hours of EDGAR submission._")},
        ]},
    ]

    # ── Related news ──────────────────────────────────────────────────────────
    if news_items:
        blocks.append({"type": "divider"})
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": f"*📰 {ticker} 관련 최신 뉴스:*"}})
        for item in news_items[:3]:
            url  = item.get("url", "")
            head = item.get("headline", "")
            src  = item.get("source", "")
            line = f"• <{url}|{head}>" if url else f"• {head}"
            if src:
                line += f"  _{src}_"
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": line}})

    blocks.append({"type": "divider"})
    text = f"⚡ {sig_icon} {ticker} — {sig_label} | {insider} ({role}) | {val_str}"
    return _post(blocks, text, channel)


def send_daily_digest(
    top_recs: list[dict[str, Any]],
    rebalancing: list[dict[str, Any]] | None = None,
    news: list[dict[str, Any]] | None = None,
    channel: str | None = None,
) -> bool:
    """Morning digest: top 5 STRONG BUY / BUY recommendations + optional news."""
    if not top_recs:
        return False

    from datetime import datetime  # noqa: PLC0415
    today = datetime.utcnow().strftime("%b %d, %Y")

    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text",
            "text": f"🐋 WhaleTracker Daily Digest — {today}", "emoji": True}},
    ]

    # ── Market headlines (optional) ──────────────────────────────────────────
    if news:
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": "*📰 Market Headlines:*"}})
        for item in news[:3]:
            src  = item.get("source", "")
            url  = item.get("url", "")
            head = item.get("headline", "")
            line = f"• <{url}|{head}>" if url else f"• {head}"
            if src:
                line += f"  _{src}_"
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": line}})
        blocks.append({"type": "divider"})

    # ── Top picks ────────────────────────────────────────────────────────────
    blocks.append({"type": "section", "text": {"type": "mrkdwn",
        "text": "*Top institutional signals for today:*"}})

    icons = {"STRONG BUY": "🚀", "BUY": "✅", "HOLD": "⏸️", "SELL": "🔻"}
    for rec in top_recs[:5]:
        ticker = rec["ticker"]
        score  = rec.get("conviction_score", 0)
        rstr   = rec.get("recommendation", "HOLD")
        sigs   = rec.get("signal_summary", "")
        icon   = icons.get(rstr, "")
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": f"{icon} *{ticker}* — {rstr}  `{score}/12`\n_{sigs}_"}})

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


def send_market_event_alert(
    event: dict,
    days_before: int,
    channel: str | None = None,
) -> bool:
    """Fire a Slack alert N days before a high-impact market event.

    Args:
        event:       Event dict from market_events.MARKET_EVENTS.
        days_before: How many days until the event (1, 7, …).
        channel:     Override Slack channel (default: SLACK_ALERT_CHANNEL).
    """
    from datetime import date  # noqa: PLC0415

    event_type   = event.get("type", "EVENT")
    title        = event.get("title", "")
    description  = event.get("description", "")
    event_date   = event.get("date", date.today())
    impact       = event.get("impact", "MEDIUM")

    # Localised type label + emoji
    type_meta = {
        "FED_MEETING":     ("🏛️", "연준 FOMC"),
        "CPI_RELEASE":     ("📊", "CPI 물가지수"),
        "JOBS_REPORT":     ("💼", "고용 지표"),
        "EARNINGS_SEASON": ("📈", "실적 시즌"),
        "ELECTION":        ("🗳️", "선거"),
        "DEBT_CEILING":    ("⚠️", "부채 한도"),
    }
    icon, type_label = type_meta.get(event_type, ("📅", "시장 이벤트"))

    impact_emoji = "🔴 HIGH IMPACT" if impact == "HIGH" else "🟡 MEDIUM IMPACT"

    # Urgency header
    if days_before == 1:
        urgency = "⚡ *내일 예정!*"
    elif days_before <= 3:
        urgency = f"🔔 *{days_before}일 후 예정*"
    else:
        urgency = f"📅 *{days_before}일 후 예정*"

    date_str = event_date.strftime("%Y년 %m월 %d일 (%a)")

    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text",
            "text": f"{icon} 시장 이벤트 알림 — {title}", "emoji": True}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*이벤트 유형*\n{type_label}"},
            {"type": "mrkdwn", "text": f"*예정일*\n{date_str}"},
            {"type": "mrkdwn", "text": f"*영향도*\n{impact_emoji}"},
            {"type": "mrkdwn", "text": f"*남은 기간*\n{urgency}"},
        ]},
        {"type": "section", "text": {"type": "mrkdwn",
            "text": f"*개요*\n{description}"}},
        {"type": "divider"},
        {"type": "context", "elements": [
            {"type": "mrkdwn",
             "text": "📊 WhaleTracker AI Market Events  ·  중요 이벤트 선제 대응"}
        ]},
    ]

    text = f"{icon} [{days_before}일 전] {title} — {date_str} | {impact_emoji}"
    return _post(blocks, text, channel)


def send_daily_news_alert(
    news_items: "dict[str, Any] | list[dict[str, Any]]",
    channel: str | None = None,
) -> bool:
    """Send institutional investor news articles to Slack with Korean summaries.

    Args:
        news_items: Single article dict (legacy) or list of article dicts.
                    Each dict may contain 'headline', 'source', 'url',
                    'published_at', and optionally 'summary_ko'.
        channel:    Override Slack channel.
    """
    from datetime import datetime  # noqa: PLC0415

    # Normalise: accept both a single dict and a list
    if isinstance(news_items, dict):
        items: list[dict] = [news_items]
    else:
        items = list(news_items)

    items = [it for it in items if it.get("headline")]
    if not items:
        return False

    today = datetime.utcnow().strftime("%b %d, %Y")

    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text",
            "text": f"🏛️ 기관투자자 동향 뉴스 — {today}", "emoji": True}},
        {"type": "section", "text": {"type": "mrkdwn",
            "text": "_헤지펀드·자산운용사 등 주요 기관투자자의 움직임과 시장 시사점_"}},
        {"type": "divider"},
    ]

    for idx, item in enumerate(items[:5], start=1):
        headline   = item.get("headline", "")
        source     = item.get("source", "")
        url        = item.get("url", "")
        pub        = item.get("published_at", "")
        summary_ko = item.get("summary_ko", "")

        head_text = f"<{url}|{headline}>" if url else headline
        body = f"*{idx}.* {head_text}"
        if summary_ko:
            body += f"\n📌 _{summary_ko}_"

        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": body}})

        meta_parts: list[str] = []
        if source:
            meta_parts.append(source)
        if pub:
            meta_parts.append(pub)
        if meta_parts:
            blocks.append({"type": "context", "elements": [
                {"type": "mrkdwn", "text": "  ·  ".join(meta_parts)}
            ]})

        if idx < len(items[:5]):
            blocks.append({"type": "divider"})

    blocks.append({"type": "divider"})
    blocks.append({"type": "context", "elements": [
        {"type": "mrkdwn",
         "text": "📊 WhaleTracker AI  ·  기관투자자 동향 뉴스  ·  구독 취소: 대시보드 뉴스 섹션에서 토글 OFF"}
    ]})

    first_headline = items[0].get("headline", "")
    text = f"🏛️ 기관투자자 동향 뉴스 ({today}): {first_headline}"
    return _post(blocks, text, channel)


def send_activist_13d_alert(
    ticker: str,
    company: str,
    filer: str,
    ownership_pct: float,
    ai_context: str | None = None,
    news_items: list[dict[str, Any]] | None = None,
    channel: str | None = None,
) -> bool:
    """Real-time alert when an activist investor files SC 13D (≥5% with intent to influence)."""
    pct_str = f"{ownership_pct:.1f}%" if ownership_pct else "5%+"

    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text",
            "text": f"🔔 행동주의 투자자 13D — {ticker}", "emoji": True}},
        {"type": "section", "text": {"type": "mrkdwn",
            "text": (
                f"*{filer}*이 *{ticker}*"
                + (f" ({company})" if company else "")
                + f" 지분 *{pct_str}*를 취득하며 *SC 13D*를 제출했습니다.\n"
                "_13D는 경영 참여 의도가 있는 적극적 투자 신호입니다._"
            )}},
    ]

    if ai_context:
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": f"🤖 *AI 투자 시사점*\n{ai_context}"}})

    blocks.append({"type": "context", "elements": [
        {"type": "mrkdwn",
         "text": "⚡ SC 13D = 행동주의 캠페인 · 경영진 교체 · 전략적 변화 요구 가능성"}
    ]})

    if news_items:
        blocks.append({"type": "divider"})
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": f"*📰 {ticker} 관련 최신 뉴스:*"}})
        for item in news_items[:3]:
            url  = item.get("url", "")
            head = item.get("headline", "")
            src  = item.get("source", "")
            line = f"• <{url}|{head}>" if url else f"• {head}"
            if src:
                line += f"  _{src}_"
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": line}})

    blocks.append({"type": "divider"})
    text = f"🔔 13D Alert: {filer} → {ticker} {pct_str} 취득"
    return _post(blocks, text, channel)


def send_batch_digest(
    items: list[dict[str, Any]],
    channel: str | None = None,
) -> bool:
    """Daily batch digest for lower-urgency signals: watchlist changes, 13G filings.

    items: list of {"type": "watchlist"|"13g", "data": dict, ...}
    """
    if not items:
        return False

    from datetime import datetime  # noqa: PLC0415
    today = datetime.utcnow().strftime("%b %d, %Y")

    watchlist_items = [i for i in items if i.get("type") == "watchlist"]
    passive_13g     = [i for i in items if i.get("type") == "13g"]

    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text",
            "text": f"📋 일일 배치 요약 — {today}", "emoji": True}},
        {"type": "section", "text": {"type": "mrkdwn",
            "text": "_긴급도 낮은 신호 묶음 (워치리스트 변경 · 패시브 13G 공시)_"}},
        {"type": "divider"},
    ]

    if watchlist_items:
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": f"*⭐ 워치리스트 점수 변경 ({len(watchlist_items)}건):*"}})
        icons = {"STRONG BUY": "🚀", "BUY": "✅", "HOLD": "⏸️", "SELL": "🔻"}
        for item in watchlist_items[:8]:
            rec       = item.get("data", {})
            old_score = item.get("old_score", 0)
            ticker    = rec.get("ticker", "")
            score     = rec.get("conviction_score", 0)
            rstr      = rec.get("recommendation", "HOLD")
            arrow     = "⬆️" if score > old_score else "⬇️"
            icon      = icons.get(rstr, "")
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": f"{arrow} *{ticker}* {icon} {rstr}  `{old_score}→{score}/12`"}})

    if passive_13g and watchlist_items:
        blocks.append({"type": "divider"})

    if passive_13g:
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": f"*📄 패시브 13G 공시 ({len(passive_13g)}건):*"}})
        for item in passive_13g[:6]:
            filing = item.get("data", {})
            ticker = filing.get("ticker", "")
            filer  = filing.get("filer", "")
            pct    = filing.get("ownership_pct", 0)
            pct_str = f"{pct:.1f}%" if pct else "5%+"
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": f"• *{ticker}*  {filer}  {pct_str} 취득 (단순 투자)"}})

    blocks.append({"type": "divider"})
    blocks.append({"type": "context", "elements": [
        {"type": "mrkdwn", "text": "📊 WhaleTracker AI  ·  배치 다이제스트 (낮은 긴급도 신호)"}
    ]})

    text = f"📋 일일 배치 요약 — {len(items)}건 ({today})"
    return _post(blocks, text, channel)


def send_premarket_briefing(
    top_recs: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
    news_items: list[dict[str, Any]] | None = None,
    channel: str | None = None,
) -> bool:
    """Pre-market briefing sent before market open (default 06:00 KST / 21:00 UTC).

    Combines top institutional signals, upcoming market events, and
    institutional investor news into a single structured morning brief.
    """
    from datetime import datetime, date  # noqa: PLC0415

    today_str = datetime.utcnow().strftime("%Y년 %m월 %d일")
    weekday   = ["월", "화", "수", "목", "금", "토", "일"][date.today().weekday()]

    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text",
            "text": f"🌅 프리마켓 브리핑 — {today_str} ({weekday})", "emoji": True}},
    ]

    # ── Upcoming events (next 7 days) ─────────────────────────────────────────
    if events:
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": "*📅 향후 7일 주요 이벤트:*"}})
        type_labels = {
            "FED_MEETING":     ("🏛️", "FOMC"),
            "CPI_RELEASE":     ("📊", "CPI"),
            "JOBS_REPORT":     ("💼", "NFP"),
            "EARNINGS_SEASON": ("📈", "실적시즌"),
            "ELECTION":        ("🗳️", "선거"),
            "DEBT_CEILING":    ("⚠️", "부채한도"),
        }
        for ev in events[:5]:
            ev_date  = ev.get("date")
            ev_type  = ev.get("type", "")
            title    = ev.get("title", "")
            days_left = (ev_date - date.today()).days if ev_date else 0
            icon, label = type_labels.get(ev_type, ("📅", "이벤트"))
            urgency = "⚡ 오늘!" if days_left == 0 else (f"D-{days_left}" if days_left > 0 else "")
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": f"{icon} *{title}*  `{label}`  _{urgency}_"}})
        blocks.append({"type": "divider"})

    # ── Top institutional signals ──────────────────────────────────────────────
    if top_recs:
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": "*🐋 오늘의 주요 기관 신호:*"}})
        icons_map = {"STRONG BUY": "🚀", "BUY": "✅", "HOLD": "⏸️", "SELL": "🔻"}
        for rec in top_recs[:5]:
            ticker = rec.get("ticker", "")
            score  = rec.get("conviction_score", 0)
            rstr   = rec.get("recommendation", "HOLD")
            sigs   = rec.get("signal_summary", "")
            whales = ", ".join(rec.get("supporting_whales", [])[:2])
            icon   = icons_map.get(rstr, "")
            line   = f"{icon} *{ticker}*  {rstr}  `{score}/12`"
            if whales:
                line += f"  — _{whales}_"
            if sigs:
                line += f"\n  _{sigs}_"
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": line}})
        blocks.append({"type": "divider"})

    # ── Institutional news headlines ───────────────────────────────────────────
    if news_items:
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
            "text": "*🏛️ 기관투자자 동향 뉴스:*"}})
        for item in news_items[:3]:
            url  = item.get("url", "")
            head = item.get("headline", "")
            src  = item.get("source", "")
            summary_ko = item.get("summary_ko", "")
            line = f"• <{url}|{head}>" if url else f"• {head}"
            if src:
                line += f"  _{src}_"
            if summary_ko:
                line += f"\n  📌 _{summary_ko}_"
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": line}})

    blocks.append({"type": "context", "elements": [
        {"type": "mrkdwn",
         "text": "📊 WhaleTracker AI  ·  프리마켓 브리핑  ·  장 시작 전 기관 동향 요약"}
    ]})

    text = f"🌅 프리마켓 브리핑 {today_str} | 신호 {len(top_recs)}건 · 이벤트 {len(events or [])}건"
    return _post(blocks, text, channel)


# Legacy alias kept for backward compatibility
def send_whale_alert(recommendation: dict[str, Any], channel: str | None = None) -> bool:
    return send_strong_buy_alert(recommendation, channel)
